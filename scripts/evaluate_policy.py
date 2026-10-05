"""This script performs post-hoc economic and carbon emission analysis on policies trained in OffshoreMaintenanceEnv. OffshoreMaintenanceEnv itself remains strictly physical and MDP-based.

Energy/revenue accounting (reporting only):
  * One env.step() advances `decision_interval_months` months. For each month the env draws
    n_slices i.i.d. Weibull wind speeds, maps them through its hardcoded 5-point power curve
    and stores the monthly MEAN power (MW) per turbine, times availability (0 if HI<=0).
  * env.power only holds the LAST month of the interval, so the env records every month in
    env.interval_power_mw / env.interval_calendar_months (reporting-only lists; they are not
    read by observations, reward or training).
  * Energy (MWh) per turbine-month = power_MW * hours_per_month (730.5). Revenue = energy *
    price sampled for that calendar month from the local post-hoc ElectricityPriceModel.
  * Downtime: the env does not reduce power for maintenance downtime, so by default neither
    does this script; --downtime-costs-energy subtracts downtime_hours from the first month.
  * Carbon emissions: calculated post-hoc from spatial mobilization distance (spatial_grouping meters)
    multiplied by vessel fuel/carbon intensity factor (default 0.05 kg CO2 / m).
  * Energy depends on the hardcoded 5-point power curve (see
    research/issues/ISSUE-power-curve-and-damage-lookup.md): absolute revenue is
    reporting-only and must NOT be cited as NREL-based.
"""
import argparse
import dataclasses
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from whisper_env.environment import OffshoreMaintenanceEnv
from whisper_env.config import get_default_config, DamageSolverConfig, DEFAULT_DATA_DIR, make_fixed_damage_solver_config
from whisper_env.physics import DamageSolver
# Same observation wrapper as used for training (single source of truth).
from train_ppo_pareto import FlattenDictWrapper

class ElectricityPriceModel:
    """
    Models the stochastic electricity price for energy revenue calculations.
    Re-implemented here for post-hoc evaluation after decoupling from the core RL environment.
    POST-HOC ONLY: not used by the env or training.
    """

    def __init__(
        self,
        monthly_mean=(90, 85, 78, 70, 62, 55, 50, 52, 60, 72, 82, 92),
        monthly_std=(10, 10, 9, 8, 8, 7, 7, 7, 8, 9, 10, 10),
    ):
        self.monthly_mean = np.asarray(monthly_mean, dtype=float)
        self.monthly_std = np.asarray(monthly_std, dtype=float)

    def sample(self, month: int, rng: np.random.Generator) -> float:
        """
        Samples the electricity price ($/MWh) for a given month (1-12).

        Args:
            month (int): The current month (1-12).
            rng (np.random.Generator): Random number generator instance.

        Returns:
            float: The sampled electricity price.
        """
        mu = self.monthly_mean[month - 1]
        sigma = self.monthly_std[month - 1]
        price = rng.normal(mu, sigma)
        return max(price, 0.0)

class CarbonEmissionModel:
    """
    Models post-hoc carbon emissions derived from spatial mobilization distance.
    Re-implemented here for post-hoc evaluation after decoupling from the core RL environment.
    POST-HOC ONLY: not used by the env or training.

    Carbon emissions (kg CO2) = spatial_grouping_distance (m) * carbon_intensity_factor (kg CO2 / m)
    """

    def __init__(self, carbon_intensity_factor: float = 0.05):
        """
        Args:
            carbon_intensity_factor (float): Vessel fuel/carbon intensity factor (kg CO2 / m).
        """
        self.carbon_intensity_factor = carbon_intensity_factor

    def calculate_emissions(self, spatial_grouping_m: float) -> float:
        """
        Calculates carbon emissions in kg CO2 for a given spatial mobilization distance in meters.

        Args:
            spatial_grouping_m (float): Spatial grouping distance in meters.

        Returns:
            float: Carbon emissions in kg CO2.
        """
        return float(spatial_grouping_m) * self.carbon_intensity_factor

class DummyAgent:
    """Fallback agent taking random valid actions when no trained model is available."""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=True):
        # We ignore deterministic flag and sample randomly
        return self.action_space.sample(), None

def resolve_model_path(model_path, allow_dummy):
    """Returns an existing model path (seeded, then unseeded), or None if --allow-dummy.

    Raises FileNotFoundError when no model exists and allow_dummy is False.
    """
    if os.path.exists(model_path):
        return model_path

    fallback_path = model_path.split("_seed")[0] + ".zip"
    if os.path.exists(fallback_path):
        print(f"Model {model_path} not found. Using unseeded fallback model {fallback_path}.")
        return fallback_path

    if allow_dummy:
        print("!!! WARNING: no trained model found at "
              f"'{model_path}' or '{fallback_path}'; using RANDOM DummyAgent (--allow-dummy). "
              "Results are NOT from a trained policy. !!!")
        return None

    raise FileNotFoundError(
        f"Trained model not found: tried '{model_path}' and '{fallback_path}'. "
        "Use --allow-dummy to evaluate a random DummyAgent instead."
    )

def build_env(w_damage, w_spatial, seed, damage_csv, wrap):
    """Builds the evaluation environment (logging enabled; optional FlattenDictWrapper for PPO).

    The damage solver mirrors scripts/train_ppo_pareto.py and scripts/benchmark_pareto.py
    (fixed DEL references), so the evaluated dynamics match training.
    """
    config = get_default_config(w_damage=w_damage, w_spatial=w_spatial, seed=seed)
    damage_solver_config = make_fixed_damage_solver_config(damage_csv)
    config = dataclasses.replace(
        config,
        damage_solver=DamageSolver(damage_solver_config),
        enable_logging=True
    )
    env = OffshoreMaintenanceEnv(config=config)
    if wrap:
        env = FlattenDictWrapper(env)
    return env

def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy post-hoc and calculate financial and carbon metrics.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for the simulation.")
    parser.add_argument("--w-damage", type=float, default=0.85, help="Weight for the damage burden objective.")
    parser.add_argument("--w-spatial", type=float, default=0.15, help="Weight for the spatial grouping objective.")
    parser.add_argument("--damage-csv", type=str, default=os.path.join(DEFAULT_DATA_DIR, "response_surface_200.csv"), help="Path to damage response CSV.")
    parser.add_argument("--allow-dummy", action="store_true", help="Use a random DummyAgent if the PPO model is missing (loud warning).")
    parser.add_argument("--downtime-costs-energy", action="store_true", help="Post-hoc only: subtract maintenance downtime_hours from first-month energy of the interval. Default off, matching the env, which does not reduce power for maintenance downtime.")
    parser.add_argument("--vessel-carbon-factor", type=float, default=0.05, help="Post-hoc only: vessel carbon intensity factor in kg CO2 per meter of spatial mobilization distance.")
    args = parser.parse_args()

    damage_csv = os.path.abspath(args.damage_csv)
    if not os.path.isfile(damage_csv):
        raise SystemExit(f"ERROR: damage CSV not found: {damage_csv} (check path; beware doubled '.csv.csv')")
    print(f"Using damage CSV: {damage_csv}")

    # 1. Model resolution (raises FileNotFoundError unless --allow-dummy)
    model_path = f"models/ppo_blade_w{args.w_damage}_seed{args.seed}.zip"
    resolved_path = resolve_model_path(model_path, args.allow_dummy)
    agent_type = "ppo" if resolved_path is not None else "dummy"

    # 2. Configuration & Environment Setup (PPO needs the training-time FlattenDictWrapper)
    env = build_env(args.w_damage, args.w_spatial, args.seed, damage_csv, wrap=(agent_type == "ppo"))
    if agent_type == "ppo":
        print(f"Loading trained model from {resolved_path}...")
        agent = PPO.load(resolved_path, env=env)
    else:
        agent = DummyAgent(env.action_space)
    print(f"Agent type: {agent_type}")

    # 3. Post-Hoc Economic & Carbon Setup
    price_model = ElectricityPriceModel()
    carbon_model = CarbonEmissionModel(carbon_intensity_factor=args.vessel_carbon_factor)
    rng = np.random.default_rng(args.seed)

    repair_cost = 2000.0
    replacement_cost = 350000.0

    total_revenue = 0.0
    total_maintenance_cost = 0.0
    total_carbon_emissions = 0.0

    print("Starting deterministic 30-year evaluation simulation...")

    # 4. Simulation Loop
    obs, info = env.reset(seed=args.seed)

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = agent.predict(obs, deterministic=True)

        # Access the environment directly for the unwrapped state variables needed
        env_unwrapped = env.unwrapped
        hours_per_month = env_unwrapped.config.hours_per_month

        action_arr = np.asarray(action, dtype=np.int32)
        maintenance_results = [
            env_unwrapped.maintenance_policy.solve(
                hi=env_unwrapped.HI[i],
                action=action_arr[i],
                protection_remaining=env_unwrapped.protection_remaining[i],
            )
            for i in range(env_unwrapped.n_turbines)
        ]

        # Post-hoc carbon emission calculation from spatial mobilization distance
        maintenance_indices = [i for i, result in enumerate(maintenance_results) if result.maintenance_type is not None]
        spatial_grouping_m = float(env_unwrapped.spatial_grouping_objective.solve(maintenance_indices))
        step_carbon_emission = carbon_model.calculate_emissions(spatial_grouping_m)
        total_carbon_emissions += step_carbon_emission

        obs, reward, terminated, truncated, info = env.step(action)

        step_cost = 0.0
        for maintenance in maintenance_results:
            if maintenance.maintenance_type is not None:
                if maintenance.maintenance_type.is_replacement:
                    step_cost += replacement_cost
                else:
                    step_cost += repair_cost

        # Energy accounting: sum over every month actually simulated in this decision interval
        step_revenue = 0.0
        for m_idx, (cal_month, power_mw) in enumerate(
            zip(env_unwrapped.interval_calendar_months, env_unwrapped.interval_power_mw)
        ):
            month_energy_mwh = power_mw * hours_per_month  # per turbine, MWh
            if args.downtime_costs_energy and m_idx == 0:
                for i, maintenance in enumerate(maintenance_results):
                    if maintenance.maintenance_type is not None:
                        lost_h = min(float(maintenance.downtime_hours), hours_per_month)
                        month_energy_mwh[i] = power_mw[i] * (hours_per_month - lost_h)
            sampled_price = price_model.sample(month=cal_month, rng=rng)
            step_revenue += float(np.sum(month_energy_mwh)) * sampled_price

        total_revenue += step_revenue
        total_maintenance_cost += step_cost

    net_profit = total_revenue - total_maintenance_cost

    # 5. Output Results
    df_results = pd.DataFrame([{
        "Seed": args.seed,
        "Agent Type": agent_type,
        "Total Revenue": total_revenue,
        "Total Maintenance Cost": total_maintenance_cost,
        "Net Profit": net_profit,
        "Total Carbon Emissions (kg CO2)": total_carbon_emissions,
    }])

    print("\nSimulation Complete. Financial & Carbon Summary:")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
