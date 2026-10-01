import argparse
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from whisper_env.environment import OffshoreMaintenanceEnv
from whisper_env.config import get_default_config

class ElectricityPriceModel:
    """
    Models the stochastic electricity price for energy revenue calculations.
    Re-implemented here for post-hoc evaluation after decoupling from the core RL environment.
    """

    def __init__(
        self,
        monthly_mean=(90, 85, 78, 70, 62, 55, 50, 52, 60, 72, 82, 92),
        monthly_std=(10, 10, 9, 8, 8, 7, 7, 7, 8, 9, 10, 10),
    ):
        self.monthly_mean = np.asarray(monthly_mean, dtype=float)
        self.monthly_std = np.asarray(monthly_std, dtype=float)

    def sample(self, month, rng):
        """
        Samples the electricity price for a given month.

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

class DummyAgent:
    """Fallback agent taking random valid actions when no trained model is available."""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=True):
        # We ignore deterministic flag and sample randomly
        return self.action_space.sample(), None

def load_agent(model_path, env):
    """Loads a PPO model or falls back to a DummyAgent if not found."""
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        return PPO.load(model_path, env=env)

    fallback_path = model_path.split("_seed")[0] + ".zip"
    if os.path.exists(fallback_path):
        print(f"Model {model_path} not found. Loading unseeded fallback model from {fallback_path}...")
        return PPO.load(fallback_path, env=env)

    print(f"Warning: No model found at {model_path} or {fallback_path}. Using DummyAgent with random actions.")
    return DummyAgent(env.action_space)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy post-hoc and calculate financial metrics.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for the simulation.")
    parser.add_argument("--w-damage", type=float, default=0.85, help="Weight for the damage burden objective.")
    parser.add_argument("--w-spatial", type=float, default=0.15, help="Weight for the spatial grouping objective.")
    parser.add_argument("--turbine-csv", type=str, default="./data/NREL_Reference_5MW_126.csv", help="Path to turbine CSV.")
    parser.add_argument("--damage-csv", type=str, default="./data/response_surface_df_200.csv", help="Path to damage response CSV.")
    args = parser.parse_args()

    # 1. Configuration & Environment Setup
    config = get_default_config(
        n_turbines=25,
        w_damage=args.w_damage,
        w_spatial=args.w_spatial,
        seed=args.seed,
        initial_hi=1.0,
    )
    env = OffshoreMaintenanceEnv(config=config)

    # 2. Model Loading
    model_path = f"models/ppo_blade_w{args.w_damage}_seed{args.seed}.zip"
    agent = load_agent(model_path, env)

    # 3. Post-Hoc Economic Setup
    price_model = ElectricityPriceModel()
    rng = np.random.default_rng(args.seed)

    repair_cost = 2000.0
    replacement_cost = 350000.0

    total_revenue = 0.0
    total_maintenance_cost = 0.0

    print("Starting deterministic 30-year evaluation simulation...")

    # 4. Simulation Loop
    obs, info = env.reset(seed=args.seed)
    # The initial month is 1 in the environment after reset

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        step_cost = 0.0
        step_energy_yield = 0.0

        # Access the environment directly for the unwrapped state variables needed
        env_unwrapped = env.unwrapped
        hours_per_month = env_unwrapped.config.hours_per_month

        maintenance_results = info.get("maintenance_results", [])

        # Calculate monthly metrics across all turbines
        for i, maintenance in enumerate(maintenance_results):
            availability = 1.0
            if maintenance.maintenance_type is not None:
                if maintenance.maintenance_type.is_replacement:
                    step_cost += replacement_cost
                else:
                    step_cost += repair_cost

                downtime_hours = maintenance.downtime_hours
                availability = max(0.0, (hours_per_month - downtime_hours) / hours_per_month)

            # Retrieve the average monthly power in Watts for this turbine from the environment state
            turbine_power_w = env_unwrapped.power[i]

            # Energy yield = Power (W) * Availability * Duration (hours)
            # Multiply by 1e-3 to get kWh or MWh depending on the downstream price expectation
            # Let's assume price is per unit of energy in MWh or similar, using raw power * hours here based on previous implicit logic
            turbine_energy = turbine_power_w * availability * hours_per_month
            step_energy_yield += turbine_energy

        # env.step increments the month, so we subtract 1 (handling wrap-around) to get the month that just ran
        sim_month = env_unwrapped.current_month - 1
        if sim_month == 0:
            sim_month = 12

        sampled_price = price_model.sample(month=sim_month, rng=rng)

        step_revenue = step_energy_yield * sampled_price

        total_revenue += step_revenue
        total_maintenance_cost += step_cost

    net_profit = total_revenue - total_maintenance_cost

    # 5. Output Results
    df_results = pd.DataFrame([{
        "Seed": args.seed,
        "Total Revenue": total_revenue,
        "Total Maintenance Cost": total_maintenance_cost,
        "Net Profit": net_profit
    }])

    print("\nSimulation Complete. Financial Summary:")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
