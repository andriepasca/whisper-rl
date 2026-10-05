import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import dataclasses
import os

from whisper_env import (
    OffshoreMaintenanceEnv,
    EnvironmentConfig,
    WindClimateConfig,
    LayoutConfig,
    DamageSolverConfig,
    MaintenanceType,
    MaintenanceConfig,
    RewardObjective,
    RewardConfig,
    MaintenancePolicy,
    DamageSolver,
    EnergySolver,
    WindClimate,
    TransitionModel,
    SpatialGroupingObjective,
    RewardModel,
    LoggingModel,
    RandomScatteredLayout,
)
from whisper_env.config import DEFAULT_DATA_DIR, make_fixed_damage_solver_config

def run_validation(damage_csv, design_life):
    layout = RandomScatteredLayout(n_turbines=4, seed=42)
    layout_config = LayoutConfig(x=layout.x, y=layout.y)

    # NOTE: fixed DEL refs were calibrated for the north_sea monthly Weibull (TI=0.10),
    # but this validation uses a custom climate (Weibull c=8, k=2). Behavior kept as-is.
    print("NOTE: using fixed north_sea-calibrated DEL refs with custom Weibull(c=8,k=2) climate.")
    damage_config = dataclasses.replace(
        make_fixed_damage_solver_config(damage_csv),
        design_life_years=design_life,
    )
    damage_solver = DamageSolver(damage_config)

    weibull = {m: {"c": 8.0, "k": 2.0} for m in range(1, 13)}
    wind_config = WindClimateConfig(monthly_weibull=weibull)
    wind_climate = WindClimate(wind_config)

    maintenance_cfg = MaintenanceConfig(
        maintenance_types=(
            MaintenanceType(
                name="repair", threshold=0.8, downtime_hours=24,
                damage_multiplier=0.7, duration_months=12
            ),
        )
    )
    maintenance_policy = MaintenancePolicy(maintenance_cfg)

    transition_model = TransitionModel()

    class DummyRewardModel:
        def solve(self, metrics):
            from whisper_env import RewardResult
            return RewardResult(reward=0.0, objectives=metrics)

    class DummySpatialGrouping:
        n_turbines = layout_config.x.shape[0]
        def solve(self, indices): return 0.0

    # 25 years simulation (300 months)
    # randomize_initial_hi=False: the baseline starts every turbine at initial_hi=1.0
    # (EnvironmentConfig defaults to randomized initial HI, which would defeat this check).
    config = EnvironmentConfig(
        wind_climate=wind_climate,
        damage_solver=damage_solver,
        maintenance_policy=maintenance_policy,
        transition_model=transition_model,
        spatial_grouping_objective=DummySpatialGrouping(),
        reward_model=DummyRewardModel(),
        energy_solver=EnergySolver(),
        logger=LoggingModel(),
        max_protection_duration=12,
        randomize_initial_hi=False,
        initial_hi=1.0,
        wind_time_slices_per_month=5,
        decision_interval_months=1,
        max_simulation_years=25
    )

    env = OffshoreMaintenanceEnv(config)
    obs, info = env.reset(seed=42)

    hi_history = []

    print("Running baseline validation (always continue)...")
    for step in range(300): # 25 years * 12 months
        action = np.zeros(env.n_turbines, dtype=np.int32)  # MultiDiscrete([2]*n): 0 = continue
        obs, reward, terminated, truncated, info = env.step(action)
        hi_history.append(obs["turbines"]["HI"].copy())

        if terminated or truncated:
            break

    hi_history = np.array(hi_history)

    plt.figure(figsize=(10, 6))
    for i in range(env.n_turbines):
        plt.plot(hi_history[:, i], label=f'Turbine {i+1}', alpha=0.8)

    plt.title('Baseline Validation: Health Index (HI) over 25 Years\nUsing Stratified Sampling (No Jensen Bias)')
    plt.xlabel('Months')
    plt.ylabel('Health Index (HI)')
    plt.legend()
    plt.grid(True)

    output_path = 'baseline_validation.png'
    plt.savefig(output_path)
    print(f"Validation complete. Saved plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline Validation")
    parser.add_argument("--damage-csv", type=str, default=os.path.join(DEFAULT_DATA_DIR, 'response_surface_200.csv'), help="Path to the response surface damage data")
    parser.add_argument("--design-life", type=float, default=20.0, help="Design life in years (Default: 20)")

    args = parser.parse_args()

    damage_csv = os.path.abspath(args.damage_csv)
    if not os.path.isfile(damage_csv):
        raise SystemExit(f"ERROR: damage CSV not found: {damage_csv} (check path; beware doubled '.csv.csv')")

    run_validation(
        damage_csv=damage_csv,
        design_life=args.design_life
    )
