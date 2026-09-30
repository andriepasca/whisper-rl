import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
import os

from whisper_env import (
    OffshoreMaintenanceEnv,
    EnvironmentConfig,
    WindClimateConfig,
    LayoutConfig,
    TurbineConfig,
    WakeSolverConfig,
    DamageSolverConfig,
    MaintenanceType,
    MaintenanceConfig,
    RewardObjective,
    RewardConfig,
    MaintenancePolicy,
    WakeSolver,
    DamageSolver,
    EnergySolver,
    WindClimate,
    TransitionModel,
    ElectricityPriceModel,
    SpatialGroupingObjective,
    RewardModel,
    LoggingModel,
    RandomScatteredLayout,
)
from py_wake.deficit_models import BastankhahGaussianDeficit
from py_wake.superposition_models import LinearSum
from py_wake.turbulence_models import STF2017TurbulenceModel

def run_validation(turbine_csv, damage_csv, design_life):
    layout = RandomScatteredLayout(n_turbines=4, seed=42)
    layout_config = LayoutConfig(x=layout.x, y=layout.y)

    turbine_config = TurbineConfig(
        name="NREL 5-MW",
        rotor_diameter=126.0,
        hub_height=90.0,
        csv_path=turbine_csv,
        power_unit="kW"
    )

    wake_config = WakeSolverConfig(
        layout=layout_config,
        turbine=turbine_config,
        wake_deficit_model=BastankhahGaussianDeficit(),
        superposition_model=LinearSum(),
        turbulence_model=STF2017TurbulenceModel()
    )
    wake_solver = WakeSolver(wake_config)

    damage_config = DamageSolverConfig(
        csv_path=damage_csv,
        u_column="u",
        ti_column="ti",
        del_flap_column="del_flap",
        del_edge_column="del_edge",
        m_coef=10.0,
        del_flap_ref=2803.716141751299,
        del_edge_ref=5588.786717232858,
        design_life_years=design_life
    )
    damage_solver = DamageSolver(damage_config)

    weibull = {m: {"c": 8.0, "k": 2.0} for m in range(1, 13)}
    wind_config = WindClimateConfig(
        monthly_weibull=weibull,
        wind_direction=np.arange(0, 360, 30),
        wind_direction_probability=np.ones(12)/12
    )
    wind_climate = WindClimate(wind_config)

    maintenance_cfg = MaintenanceConfig(
        maintenance_types=(
            MaintenanceType(
                name="repair", threshold=0.8, cost=50000, downtime_hours=24,
                carbon_emission=1000, damage_multiplier=0.7, duration_months=12
            ),
        )
    )
    maintenance_policy = MaintenancePolicy(maintenance_cfg)

    transition_model = TransitionModel()

    class DummyPriceModel:
        def sample(self, month, rng): return 50.0

    class DummyRewardModel:
        def solve(self, metrics):
            from whisper_env import RewardResult
            return RewardResult(reward=0.0, objectives=metrics)

    class DummySpatialGrouping:
        def solve(self, indices): return 0.0

    # 25 years simulation (300 months)
    config = EnvironmentConfig(
        wind_climate=wind_climate,
        wake_solver=wake_solver,
        damage_solver=damage_solver,
        maintenance_policy=maintenance_policy,
        transition_model=transition_model,
        electricity_price_model=DummyPriceModel(),
        spatial_grouping_objective=DummySpatialGrouping(),
        reward_model=DummyRewardModel(),
        energy_solver=EnergySolver(),
        logger=LoggingModel(),
        max_protection_duration=12,
        initial_hi=1.0,
        wind_time_slices_per_month=5,  # Stratified sampling parameter
        decision_interval_months=1,
        max_simulation_years=25
    )

    env = OffshoreMaintenanceEnv(config)
    obs, info = env.reset(seed=42)

    hi_history = []

    print("Running baseline validation (always continue)...")
    for step in range(300): # 25 years * 12 months
        action = np.zeros(env.n_turbines, dtype=np.int32)
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
    parser.add_argument("--turbine-csv", type=str, required=True, help="Path to the turbine power curve data")
    parser.add_argument("--damage-csv", type=str, required=True, help="Path to the response surface damage data")
    parser.add_argument("--design-life", type=float, default=20.0, help="Design life in years (Default: 20)")

    args = parser.parse_args()

    run_validation(
        turbine_csv=args.turbine_csv,
        damage_csv=args.damage_csv,
        design_life=args.design_life
    )
