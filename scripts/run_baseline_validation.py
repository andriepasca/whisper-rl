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

def create_mock_csvs():
    # Turbine
    fd_t, path_t = tempfile.mkstemp(suffix='.csv')
    df_t = pd.DataFrame({
        "Wind speed [m/s]": np.linspace(3, 25, 23),
        "Power [kW]": np.linspace(0, 5000, 23),
        "Thrust coefficient [-]": np.linspace(0.1, 0.8, 23),
        "Cp": np.linspace(0.2, 0.4, 23),
        "Ct": np.linspace(0.1, 0.8, 23)
    })
    df_t.to_csv(path_t, index=False)
    os.close(fd_t)

    # Damage
    fd_d, path_d = tempfile.mkstemp(suffix='.csv')
    u = np.linspace(3, 25, 5)
    ti = np.linspace(0.05, 0.25, 5)
    U, TI = np.meshgrid(u, ti)

    # We want degradation to be small enough so it runs 25 years without breaking immediately
    df_d = pd.DataFrame({
        "u": U.flatten(),
        "ti": TI.flatten(),
        "del_flap": np.random.uniform(100, 500, 25),
        "del_edge": np.random.uniform(200, 800, 25)
    })
    df_d.to_csv(path_d, index=False)
    os.close(fd_d)

    return path_t, path_d

def run_validation():
    path_t, path_d = create_mock_csvs()

    layout = RandomScatteredLayout(n_turbines=4, seed=42)
    layout_config = LayoutConfig(x=layout.x, y=layout.y)

    turbine_config = TurbineConfig(
        name="ValidationTurbine",
        rotor_diameter=120.0,
        hub_height=90.0,
        csv_path=path_t
    )

    wake_config = WakeSolverConfig(
        layout=layout_config,
        turbine=turbine_config,
        wake_deficit_model=BastankhahGaussianDeficit(),
        superposition_model=LinearSum(),
        turbulence_model=STF2017TurbulenceModel()
    )
    wake_solver = WakeSolver(wake_config)

    damage_config = DamageSolverConfig(csv_path=path_d)
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

    os.remove(path_t)
    os.remove(path_d)

if __name__ == "__main__":
    run_validation()
