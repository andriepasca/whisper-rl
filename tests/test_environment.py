import whisper_env
import pytest
import numpy as np
import os
import tempfile
import pandas as pd

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

@pytest.fixture
def mock_turbine_csv():
    fd, path = tempfile.mkstemp(suffix='.csv')
    df = pd.DataFrame({
        "Wind speed [m/s]": np.linspace(3, 25, 23),
        "Power [kW]": np.linspace(0, 5000, 23),
        "Thrust coefficient [-]": np.linspace(0.1, 0.8, 23),
        "Cp": np.linspace(0.2, 0.4, 23),
        "Ct": np.linspace(0.1, 0.8, 23)
    })
    df.to_csv(path, index=False)
    yield path
    os.close(fd)
    os.remove(path)

@pytest.fixture
def mock_damage_csv():
    fd, path = tempfile.mkstemp(suffix='.csv')
    u = np.linspace(3, 25, 5)
    ti = np.linspace(0.05, 0.25, 5)
    U, TI = np.meshgrid(u, ti)

    df = pd.DataFrame({
        "u": U.flatten(),
        "ti": TI.flatten(),
        "del_flap": np.random.uniform(1000, 5000, 25),
        "del_edge": np.random.uniform(2000, 8000, 25)
    })
    df.to_csv(path, index=False)
    yield path
    os.close(fd)
    os.remove(path)

@pytest.fixture
def env_config(mock_turbine_csv, mock_damage_csv):
    layout = RandomScatteredLayout(n_turbines=4, seed=42)
    layout_config = LayoutConfig(x=layout.x, y=layout.y)

    turbine_config = TurbineConfig(
        name="TestTurbine",
        rotor_diameter=120.0,
        hub_height=90.0,
        csv_path=mock_turbine_csv
    )

    wake_config = WakeSolverConfig(
        layout=layout_config,
        turbine=turbine_config,
        wake_deficit_model=BastankhahGaussianDeficit(),
        superposition_model=LinearSum(),
        turbulence_model=STF2017TurbulenceModel()
    )
    wake_solver = WakeSolver(wake_config)

    damage_config = DamageSolverConfig(csv_path=mock_damage_csv)
    damage_solver = DamageSolver(damage_config)

    # Mocking wind climate, simplistic
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
        def solve(self, metrics): return whisper_env.RewardResult(reward=0.0, objectives=metrics)

    class DummySpatialGrouping:
        def solve(self, indices): return 0.0

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
        wind_time_slices_per_month=10,
        decision_interval_months=1,
        max_simulation_years=1
    )
    return config

def test_environment_initialization(env_config):
    env = OffshoreMaintenanceEnv(env_config)
    obs, info = env.reset()
    assert obs is not None
    assert info is not None

def test_environment_step(env_config):
    env = OffshoreMaintenanceEnv(env_config)
    env.reset()

    action = np.zeros(env.n_turbines, dtype=np.int32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert not np.any(np.isnan(env.delta_damage)), "Damage should not contain NaNs"
    assert not np.any(np.isnan(env.power)), "Power should not contain NaNs"
    assert not terminated

def test_environment_multiple_steps(env_config):
    env = OffshoreMaintenanceEnv(env_config)
    env.reset()

    for _ in range(5):
        action = np.zeros(env.n_turbines, dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert not np.any(np.isnan(env.delta_damage))
        assert not np.any(np.isnan(env.power))
