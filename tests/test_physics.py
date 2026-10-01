import pytest
import numpy as np
import os
import tempfile
import pandas as pd

from whisper_env import (
    WakeSolverConfig, WakeSolver,
    DamageSolverConfig, DamageSolver,
    LayoutConfig, TurbineConfig, WindClimateConfig, WindClimate
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

def test_wake_solver_no_nans(mock_turbine_csv):
    layout_config = LayoutConfig(x=np.array([0, 500]), y=np.array([0, 0]))
    turbine_config = TurbineConfig(
        name="Test", rotor_diameter=100.0, hub_height=100.0, csv_path=mock_turbine_csv
    )
    config = WakeSolverConfig(
        layout=layout_config,
        turbine=turbine_config,
        wake_deficit_model=BastankhahGaussianDeficit(),
        superposition_model=LinearSum(),
        turbulence_model=STF2017TurbulenceModel()
    )
    solver = WakeSolver(config)

    res = solver.solve(ambient_u=10.0, ambient_wd=270.0)
    assert "power" in res
    assert not np.any(np.isnan(res["power"]))
    assert not np.any(np.isnan(res["u_eff"]))
    assert not np.any(np.isnan(res["ti_eff"]))

def test_damage_solver_no_nans(mock_damage_csv):
    config = DamageSolverConfig(
        csv_path=mock_damage_csv,
        del_flap_ref=2800.0,
        del_edge_ref=5500.0
    )
    solver = DamageSolver(config)

    u_eff = np.array([10.0, 8.5])
    ti_eff = np.array([0.1, 0.15])

    damage = solver.solve(u_eff, ti_eff, duration_minutes=10.0)
    assert not np.any(np.isnan(damage))

def test_sample_no_nans():
    weibull = {m: {"c": 8.0, "k": 2.0} for m in range(1, 13)}
    wind_config = WindClimateConfig(
        monthly_weibull=weibull,
        wind_direction=np.arange(0, 360, 30),
        wind_direction_probability=np.ones(12)/12
    )
    climate = WindClimate(wind_config)

    rng = np.random.default_rng(42)

    # Check that stochastic sampling correctly generates valid arrays
    for _ in range(10):
        u, wd = climate.sample(month=1, rng=rng)
        assert not np.isnan(u)
        assert not np.isnan(wd)
        assert 3.0 <= u <= 25.0
