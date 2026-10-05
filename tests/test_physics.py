import pytest
import numpy as np
import os
import tempfile
import pandas as pd

from whisper_env import (
    DamageSolverConfig, DamageSolver,
    WindClimateConfig, WindClimate
)

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
    wind_config = WindClimateConfig(monthly_weibull=weibull)
    climate = WindClimate(wind_config)

    rng = np.random.default_rng(42)

    # Check that stochastic sampling returns valid scalars (wind direction is not modeled)
    for _ in range(10):
        u = climate.sample(month=1, rng=rng)
        assert not np.isnan(u)
        assert 3.0 <= u <= 25.0

    arr = climate.sample(month=1, rng=rng, size=30)
    assert arr.shape == (30,)
    assert np.all((arr >= 3.0) & (arr <= 25.0))
