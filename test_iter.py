import sys
import numpy as np
from whisper_env.config import EnvironmentConfig, WakeSolverConfig, DamageSolverConfig, WindClimateConfig
from whisper_env.utils import WindClimate

try:
    wind_climate = WindClimate(WindClimateConfig())
    data = list(wind_climate.iter_stratified(1, 30))
    print(f"Total slices: {len(data)}")
    for i, (u, wd, p) in enumerate(data[:5]):
        print(f"i={i}, u={u}, wd={wd}, p={p}")
except Exception as e:
    print(e)
