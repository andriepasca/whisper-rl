# Getting Started

## Installation

WHISPER-RL must be installed locally via pip. First, clone the repository, then install it in editable mode:

```bash
git clone <repository_url>
cd <repository_directory>
pip install -e .
```

Dependencies include `gymnasium`, `numpy`, `pandas`, `xarray`, `stable-baselines3`, and `seaborn`.

## Quickstart

Here is a simple 10-line Gymnasium quickstart snippet to initialize and run the environment:

```python
import gymnasium as gym
from whisper_env import get_default_config, OffshoreMaintenanceEnv

config = get_default_config(n_turbines=5)
env = OffshoreMaintenanceEnv(config)
obs, info = env.reset(seed=42)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```
