import sys
import numpy as np

# Add the current directory to sys.path if needed
sys.path.append('.')

from whisper_env.environment import OffshoreMaintenanceEnv
from whisper_env.config import get_default_config

def main():
    try:
        config = get_default_config()
        env = OffshoreMaintenanceEnv(config)
        env.reset()
        n_turbines = env.n_turbines
        action = np.zeros(n_turbines)
        for _ in range(5):
            env.step(action)
        print("Environment successfully ran 5 steps without errors.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
