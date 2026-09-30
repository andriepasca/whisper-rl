import gymnasium as gym
import json
import os
import datetime
import numpy as np

class TrajectoryLogger(gym.Wrapper):
    """
    A Gymnasium wrapper that automatically saves step-by-step metrics
    (Health Index, actions taken, wind speed, J_damage, J_spatial)
    into a structured JSON file at the end of an episode.
    """
    def __init__(self, env, log_dir="logs"):
        super().__init__(env)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.episode_count = 0
        self.trajectory = []

    def reset(self, **kwargs):
        self.trajectory = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        env_unwrapped = self.env.unwrapped

        # Collect the specifically requested metrics
        hi = getattr(env_unwrapped, 'HI', [])
        hi_mean = float(np.mean(hi)) if len(hi) > 0 else 0.0
        wind_speed = getattr(env_unwrapped, 'ambient_u', 0.0)

        j_damage = 0.0
        j_spatial = 0.0
        if hasattr(env_unwrapped, 'logger') and hasattr(env_unwrapped.logger, 'history') and len(env_unwrapped.logger.history) > 0:
            last_log = env_unwrapped.logger.history[-1]
            j_damage = float(last_log.get("interval_damage_burden", 0.0))
            j_spatial = float(last_log.get("spatial_grouping", 0.0))

        elapsed_month = getattr(env_unwrapped, 'elapsed_month', 0)

        step_data = {
            "elapsed_month": elapsed_month,
            "Health Index": hi_mean,
            "actions taken": action.tolist() if hasattr(action, "tolist") else list(action),
            "wind speed": float(wind_speed),
            "J_damage": j_damage,
            "J_spatial": j_spatial
        }

        self.trajectory.append(step_data)

        if terminated or truncated:
            self._save_trajectory()
            self.episode_count += 1

        return obs, reward, terminated, truncated, info

    def _save_trajectory(self):
        if not self.trajectory:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.log_dir, f"trajectory_ep{self.episode_count}_{timestamp}.json")

        with open(filename, 'w') as f:
            json.dump(self.trajectory, f, indent=4)

        print(f"Saved trajectory to {filename}")
