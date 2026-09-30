import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from whisper_env import get_default_config, OffshoreMaintenanceEnv
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings("ignore")

class FlattenDictWrapper(gym.ObservationWrapper):
    """
    SB3 does not support nested Dict observation spaces natively out of the box with MultiInputPolicy properly without explicit handling.
    This wrapper flattens the nested dictionary into a single-level dictionary, which MultiInputPolicy can handle easily.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "farm_month": env.observation_space["farm"]["month"],
            "turbines_HI": env.observation_space["turbines"]["HI"],
            "turbines_protection_remaining": env.observation_space["turbines"]["protection_remaining"]
        })

    def observation(self, obs):
        return {
            "farm_month": obs["farm"]["month"],
            "turbines_HI": obs["turbines"]["HI"],
            "turbines_protection_remaining": obs["turbines"]["protection_remaining"]
        }

def train_pareto_agents(timesteps, seed=42, lr=3e-4, n_steps=2048):
    os.makedirs("models", exist_ok=True)
    weights = [0.0, 0.2, 0.5, 0.8, 1.0]

    # Set global seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    for w in weights:
        print(f"\n--- Training agent with w_damage = {w} (Seed: {seed}) ---")
        config = get_default_config(w_damage=w, seed=seed)

        env = OffshoreMaintenanceEnv(config)
        env = FlattenDictWrapper(env)

        # Seed the action and observation spaces
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=lr,
            n_steps=n_steps,
            seed=seed,
            verbose=1
        )
        model.learn(total_timesteps=timesteps)

        model_path = os.path.join("models", f"ppo_blade_w{w}_seed{seed}.zip")
        model.save(model_path)
        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agents for different Pareto weights")
    parser.add_argument("--timesteps", type=int, default=50000, help="Number of timesteps to train each agent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for PPO")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to run for each environment per update")
    args = parser.parse_args()

    train_pareto_agents(args.timesteps, seed=args.seed, lr=args.lr, n_steps=args.n_steps)
