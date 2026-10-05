import os
import argparse
import numpy as np
import torch
import dataclasses
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from whisper_env import (
    get_default_config, 
    OffshoreMaintenanceEnv,
    DamageSolverConfig,
    DamageSolver
)
from whisper_env.config import DEFAULT_DATA_DIR, make_fixed_damage_solver_config
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
            "farm_progress_ratio": env.observation_space["farm"]["progress_ratio"],
            "turbines_HI": env.observation_space["turbines"]["HI"],
            "turbines_protection_remaining": env.observation_space["turbines"]["protection_remaining"]
        })

    def observation(self, obs):
        return {
            "farm_month": obs["farm"]["month"],
            "farm_progress_ratio": obs["farm"]["progress_ratio"],
            "turbines_HI": obs["turbines"]["HI"],
            "turbines_protection_remaining": obs["turbines"]["protection_remaining"]
        }

def train_pareto_agents(args):
    damage_csv = os.path.abspath(args.damage_csv)
    if not os.path.isfile(damage_csv):
        raise SystemExit(f"ERROR: damage CSV not found: {damage_csv} (check path; beware doubled '.csv.csv')")
    print(f"Using damage CSV: {damage_csv}")

    os.makedirs("models", exist_ok=True)
    weights = args.weights

    # Set global seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for w in weights:
        print(f"\n--- Training agent with w_damage = {w} (Seed: {args.seed}) ---")
        config = get_default_config(w_damage=w, seed=args.seed)

        damage_solver_config = make_fixed_damage_solver_config(damage_csv)

        config = dataclasses.replace(config, 
            damage_solver=DamageSolver(damage_solver_config),
            enable_logging=False
        )

        env = OffshoreMaintenanceEnv(config)
        env = FlattenDictWrapper(env)

        # Seed the action and observation spaces
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)

        # Configure CheckpointCallback and optional EvalCallback
        checkpoint_dir = os.path.join("models", "checkpoints", f"w_{w}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        callbacks = []
        if args.save_freq > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=args.save_freq,
                save_path=checkpoint_dir,
                name_prefix=f"ppo_blade_w_{w}"
            )
            callbacks.append(checkpoint_callback)

        if args.eval_freq > 0:
            eval_env = OffshoreMaintenanceEnv(config)
            eval_env = FlattenDictWrapper(eval_env)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(checkpoint_dir, "best_model"),
                log_path=os.path.join(checkpoint_dir, "eval_logs"),
                eval_freq=args.eval_freq,
                deterministic=True
            )
            callbacks.append(eval_callback)

        callback_list = CallbackList(callbacks) if callbacks else None

        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            seed=args.seed,
            verbose=1,
            tensorboard_log="./logs/tensorboard/"
        )
        model.learn(total_timesteps=args.timesteps, callback=callback_list, progress_bar=True)

        # Save final model for 100% backward compatibility
        model_path_seeded = os.path.join("models", f"ppo_blade_w{w}_seed{args.seed}.zip")
        model_path_w = os.path.join("models", f"ppo_blade_w_{w}.zip")
        model_path_unseeded = os.path.join("models", f"ppo_blade_w{w}.zip")

        model.save(model_path_seeded)
        model.save(model_path_w)
        model.save(model_path_unseeded)
        print(f"Saved final model to {model_path_seeded} and {model_path_w}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agents for different Pareto weights")
    parser.add_argument("--timesteps", type=int, default=50000, help="Number of timesteps to train each agent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for PPO")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to run for each environment per update")
    parser.add_argument("--damage-csv", type=str, default=os.path.join(DEFAULT_DATA_DIR, 'response_surface.csv'), help="Path to the response surface damage data")
    parser.add_argument("--weights", type=float, nargs='+', default=[0.0, 0.2, 0.5, 0.8, 1.0], help="List of scalarization weights w_damage to train")
    parser.add_argument("--save-freq", type=int, default=50000, help="Save model checkpoint every N timesteps per weight")
    parser.add_argument("--eval-freq", type=int, default=0, help="Frequency of evaluation in timesteps (0 to disable)")
    args = parser.parse_args()

    train_pareto_agents(args)
