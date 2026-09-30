import os
import argparse
import numpy as np
import torch
import dataclasses
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from whisper_env import (
    get_default_config, 
    OffshoreMaintenanceEnv,
    TurbineConfig,
    DamageSolverConfig,
    WakeSolverConfig,
    WakeSolver,
    DamageSolver
)
from py_wake.deficit_models import BastankhahGaussianDeficit
from py_wake.superposition_models import LinearSum
from py_wake.turbulence_models import STF2017TurbulenceModel
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

def train_pareto_agents(args):
    os.makedirs("models", exist_ok=True)
    weights = [0.0, 0.2, 0.5, 0.8, 1.0]

    # Set global seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for w in weights:
        print(f"\n--- Training agent with w_damage = {w} (Seed: {args.seed}) ---")
        config = get_default_config(w_damage=w, seed=args.seed)

        turbine_config = TurbineConfig(
            name="NREL 5-MW",
            rotor_diameter=126.0,
            hub_height=90.0,
            csv_path=args.turbine_csv,
            power_unit="kW"
        )

        damage_solver_config = DamageSolverConfig(
            csv_path=args.damage_csv,
            u_column="u",
            ti_column="ti",
            del_flap_column="del_flap",
            del_edge_column="del_edge",
            m_coef=10.0,
            del_flap_ref=2803.716141751299,
            del_edge_ref=5588.786717232858,
            design_life_years=20
        )

        wake_config = WakeSolverConfig(
            layout=config.wake_solver.config.layout,
            turbine=turbine_config,
            wake_deficit_model=BastankhahGaussianDeficit(),
            superposition_model=LinearSum(),
            turbulence_model=STF2017TurbulenceModel()
        )
        config = dataclasses.replace(config, 
            wake_solver=WakeSolver(wake_config), 
            damage_solver=DamageSolver(damage_solver_config)
        )

        env = OffshoreMaintenanceEnv(config)
        env = FlattenDictWrapper(env)

        # Seed the action and observation spaces
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)

        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            seed=args.seed,
            verbose=1
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=True)

        model_path = os.path.join("models", f"ppo_blade_w{w}_seed{args.seed}.zip")
        model.save(model_path)
        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agents for different Pareto weights")
    parser.add_argument("--timesteps", type=int, default=50000, help="Number of timesteps to train each agent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for PPO")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to run for each environment per update")
    parser.add_argument("--turbine-csv", type=str, required=True, help="Path to the turbine power curve data")
    parser.add_argument("--damage-csv", type=str, required=True, help="Path to the response surface damage data")
    args = parser.parse_args()

    train_pareto_agents(args)
