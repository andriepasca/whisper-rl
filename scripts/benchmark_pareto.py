import os
import csv
import argparse
import numpy as np
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

class FlattenDictWrapper(gym.ObservationWrapper):
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

class DummyAgent:
    def __init__(self, action_prob=0.0):
        self.action_prob = action_prob

    def predict(self, obs, deterministic=True):
        if "turbines_HI" in obs:
            n_turbines = obs["turbines_HI"].shape[0]
        else:
            n_turbines = obs["turbines"]["HI"].shape[0]
        return (np.random.rand(n_turbines) < self.action_prob).astype(np.int32), None

def main(args):
    seed = args.seed
    print(f"Starting Pareto benchmark evaluation with seed {seed}...")

    try:
        from stable_baselines3 import PPO
        sb3_available = True
    except ImportError:
        sb3_available = False

    weights = [0.0, 0.2, 0.5, 0.8, 1.0]
    dummy_probs = [0.0, 0.05, 0.1, 0.2, 1.0]

    agents = []

    for w, prob in zip(weights, dummy_probs):
        model_id = f"model_w_{w}"
        model_path_seeded = f"models/ppo_blade_w{w}_seed{seed}.zip"
        model_path_unseeded = f"models/ppo_blade_w{w}.zip"

        if sb3_available:
            if os.path.exists(model_path_seeded):
                print(f"Found trained model for {model_id} (seeded), loading...")
                agent = PPO.load(model_path_seeded)
            elif os.path.exists(model_path_unseeded):
                print(f"Found trained model for {model_id} (unseeded), loading...")
                agent = PPO.load(model_path_unseeded)
            else:
                print(f"Trained model not found for {model_id}, falling back to DummyAgent.")
                agent = DummyAgent(action_prob=prob)
        else:
            print(f"Trained model not found for {model_id}, falling back to DummyAgent.")
            agent = DummyAgent(action_prob=prob)

        agents.append({
            "model_id": model_id,
            "w_damage": w,
            "agent": agent
        })

    results = []
    base_seed = seed

    for agent_info in agents:
        print(f"Evaluating {agent_info['model_id']}...")
        config = get_default_config(seed=base_seed)

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
        config.wake_solver = WakeSolver(wake_config)
        config.damage_solver = DamageSolver(damage_solver_config)

        env = OffshoreMaintenanceEnv(config)

        is_ppo = hasattr(agent_info["agent"], "policy")
        if is_ppo:
            env = FlattenDictWrapper(env)

        obs, _ = env.reset(seed=base_seed)

        total_J_damage = 0.0
        total_J_spatial = 0.0

        done = False
        while not done:
            action, _ = agent_info["agent"].predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        unwrapped_env = env.unwrapped if is_ppo else env
        history_df = unwrapped_env.logger.dataframe()

        if not history_df.empty:
            decision_events = history_df[history_df["decision_event"] == True]
            if not decision_events.empty:
                total_J_damage = decision_events["interval_damage_burden"].sum()
                total_J_spatial = decision_events["spatial_grouping"].sum()

        results.append({
            "model_id": agent_info["model_id"],
            "J_damage": total_J_damage,
            "J_spatial": total_J_spatial
        })

        print(f"  Finished {agent_info['model_id']}: J_damage={total_J_damage:.2f}, J_spatial={total_J_spatial:.2f}")

    csv_path = "pareto_benchmark_results.csv"
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["model_id", "J_damage", "J_spatial"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results exported to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Pareto agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation (default: 42)")
    parser.add_argument("--turbine-csv", type=str, required=True, help="Path to the turbine power curve data")
    parser.add_argument("--damage-csv", type=str, required=True, help="Path to the response surface damage data")

    args = parser.parse_args()

    main(args)
