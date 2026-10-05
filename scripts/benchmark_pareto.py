import os
import csv
import argparse
import numpy as np
import dataclasses
from whisper_env import (
    get_default_config,
    OffshoreMaintenanceEnv,
    DamageSolverConfig,
    DamageSolver
)
from whisper_env.config import DEFAULT_DATA_DIR, make_fixed_damage_solver_config
import gymnasium as gym
from gymnasium import spaces


class FlattenDictWrapper(gym.ObservationWrapper):
    """Exact copy of the wrapper in train_ppo_pareto.py (must stay identical)."""
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


class DummyAgent:
    """Random-action placeholder. Only used with --allow-dummy. NOT a trained policy."""
    def __init__(self, action_prob=0.0):
        self.action_prob = action_prob

    def predict(self, obs, deterministic=True):
        if "turbines_HI" in obs:
            n_turbines = obs["turbines_HI"].shape[0]
        else:
            n_turbines = obs["turbines"]["HI"].shape[0]
        return (np.random.rand(n_turbines) < self.action_prob).astype(np.int32), None


class NoopAgent:
    """Baseline: all actions 0."""
    def __init__(self, action_space):
        self.action = np.zeros(action_space.shape, dtype=action_space.dtype)

    def predict(self, obs, deterministic=True):
        return self.action.copy(), None


def build_env(w, seed, damage_csv, wrap):
    config = get_default_config(w_damage=w, seed=seed)
    damage_solver_config = make_fixed_damage_solver_config(damage_csv)
    # enable_logging=True: metrics are read from the logger
    config = dataclasses.replace(
        config,
        damage_solver=DamageSolver(damage_solver_config),
        enable_logging=True
    )
    env = OffshoreMaintenanceEnv(config)
    if wrap:
        env = FlattenDictWrapper(env)
    return env


def run_episode(env, agent, reset_seed):
    obs, _ = env.reset(seed=reset_seed)
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    J_damage = 0.0
    J_spatial = 0.0
    history_df = env.unwrapped.logger.dataframe()
    if not history_df.empty:
        decision_events = history_df[history_df["decision_event"] == True]
        if not decision_events.empty:
            J_damage = decision_events["interval_damage_burden"].sum()
            J_spatial = decision_events["spatial_grouping"].sum()
    return float(J_damage), float(J_spatial)


def evaluate(env, agent, seed, n_episodes):
    jd, js = [], []
    for ep in range(n_episodes):
        d, s = run_episode(env, agent, seed + ep)
        jd.append(d)
        js.append(s)
    return (float(np.mean(jd)), float(np.std(jd)),
            float(np.mean(js)), float(np.std(js)))


def main(args):
    seed = args.seed
    damage_csv = os.path.abspath(args.damage_csv)
    if not os.path.isfile(damage_csv):
        raise SystemExit(f"ERROR: damage CSV not found: {damage_csv} (check path; beware doubled '.csv.csv')")
    print(f"Using damage CSV: {damage_csv}")
    print(f"Starting Pareto benchmark evaluation with seed {seed}, n_episodes={args.n_episodes}...")

    weights = args.weights
    dummy_probs = [0.0, 0.05, 0.1, 0.2, 1.0]

    results = []

    def record(model_id, w, agent_type, stats):
        results.append({
            "model_id": model_id,
            "w_damage": w,
            "agent_type": agent_type,
            "n_episodes": args.n_episodes,
            "J_damage_mean": stats[0],
            "J_damage_std": stats[1],
            "J_spatial_mean": stats[2],
            "J_spatial_std": stats[3],
        })
        print(f"  Finished {model_id} [{agent_type}]: "
              f"J_damage={stats[0]:.2f}+/-{stats[1]:.2f}, J_spatial={stats[2]:.2f}+/-{stats[3]:.2f}")

    for i, w in enumerate(weights):
        prob = dummy_probs[i] if i < len(dummy_probs) else 0.0
        model_id = f"model_w_{w}"
        model_path_seeded = f"models/ppo_blade_w{w}_seed{seed}.zip"
        model_path_unseeded = f"models/ppo_blade_w{w}.zip"

        if os.path.exists(model_path_seeded):
            model_path = model_path_seeded
        elif os.path.exists(model_path_unseeded):
            model_path = model_path_unseeded
        else:
            model_path = None

        if model_path is not None:
            from stable_baselines3 import PPO
            print(f"Loading {model_id} from {model_path}")
            agent = PPO.load(model_path)
            agent_type = "ppo"
            wrap = True
        elif args.allow_dummy:
            print(f"!!! WARNING: model for {model_id} not found; using RANDOM DummyAgent "
                  f"(--allow-dummy). Results are NOT from a trained policy. !!!")
            agent = DummyAgent(action_prob=prob)
            agent_type = "dummy"
            wrap = False
        else:
            raise FileNotFoundError(
                f"Trained model not found for {model_id}: tried '{model_path_seeded}' and "
                f"'{model_path_unseeded}'. Use --allow-dummy to run a random DummyAgent instead."
            )

        print(f"Evaluating {model_id}...")
        env = build_env(w, seed, damage_csv, wrap)
        record(model_id, w, agent_type, evaluate(env, agent, seed, args.n_episodes))

    # No-op baseline (all actions 0). Reward weights do not affect the logged J metrics
    # under a fixed action sequence, so a single env with the default w_damage=0.5 is used.
    print("Evaluating no-op baseline...")
    env = build_env(0.5, seed, damage_csv, False)
    noop = NoopAgent(env.action_space)
    record("noop_baseline", "NA", "noop_baseline", evaluate(env, noop, seed, args.n_episodes))

    csv_path = args.output_csv
    if os.path.dirname(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_id", "w_damage", "agent_type", "n_episodes",
            "J_damage_mean", "J_damage_std", "J_spatial_mean", "J_spatial_std"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results exported to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Pareto agents")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed; episode i resets with seed+i (default: 42)")
    parser.add_argument("--n-episodes", type=int, default=20, help="Episodes per agent (default: 20)")
    parser.add_argument("--damage-csv", type=str, default=os.path.join(DEFAULT_DATA_DIR, 'response_surface.csv'), help="Path to the response surface damage data")
    parser.add_argument("--allow-dummy", action="store_true", help="Use a random DummyAgent when a model zip is missing (loud warning)")
    parser.add_argument("--weights", type=float, nargs='+', default=[0.0, 0.2, 0.5, 0.8, 1.0], help="List of scalarization weights w_damage to benchmark")
    parser.add_argument("--output-csv", type=str, default=os.path.join("results", "pareto_benchmark_results.csv"), help="Output CSV path (default: results/pareto_benchmark_results.csv)")

    args = parser.parse_args()

    main(args)
