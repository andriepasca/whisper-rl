import os
import csv
import numpy as np
from whisper_env import get_default_config, OffshoreMaintenanceEnv

class DummyAgent:
    def __init__(self, action_prob=0.0):
        self.action_prob = action_prob

    def predict(self, obs, deterministic=True):
        n_turbines = obs["turbines"]["HI"].shape[0]
        return (np.random.rand(n_turbines) < self.action_prob).astype(np.int32)

def main():
    print("Starting Pareto benchmark evaluation...")

    agents = [
        {"model_id": "model_w_0.0", "w_damage": 0.0, "agent": DummyAgent(action_prob=0.0)},
        {"model_id": "model_w_0.2", "w_damage": 0.2, "agent": DummyAgent(action_prob=0.05)},
        {"model_id": "model_w_0.5", "w_damage": 0.5, "agent": DummyAgent(action_prob=0.1)},
        {"model_id": "model_w_0.8", "w_damage": 0.8, "agent": DummyAgent(action_prob=0.2)},
        {"model_id": "model_w_1.0", "w_damage": 1.0, "agent": DummyAgent(action_prob=1.0)},
    ]

    results = []
    base_seed = 12345

    # We use full 30-year episode (max_simulation_years=30) but we don't change n_turbines or wind_time_slices_per_month
    # to evaluate correctly, let's keep defaults but it takes time. Actually, if I use the default config exactly,
    # the evaluation runs the standard full setup. Since I just tested the script locally with scaled down params,
    # I'll revert it to proper defaults so it is correct in the PR.

    for agent_info in agents:
        print(f"Evaluating {agent_info['model_id']}...")
        config = get_default_config(seed=base_seed)

        env = OffshoreMaintenanceEnv(config)
        obs, _ = env.reset(seed=base_seed)

        total_J_damage = 0.0
        total_J_spatial = 0.0

        done = False
        while not done:
            action = agent_info["agent"].predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        history_df = env.logger.dataframe()

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
    main()
