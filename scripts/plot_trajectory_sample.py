import argparse
import dataclasses
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the root path to ensure whisper_env can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from whisper_env.config import (
    TurbineConfig,
    DamageSolverConfig,
    get_default_config
)
from whisper_env.environment import OffshoreMaintenanceEnv
from whisper_env.physics import WakeSolver, DamageSolver

def main():
    parser = argparse.ArgumentParser(description="Simulate and plot 30-year stochastic HI trajectory.")
    parser.add_argument("--turbine-csv", type=str, required=True, help="Path to the turbine CSV file.")
    parser.add_argument("--damage-csv", type=str, required=True, help="Path to the damage CSV file.")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 100, 2024], help="Seeds to use for generating trajectories.")
    args = parser.parse_args()

    seeds = args.seeds
    
    plt.figure(figsize=(10, 6))

    for seed in seeds:
        # 1. Create a base config for this seed
        base_env_config = get_default_config(seed=seed)
        
        # 2. Override TurbineConfig
        turbine_config = TurbineConfig(csv_path=args.turbine_csv)
        
        # Recreate WakeSolverConfig with new TurbineConfig, preserving layout
        base_wake_config = base_env_config.wake_solver.config
        wake_solver_config = dataclasses.replace(
            base_wake_config,
            turbine=turbine_config
        )
        wake_solver = WakeSolver(wake_solver_config)
        
        # 3. Set up DamageSolverConfig with the required values
        damage_solver_config = DamageSolverConfig(
            csv_path=args.damage_csv,
            design_life_years=20.0
        )
        damage_solver = DamageSolver(damage_solver_config)
        
        # 4. Replace solvers and update max_simulation_years (Frozen Dataclass mutability rule)
        env_config = dataclasses.replace(
            base_env_config,
            wake_solver=wake_solver,
            damage_solver=damage_solver,
            max_simulation_years=30
        )
        
        # Initialize Environment
        env = OffshoreMaintenanceEnv(env_config)
        
        # Reset environment with the seed
        obs, info = env.reset(seed=seed)
        
        months = [0]
        # HI[0] is the Health Index for Turbine 0
        his = [env.unwrapped.HI[0]]
        
        terminated, truncated = False, False
        while not (terminated or truncated):
            # No intervention
            action = np.zeros(env.unwrapped.n_turbines, dtype=np.int32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record elapsed month and HI for Turbine 0
            months.append(env.unwrapped.elapsed_month)
            his.append(env.unwrapped.HI[0])
            
        plt.plot(months, his, label=f"Seed {seed}")

    # Theoretical 20-Year Baseline: from (0, 1.0) to (240, 0.0)
    plt.plot([0, 240], [1.0, 0.0], 'k--', label='Theoretical 20-Year Baseline')

    plt.xlabel('Elapsed Month')
    plt.ylabel('Health Index (HI) Turbine 0')
    plt.title('30-year Stochastic HI Trajectory vs 20-year Baseline')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'trajectory.png'))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
