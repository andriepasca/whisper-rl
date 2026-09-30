import sys
import os

project_dir = "/Users/andriepasca/Documents/AI-Research-Office/projects/paper-3/research/experiments/whisper-rl"
sys.path.insert(0, project_dir)

from whisper_env.config import get_default_config
from whisper_env.environment import OffshoreMaintenanceEnv

config = get_default_config()
env = OffshoreMaintenanceEnv(config)

print("Calibrated del_flap_ref:", env.damage_solver.config.del_flap_ref)
print("Calibrated del_edge_ref:", env.damage_solver.config.del_edge_ref)
