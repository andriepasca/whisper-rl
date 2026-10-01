import re
with open("/Users/andriepasca/Documents/AI-Research-Office/projects/paper-3/research/experiments/whisper-rl/whisper_env/physics.py", "r") as f:
    text = f.read()

# Remove WakeSolver class
new_text = re.sub(r'class WakeSolver:.*?(?=class DamageSolver:)', '', text, flags=re.DOTALL)

with open("/Users/andriepasca/Documents/AI-Research-Office/projects/paper-3/research/experiments/whisper-rl/whisper_env/physics.py", "w") as f:
    f.write(new_text)
