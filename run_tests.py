import subprocess
with open("/Users/andriepasca/Documents/AI-Research-Office/projects/paper-3/research/experiments/whisper-rl/test_output.txt", "w") as f:
    subprocess.run(["pytest", "tests/"], stdout=f, stderr=subprocess.STDOUT, cwd="/Users/andriepasca/Documents/AI-Research-Office/projects/paper-3/research/experiments/whisper-rl")
