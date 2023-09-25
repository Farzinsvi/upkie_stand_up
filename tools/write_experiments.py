
final_envs = ["UpkieAlphaEnv", "UpkieAlphaEnv2", "UpkieAlphaEnv3", "UpkieAlphaEnv4", "UpkieAlphaEnv5", 
        "UpkieBetaEnv", "UpkieBetaEnv2", "UpkieBetaEnv3", "UpkieBetaEnv4", "UpkieBetaEnv5",
        "UpkieGammaEnv", "UpkieGammaEnv2", "UpkieGammaEnv3", "UpkieGammaEnv4", "UpkieGammaEnv5"]

wheels_envs = ["UpkieWheelsNewEnv", "UpkieWheelsNewEnv2", "UpkieWheelsNewEnv3"]

with open('experiments.sh', 'w')as f:
    f.write('#!/bin/bash\n')
    f.write("echo 'Starting experiments'\n")
    for name in final_envs:
        f.write(f"echo 'Experiment {name}'\n")
        f.write(f"python tools/train_agent.py --agent SAC --exp_name {name} --env {name}-v0\n")
    for name in wheels_envs:
        f.write(f"echo 'Experiment {name}'\n")
        f.write(f"python tools/train_agent.py --agent SAC --exp_name {name} --env {name}-v0\n")
    f.write("echo 'Experiments finished'\n")