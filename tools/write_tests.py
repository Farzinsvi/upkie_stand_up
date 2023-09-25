
final_envs = ["UpkieAlphaEnv", "UpkieAlphaEnv2", "UpkieAlphaEnv3", "UpkieAlphaEnv4", "UpkieAlphaEnv5", 
        "UpkieBetaEnv", "UpkieBetaEnv2", "UpkieBetaEnv3", "UpkieBetaEnv4", "UpkieBetaEnv5",
        "UpkieGammaEnv", "UpkieGammaEnv2", "UpkieGammaEnv3", "UpkieGammaEnv4", "UpkieGammaEnv5"]

wheels_envs = ["UpkieWheelsNewEnv", "UpkieWheelsNewEnv2", "UpkieWheelsNewEnv3"]

with open('tests.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write("echo 'Starting tests'\n")
    for i in range(5):
        for j in range(3):
            name = final_envs[i + 5*j]
            f.write(f"echo 'Testing policy of {name}'\n")
            f.write(f"python tools/test_agent.py --agent SAC --exp_name {name} --env {name}-v0 --steps 1_000\n")
    for name in wheels_envs:
        f.write(f"echo 'Testing policy of {name}'\n")
        f.write(f"python tools/test_agent.py --agent SAC --exp_name {name} --env {name}-v0 --steps 2_500\n")
    f.write("echo 'Tests finished'\n")