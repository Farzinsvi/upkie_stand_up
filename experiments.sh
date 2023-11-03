#!/bin/bash
echo 'Starting experiments'
# echo 'Experiment UpkieAlphaEnv'
# python tools/train_agent.py --agent SAC --exp_name UpkieAlphaEnv --env UpkieAlphaEnv-v0
# echo 'Experiment UpkieAlphaEnv2'
# python tools/train_agent.py --agent SAC --exp_name UpkieAlphaEnv2 --env UpkieAlphaEnv2-v0
# echo 'Experiment UpkieAlphaEnv3'
# python tools/train_agent.py --agent SAC --exp_name UpkieAlphaEnv3 --env UpkieAlphaEnv3-v0
echo 'Experiment UpkieAlphaEnv4'
python tools/train_agent.py --agent SAC --exp_name UpkieAlphaEnv4 --env UpkieAlphaEnv4-v0
echo 'Experiment UpkieAlphaEnv5'
python tools/train_agent.py --agent SAC --exp_name UpkieAlphaEnv5 --env UpkieAlphaEnv5-v0
# echo 'Experiment UpkieBetaEnv'
# python tools/train_agent.py --agent SAC --exp_name UpkieBetaEnv --env UpkieBetaEnv-v0
# echo 'Experiment UpkieBetaEnv2'
# python tools/train_agent.py --agent SAC --exp_name UpkieBetaEnv2 --env UpkieBetaEnv2-v0
# echo 'Experiment UpkieBetaEnv3'
# python tools/train_agent.py --agent SAC --exp_name UpkieBetaEnv3 --env UpkieBetaEnv3-v0
echo 'Experiment UpkieBetaEnv4'
python tools/train_agent.py --agent SAC --exp_name UpkieBetaEnv4 --env UpkieBetaEnv4-v0
echo 'Experiment UpkieBetaEnv5'
python tools/train_agent.py --agent SAC --exp_name UpkieBetaEnv5 --env UpkieBetaEnv5-v0
# echo 'Experiment UpkieGammaEnv'
# python tools/train_agent.py --agent SAC --exp_name UpkieGammaEnv --env UpkieGammaEnv-v0
# echo 'Experiment UpkieGammaEnv2'
# python tools/train_agent.py --agent SAC --exp_name UpkieGammaEnv2 --env UpkieGammaEnv2-v0
# echo 'Experiment UpkieGammaEnv3'
# python tools/train_agent.py --agent SAC --exp_name UpkieGammaEnv3 --env UpkieGammaEnv3-v0
echo 'Experiment UpkieGammaEnv4'
python tools/train_agent.py --agent SAC --exp_name UpkieGammaEnv4 --env UpkieGammaEnv4-v0
echo 'Experiment UpkieGammaEnv5'
python tools/train_agent.py --agent SAC --exp_name UpkieGammaEnv5 --env UpkieGammaEnv5-v0
echo 'Experiment UpkieWheelsNewEnv'
python tools/train_agent.py --agent SAC --exp_name UpkieWheelsNewEnv --env UpkieWheelsNewEnv-v0
echo 'Experiment UpkieWheelsNewEnv2'
python tools/train_agent.py --agent SAC --exp_name UpkieWheelsNewEnv2 --env UpkieWheelsNewEnv2-v0
echo 'Experiment UpkieWheelsNewEnv3'
python tools/train_agent.py --agent SAC --exp_name UpkieWheelsNewEnv3 --env UpkieWheelsNewEnv3-v0
echo 'Experiments finished'
