#!/bin/bash
echo 'Starting tests'
echo 'Testing policy of UpkieAlphaEnv'
python tools/test_agent.py --agent SAC --exp_name UpkieAlphaEnv --env UpkieAlphaEnv-v0 --steps 1_000
echo 'Testing policy of UpkieBetaEnv'
python tools/test_agent.py --agent SAC --exp_name UpkieBetaEnv --env UpkieBetaEnv-v0 --steps 1_000
echo 'Testing policy of UpkieGammaEnv'
python tools/test_agent.py --agent SAC --exp_name UpkieGammaEnv --env UpkieGammaEnv-v0 --steps 1_000
echo 'Testing policy of UpkieAlphaEnv2'
python tools/test_agent.py --agent SAC --exp_name UpkieAlphaEnv2 --env UpkieAlphaEnv2-v0 --steps 1_000
echo 'Testing policy of UpkieBetaEnv2'
python tools/test_agent.py --agent SAC --exp_name UpkieBetaEnv2 --env UpkieBetaEnv2-v0 --steps 1_000
echo 'Testing policy of UpkieGammaEnv2'
python tools/test_agent.py --agent SAC --exp_name UpkieGammaEnv2 --env UpkieGammaEnv2-v0 --steps 1_000
echo 'Testing policy of UpkieAlphaEnv3'
python tools/test_agent.py --agent SAC --exp_name UpkieAlphaEnv3 --env UpkieAlphaEnv3-v0 --steps 1_000
echo 'Testing policy of UpkieBetaEnv3'
python tools/test_agent.py --agent SAC --exp_name UpkieBetaEnv3 --env UpkieBetaEnv3-v0 --steps 1_000
echo 'Testing policy of UpkieGammaEnv3'
python tools/test_agent.py --agent SAC --exp_name UpkieGammaEnv3 --env UpkieGammaEnv3-v0 --steps 1_000
echo 'Testing policy of UpkieAlphaEnv4'
python tools/test_agent.py --agent SAC --exp_name UpkieAlphaEnv4 --env UpkieAlphaEnv4-v0 --steps 1_000
echo 'Testing policy of UpkieBetaEnv4'
python tools/test_agent.py --agent SAC --exp_name UpkieBetaEnv4 --env UpkieBetaEnv4-v0 --steps 1_000
echo 'Testing policy of UpkieGammaEnv4'
python tools/test_agent.py --agent SAC --exp_name UpkieGammaEnv4 --env UpkieGammaEnv4-v0 --steps 1_000
echo 'Testing policy of UpkieAlphaEnv5'
python tools/test_agent.py --agent SAC --exp_name UpkieAlphaEnv5 --env UpkieAlphaEnv5-v0 --steps 1_000
echo 'Testing policy of UpkieBetaEnv5'
python tools/test_agent.py --agent SAC --exp_name UpkieBetaEnv5 --env UpkieBetaEnv5-v0 --steps 1_000
echo 'Testing policy of UpkieGammaEnv5'
python tools/test_agent.py --agent SAC --exp_name UpkieGammaEnv5 --env UpkieGammaEnv5-v0 --steps 1_000
echo 'Testing policy of UpkieWheelsNewEnv'
python tools/test_agent.py --agent SAC --exp_name UpkieWheelsNewEnv --env UpkieWheelsNewEnv-v0 --steps 2_500
echo 'Testing policy of UpkieWheelsNewEnv2'
python tools/test_agent.py --agent SAC --exp_name UpkieWheelsNewEnv2 --env UpkieWheelsNewEnv2-v0 --steps 2_500
echo 'Testing policy of UpkieWheelsNewEnv3'
python tools/test_agent.py --agent SAC --exp_name UpkieWheelsNewEnv3 --env UpkieWheelsNewEnv3-v0 --steps 2_500
echo 'Tests finished'
