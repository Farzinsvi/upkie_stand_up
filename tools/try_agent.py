import os
import argparse

import upkie

from upkie.envs import register
from upkie.envs import UpkieWheelsEnv

import xpag
from xpag.agents import SAC, TQC

import gymnasium as gym

parser = argparse.ArgumentParser(description='Experiment plotter')
parser.add_argument('--exp_path', type=str, help='Path to the experiment', required=True)
parser.add_argument('--agent', choices=['SAC', 'TQC'], help='SAC or TQC', required=True)
parser.add_argument('--steps', type=int, default=1_000_000, help='Number of steps')
args = parser.parse_args()
exp_path = args.exp_path
agent = args.agent
steps = args.steps

agent_dict = {
    'SAC': SAC,
    'TQC': TQC
}

register()

# NOTE: frequency=None does not work
env = gym.make("UpkieWheelsEnv-v3", frequency=200.0)

print(env.observation_space.shape[0],
    env.action_space.shape[0])

# Create agent
# NOTE: Error when creating agent without params
agent = agent_dict[agent](
    env.observation_space.shape[0],
    env.action_space.shape[0],
    {
        "actor_lr": 3e-3,
        "critic_lr": 3e-3,
        "temp_lr": 3e-4,
        "tau": 5e-2,
        "seed": 0
    }
)

# Load agent
agent.load(os.path.join(exp_path, 'agent'))

# TODO: solve the way observations are handled
observation = (env.reset()[0])
action = agent.select_action(observation, eval_mode=True)
for step in range(steps):
    observation, reward, done, _, _ = env.step(action)
    if done:
        observation = (env.reset()[0])
    action = agent.select_action(observation, eval_mode=True)



