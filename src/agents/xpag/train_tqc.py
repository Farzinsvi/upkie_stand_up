import xpag

import os
import sys
import PIL
import importlib
from packaging import version
from IPython import get_ipython
from IPython.display import clear_output, HTML, display
import inspect
import jax
import argparse


from xpag.wrappers import gym_vec_env
from xpag.buffers import DefaultBuffer
from xpag.samplers import DefaultSampler
from xpag.setters import DefaultSetter
from xpag.agents import SAC, TQC
from xpag.tools import learn
from xpag.tools import mujoco_notebook_replay

import gymnasium as gym

import upkie

from upkie.envs import register
from upkie.envs import UpkieWheelsEnv

print(jax.lib.xla_bridge.get_backend().platform)

# register Upkie environments
register()
# __import__("IPython").embed()

# remove warnings from tensorflow_probability, a library used by the SAC agent in xpag
# ("WARNING:root:The use of `check_types` is deprecated and does not have any effect.)
import logging
logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

# block warnings
logger.setLevel(logging.CRITICAL)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Upkie wheel balancer with SAC')
    
    parser.add_argument('--max_steps', type=int, default=1_000_000_000,
                        help='Maximum number of steps for training (default: 1_000_000_000)')
    
    parser.add_argument('--evaluate_every_x_steps', type=int, default=5_000,
                        help='Evaluate the agent every x steps (default: 5_000)')
    
    parser.add_argument('--save_agent_every_x_steps', type=int, default=50_000,
                        help='Save the agent every x steps (default: 50_000)')
    
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Name of the experiment')
    
    return parser.parse_args()


args = parse_arguments()

print('Max steps:', args.max_steps)
print('Evaluate every x steps:', args.evaluate_every_x_steps)
print('Save agent every x steps:', args.save_agent_every_x_steps)
print('Experiment name:', args.exp_name)


# Training and eval environments
num_envs = 1
env, eval_env, env_info = gym_vec_env('UpkieWheelsEnv-v3', num_envs)

agent = TQC(
    env_info['observation_dim'],
    env_info['action_dim'],
    {
        "actor_lr": 3e-3,
        "critic_lr": 3e-3,
        "temp_lr": 3e-4,
        "tau": 5e-2,
        "seed": 0
    }
)
sampler = DefaultSampler()
buffer = DefaultBuffer(
    buffer_size=200_000,
    sampler=sampler
)
setter = DefaultSetter()

batch_size = 256
gd_steps_per_step = 1
start_training_after_x_steps = 0
max_steps = args.max_steps
evaluate_every_x_steps = args.evaluate_every_x_steps
save_agent_every_x_steps = args.save_agent_every_x_steps
save_dir = os.path.join(os.path.dirname(__file__), 'results', args.exp_name)
save_episode = True
plot_projection = None
seed = 0


learn(
    env,
    eval_env,
    env_info,
    agent,
    buffer,
    setter,
    batch_size=batch_size,
    gd_steps_per_step=gd_steps_per_step,
    start_training_after_x_steps=start_training_after_x_steps,
    max_steps=max_steps,
    evaluate_every_x_steps=evaluate_every_x_steps,
    save_agent_every_x_steps=save_agent_every_x_steps,
    save_dir=save_dir,
    save_episode=save_episode,
    plot_projection=plot_projection,
    custom_eval_function=None,
    additional_step_keys=None,
    seed=seed
)