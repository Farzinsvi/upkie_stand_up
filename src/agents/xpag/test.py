#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 ISIR. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from upkie_stand_up.src.envs import register

import xpag
from xpag.agents import SAC, TQC, TD3

import gymnasium as gym

from upkie_stand_up.tools.utils import print_arguments_as_table

agent_dict = {
    'SAC': SAC,
    'TQC': TQC,
    'TD3': TD3
}

register()

def test_model(
            agent_name='SAC',
            env_name='UpkieWheelsEnv-v3',
            exp_name='unnamed_exp',
            steps=1_000_000
            ):
    """Function to test the model learned in exp_name using agent_name in env_name for the
    given number of steps


    Args:   
        agent_name (str, optional): SAC, TD3 or TQC. Defaults to 'SAC'.
        env_name (str, optional): Environment. Defaults to 'UpkieWheelsEnv-v3'.
        exp_name (str, optional): Experiment name. Defaults to 'unnamed_exp'.
        steps (_type_, optional): Number of steps. Defaults to 1_000_000.
    """
    
    # Print arguments
    print_arguments_as_table(locals())

    # Create environment
    # TODO: test frequency=None
    env = gym.make(env_name, frequency=200.0)

    # Create agent
    agent = agent_dict[agent_name](
        env.observation_space.shape[0],
        env.action_space.shape[0],
        {
            'seed': 0
        }
    )

    # Load agent
    agent.load(os.path.join(os.getcwd(), 'results', agent_name, exp_name, 'best_agent'))

    # Test agent
    observation = (env.reset()[0])
    action = agent.select_action(observation, eval_mode=True)
    for step in range(steps):
        observation, reward, done, _, _ = env.step(action)
        if done:
            observation = (env.reset()[0])
        action = agent.select_action(observation, eval_mode=True)
    
if __name__ == "__main__":
    test_model()
