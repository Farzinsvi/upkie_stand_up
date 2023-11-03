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

import xpag
from xpag.wrappers import gym_vec_env
from xpag.buffers import DefaultBuffer
from xpag.samplers import DefaultSampler
from xpag.setters import DefaultSetter

from xpag.agents import SAC, TQC, TD3

from xpag.tools import learn

from upkie_stand_up.src.envs import register

from upkie_stand_up.tools.utils import print_arguments_as_table

agent_dict = {
    'SAC': SAC,
    'TQC': TQC,
    'TD3': TD3
}

register()


def train_model(
                agent_name='SAC',
                env_name='UpkieWheelsEnv-v3',
                exp_name='unnamed_exp',
                # Agent arguments
                actor_lr=3e-3,
                critic_lr=3e-3,
                temp_lr=3e-4,
                tau=5e-2,
                # Buffer arguments
                buffer_size=100_000,
                # Learning arguments
                batch_size=256,
                gradient_steps=1,
                start_training_after=0,
                max_steps=1_000_000,
                evaluate_every=2_500,
                save_every=50_000,
                seed=0
                ):
    """Functions that trains an agent in an environment and saves it in exp_name

    Args:
        agent_name (str, optional): SAC, TD3 or TQC. Defaults to 'SAC'.
        env_name (str, optional): Environment. Defaults to 'UpkieWheelsEnv-v3'.
        exp_name (str, optional): Experiment name. Defaults to 'unnamed_exp'.
        actor_lr (_type_, optional): Actor learning rate. Defaults to 3e-3.
        critic_lr (_type_, optional): Critic learning rate. Defaults to 3e-3.
        temp_lr (_type_, optional): Temp. learning rate. Defaults to 3e-4.
        tau (_type_, optional): Tau. Defaults to 5e-2.
        buffer_size (_type_, optional): Buffer size. Defaults to 100_000.
        batch_size (int, optional): Batch size. Defaults to 256.
        gradient_steps (int, optional): Number of gradient steps. Defaults to 1.
        start_training_after (int, optional): Number of steps before starting training. Defaults to 0.
        max_steps (_type_, optional): Number of steps of training. Defaults to 1_000_000.
        evaluate_every (_type_, optional): Evaluate every X number of steps. Defaults to 2_500.
        save_every (_type_, optional): Save every X number of steps. Defaults to 50_000.
        seed (int, optional): Random seed. Defaults to 0.
    """
    
    # Print arguments
    print_arguments_as_table(locals())

    # Create the environment
    num_envs = 1
    env, eval_env, env_info = gym_vec_env(env_name, num_envs)

    # Create the agent
    agent = agent_dict[agent_name](
        env_info['observation_dim'],
        env_info['action_dim'],
        {
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "temp_lr": temp_lr,
            "tau": tau,
            "seed": seed
        }
    )

    # Create the replay buffer
    sampler = DefaultSampler()
    buffer = DefaultBuffer(
        buffer_size=buffer_size,
        sampler=sampler
    )
    setter = DefaultSetter()

    save_dir = os.path.join(os.getcwd(), 'results', agent_name, exp_name)
    save_episode = True
    plot_projection = None

    # Start training
    learn(
        env,
        eval_env,
        env_info,
        agent,
        buffer,
        setter,
        batch_size=batch_size,
        gd_steps_per_step=gradient_steps,
        start_training_after_x_steps=start_training_after,
        max_steps=max_steps,
        evaluate_every_x_steps=evaluate_every,
        save_agent_every_x_steps=save_every,
        save_dir=save_dir,
        save_episode=save_episode,
        plot_projection=plot_projection,
        custom_eval_function=None,
        additional_step_keys=None,
        seed=seed
    )

if __name__ == "__main__":
    train_model()

