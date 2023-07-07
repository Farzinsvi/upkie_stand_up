#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria

"""Experiment to see how the robot is controlled."""

import gymnasium as gym
import numpy as np

import upkie.envs

import math

steps = 1000
repetitions = 1
joints = np.arange(0, 6)

config = {
    "bullet": {
        "orientation_init_base_in_world": [0.707, 0.0, -0.707, 0.0],
        "position_init_base_in_world": [0.0, 0.0, 0.1],
    }
}

pos1 = np.arange(0, 6)
pos2 = np.arange(6, 12)
pos3 = np.arange(12, 18)

pos_list = [pos1, pos2, pos3]

if __name__ == "__main__":
    upkie.envs.register()
    with gym.make("UpkieServosEnv-v2", config=config, frequency=200.0) as env:
        action_space = env.action_space
        for i, pos in enumerate(pos_list):
            for _ in range(repetitions):
                observation = env.reset()
                action = np.zeros(env.action_space.shape)
                action[pos] = action_space.sample()[pos]
                print(action)
                for step in range(steps):
                    observation, _, _, _, _ = env.step(action)
                    action[pos] = action_space.sample()[pos]
