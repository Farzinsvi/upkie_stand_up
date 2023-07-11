#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria

"""Getting up from an asymmetric split position."""

import gymnasium as gym
import numpy as np

import upkie
import upkie.envs

from upkie_stand_up.tools.utils import generate_trajectory

joints = np.arange(0, 6)

points = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.54, 0.0, 0.0, 1.54, 0.0, 0.0],
    [1.54, 0.0, 0.0, 1.54, 0.0, 0.0],
    [1.54, 0.0, 0.0, 1.54, 0.0, 0.0],
    [0.0, 0.0, -6.0, 0.0, 0.0, -6.0],
    [0.0, 0.0, -9.0, 0.0, 0.0, -9.0],
])

steps = np.array([
    20,
    100,
    100,
    20,
    300
])

total_steps = np.sum(steps)
actions = generate_trajectory(points, steps)
repetitions = 5


config = {
    "bullet": {
        "position_init_base_in_world": [0.0, 0.0, 0.6],
    }
}

if __name__ == "__main__":
    upkie.envs.register()
    with gym.make("UpkieServosEnv-v2", config=config, frequency=200.0) as env:
        for _ in range(repetitions):
            observation = env.reset()
            action = np.zeros(env.action_space.shape)
            for step in range(total_steps):
                observation, _, _, _, _ = env.step(action)
                action[joints] = actions[step]
