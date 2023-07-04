#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria

"""Genuflect while lying on a horizontal floor."""

import gymnasium as gym
import numpy as np

import upkie.envs

# import upkie_stand_up

# from upkie_stand_up.tools.trajectory import generate_trajectory

import numpy as np


# Given a list of configuration space points C = [c_1, ..., c_N] and
# list of steps between points M = [m_1, ..., m_N-1], generate the
# trajectory in the configuration space joining the points with the
# required number of inbetween points

def interpolate_two_points(x, y, n):
    trajectory = []
    for t in np.linspace(0, 1, n):
        trajectory.append(x + t * (y - x))
    return trajectory

def generate_trajectory(points, steps):
    trajectory = []
    for i in range(len(steps)):
        trajectory += interpolate_two_points(points[i], points[i+1], steps[i])
    return trajectory
    

joints = [0,1,2,3,4,5]

points = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 2.57, 0.0, 0.0, -2.57, 0.0],
    [1.54, 2.57, 0.0, -1.54, -2.57, 0.0],
    [1.54, 2.57, 0.0, -1.54, -2.57, 0.0]
])

steps = np.array([
    10,
    10,
    100
])

total_steps = np.sum(steps)

actions = generate_trajectory(points, steps)

repetitions = 10

amplitude = 2.0  # in radians

config = {
    "bullet": {
        "orientation_init_base_in_world": [0.707, 0.0, -0.707, 0.0],
        "position_init_base_in_world": [0.0, 0.0, 0.1],
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
                action[joints] = amplitude * actions[step]
