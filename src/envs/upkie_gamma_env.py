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

from typing import Optional

import numpy as np
import pinocchio as pin
import upkie_description

import math

from gymnasium import spaces

from upkie.utils.clamp import clamp_and_warn
from upkie.utils.pinocchio import (
    box_position_limits,
    box_torque_limits,
    box_velocity_limits,
)

from upkie.observers.base_pitch import compute_base_pitch_from_imu
from upkie_stand_up.tools.new_observators import compute_height

from upkie_stand_up.src.envs.upkie_base_env import UpkieBaseEnv
from upkie_stand_up.src.envs.upkie_base_env import DEFAULT_CONFIG, LYING_CONFIG

from upkie_stand_up.src.envs.new_rewards import *

from upkie_stand_up.tools.normalize import normalize_vector, unnormalize_vector
from upkie_stand_up.tools.actions import create_action_dict

imu_max = np.array([1.0, 1.0, 1.0, 1.0])
imu_dim = 4

# max height: is less than one, max pitch: pi/2
MAX_HEIGHT = 0.34299999999999997
MAX_PITCH = 1.5707963267948966

MAX_WHEEL_POSITION = 10.0

class UpkieGammaEnv(UpkieBaseEnv):

    """!
        Environment used to learn the stand up motion with Upkie. It reduces the
        state and action space dimensions using bilateral symmetry of the robot
        and the movement we want to learn.

        8-dimensional state space:
        - 3 average position
        - 3 average velocity
        - height
        - pitch

        9-dimensional control:
        - 3 position
        - 3 velocity
        - 3 maximum torque

        The same action is commanded symmetrically to both sides of the robot.

        This is the environment with the most general control among UpkieAlpha, UpkieBeta
        and UpkieGamma.
    """

    robot: pin.RobotWrapper
    version: int = 0

    def __init__(
        self,
        config: Optional[dict] = LYING_CONFIG,
        fall_pitch: float = np.inf,
        frequency: float = None,
        reward = MinReward,
        shm_name: str = "/vulp",
    ):
        """!
        Initialize environment.

        @param config Configuration dictionary, also sent to the spine.
        @param fall_pitch Fall pitch angle, in radians.
        @param frequency Regulated frequency of the control loop, in Hz.
        @param shm_name Name of shared-memory file.
        """
        super().__init__(
            config=config,
            fall_pitch=fall_pitch,
            frequency=None,
            shm_name=shm_name,
        )

        # load robot and model
        robot = upkie_description.load_in_pinocchio(root_joint=None)
        model = robot.model

        # get max and min values of position, velocity and torque
        q_min, q_max = box_position_limits(model)
        q_max[2] = MAX_WHEEL_POSITION
        q_min[2] = -MAX_WHEEL_POSITION
        v_max = box_velocity_limits(model)
        tau_max = box_torque_limits(model)
        height_pitch_max = np.array([MAX_HEIGHT, MAX_PITCH])
        height_pitch_min = np.array([-MAX_HEIGHT, -MAX_PITCH])
        # position: 3 dim, velocity: 3 dim, height + pitch = 2 dim
        state_dim = 3 + 3 + 2
        state_max = np.hstack([q_max[:3], v_max[:3], height_pitch_max])
        state_min = np.hstack([q_min[:3], -v_max[:3], height_pitch_min])
        state_max = np.float32(state_max)
        state_min = np.float32(state_min)

        action_dim = 9
        action_max = np.hstack([q_max[:3], v_max[:3], tau_max[:3]])
        action_min = np.hstack([q_min[:3], -v_max[:3], np.zeros(3)])
    

        # gym.Env: action_space
        self.action_space = spaces.Box(
            np.float32(-np.ones(action_dim)),
            np.float32(np.ones(action_dim)),
            shape=(action_dim,),
            dtype=np.float32,
        )

        # gym.Env: observation_space
        self.observation_space = spaces.Box(
            state_min,
            state_max,
            shape=(state_dim,),
            dtype=np.float32,
        )

        # gym.Env: reward_range
        self.reward = reward()
        self.reward_range = reward.get_range()

        # Class members
        self.__joints = list(robot.model.names)[1:]
        self.left_joint_list = ['left_hip', 'left_knee', 'left_wheel']
        self.right_joint_list = ['right_hip', 'right_knee', 'right_wheel']
        self.__last_positions = {}
        self.q_max = q_max
        self.q_min = q_min
        self.robot = robot
        self.model = model
        self.tau_max = tau_max
        self.v_max = v_max
        self.height_pitch_max = height_pitch_max
        self.height_pitch_min = height_pitch_min

        self.state_dim = state_dim
        self.state_min = state_min
        self.state_max = state_max

        self.action_dim = action_dim
        self.action_max = action_max
        self.action_min = action_min

    def parse_first_observation(self, observation_dict: dict) -> None:
        """!
        Parse first observation after the spine interface is initialize.

        @param observation_dict First observation.
        """
        self.__last_positions = {
            joint: observation_dict["servo"][joint]["position"]
            for joint in self.__joints
        }

    def vectorize_observation(self, observation_dict: dict) -> np.ndarray:
        """!
        Extract observation vector from a full observation dictionary.

        @param observation_dict Full observation dictionary from the spine.
        @returns Observation vector.
        """
        obs = np.empty(self.state_dim)
        for left_joint, right_joint in zip(self.left_joint_list, self.right_joint_list):
            i = self.model.getJointId(left_joint) - 1
            obs[i] = (observation_dict["servo"][left_joint]["position"] - observation_dict["servo"][right_joint]["position"])/2
            obs[3 + i] = (observation_dict["servo"][left_joint]["velocity"] - observation_dict["servo"][right_joint]["velocity"])/2
        imu_orientation = observation_dict["imu"]["orientation"]
        obs[6] = compute_height(np.hstack([obs[:3], -obs[:3]]), imu_orientation)
        obs[7] = compute_base_pitch_from_imu(imu_orientation)
        return obs

    def dictionarize_action(self, action: np.ndarray) -> dict:
        """!
        Convert action vector into a spine action dictionary.

        @param action Action vector.
        @returns Action dictionary.
        """

        action = unnormalize_vector(action, self.action_max, self.action_min)

        position_action = np.hstack([action[:3], -action[:3]])
        velocity_action = np.hstack([action[3:6], -action[3:6]])
        torque_action = np.hstack([action[6:9], action[6:9]])

        action_dict = create_action_dict(
            position = position_action,
            velocity = velocity_action,
            torque = torque_action
        )

        return action_dict
