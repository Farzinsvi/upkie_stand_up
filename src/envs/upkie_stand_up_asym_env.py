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

from upkie_stand_up.src.envs.asym_rewards import *

from upkie_stand_up.tools.normalize import normalize_vector, unnormalize_vector
from upkie_stand_up.tools.actions import create_action_dict

imu_max = np.array([1.0, 1.0, 1.0, 1.0])
imu_dim = 4

# max height: is less than one, max pitch: pi/2
MAX_HEIGHT = 0.34299999999999997
MAX_PITCH = 1.5707963267948966

MAX_GROUND_POSITION: float = float("inf")
MAX_GROUND_VELOCITY = 1.0
MAX_IMU_ANGULAR_VELOCITY: float = 1000.0  # rad/s

class UpkieStandUpAsymEnv(UpkieBaseEnv):
    """
        Most general environment of Upkie to learn the standing up motion. Full
        observation of position, velocity and torque, and the five extra observations
        from UpkieStandUpSymEnv.

        23-dimensional state space:
        - 6 position
        - 6 velocity
        - 6 torque
        - height
        - pitch
        - angular velocity
        - ground position
        - ground velocity

        18-dimensional action space:
        - 6 position
        - 6 velocity
        - 6 max torque

        Any reward defined in asym_rewards can be used in this environment.
    """

    robot: pin.RobotWrapper
    version: int = 0

    def __init__(
        self,
        config: Optional[dict] = LYING_CONFIG,
        fall_pitch: float = np.inf,
        frequency: float = 200,
        shm_name: str = "/vulp",
        reward = StandingReward,
        torque_reduction = 10.0
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

        # Load robot and model
        robot = upkie_description.load_in_pinocchio(root_joint=None)
        model = robot.model

        # Get max and min values of position, velocity and torque
        q_min, q_max = box_position_limits(model)
        v_max = box_velocity_limits(model)
        non_wheel_joints = [0, 1, 3, 4]
        tau_max[non_wheel_joints] = box_torque_limits(model)[non_wheel_joints] / torque_reduction
        extra_observation_limit = np.array(
            [
                MAX_HEIGHT,
                MAX_PITCH,
                MAX_IMU_ANGULAR_VELOCITY,
                MAX_GROUND_POSITION,
                MAX_GROUND_VELOCITY
            ],
            dtype=np.float32,
        )
        # Position, velocity and torque: 18 dim; height + pitch = 2 dim;
        # + angular velocity, position and velocity given by wheel odometry
        state_dim = 18 + 5
        state_max = np.hstack([q_max, v_max, tau_max, extra_observation_limit])
        state_min = np.hstack([q_min, -v_max, -tau_max, -extra_observation_limit])
        state_max = np.float32(state_max)
        state_min = np.float32(state_min)

        action_dim = 18
        action_max = np.hstack([q_max, v_max, tau_max])
        action_min = np.hstack([q_min, -v_max, np.zeros(6)])
    

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
        self.joint_list = ['left_hip', 'left_knee', 'left_wheel', 'right_hip', 'right_knee', 'right_wheel']
        self.__last_positions = {}
        self.q_max = q_max
        self.q_min = q_min
        self.robot = robot
        self.model = model
        self.tau_max = tau_max
        self.v_max = v_max
        self.extra_observation_limit = extra_observation_limit

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
        for joint in self.joint_list:
            i = self.model.getJointId(joint) - 1
            obs[i] = observation_dict["servo"][joint]["position"]
            obs[6 + i] = observation_dict["servo"][joint]["velocity"]
            obs[12 + i] = observation_dict["servo"][joint]["torque"]
        imu_orientation = observation_dict["imu"]["orientation"]
        height = compute_height(obs[:6], imu_orientation)
        theta = compute_base_pitch_from_imu(imu_orientation)
        d_theta = observation_dict["imu"]["angular_velocity"][1]
        position = observation_dict["wheel_odometry"]["position"]
        d_position = observation_dict["wheel_odometry"]["velocity"]

        obs[18:23] = [height, theta, d_theta, position, d_position]
        
        return obs

    def dictionarize_action(self, action: np.ndarray) -> dict:
        """!
        Convert action vector into a spine action dictionary.

        @param action Action vector.
        @returns Action dictionary.
        """

        action = unnormalize_vector(action, self.action_max, self.action_min)

        position_action = action[:6]
        velocity_action = action[6:12]
        torque_action = action[12:18]

        action_dict = create_action_dict(
            position = position_action,
            velocity = velocity_action,
            torque = torque_action
        )

        return action_dict
