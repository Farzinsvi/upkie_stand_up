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

from gymnasium import spaces

from upkie.utils.clamp import clamp_and_warn
from upkie.utils.pinocchio import (
    box_position_limits,
    box_torque_limits,
    box_velocity_limits,
)

from upkie_stand_up.src.envs.pseudo_height_reward import PseudoHeightReward
from upkie_stand_up.src.envs.upkie_base_env import UpkieBaseEnv
from upkie_stand_up.src.envs.upkie_base_env import LYING_CONFIG

from upkie_stand_up.tools.normalize import normalize_vector, unnormalize_vector
from upkie_stand_up.tools.actions import create_action_dict

imu_max = np.array([1.0, 1.0, 1.0, 1.0])
imu_dim = 4

max_wheels_pos = 10
wheels_pos_idx = [2]

class UpkieNormExtSymServosEnv(UpkieBaseEnv):

    """!
    Upkie with full observation extended with IMU and joint position-velocity-torque actions.

    TODO: normalize actions.

    The environment has the following attributes:

    - ``reward``: Reward function.
    - ``robot``: Pinocchio robot wrapper.
    - ``state_max``: Maximum values for the action and observation vectors.
    - ``state_min``: Minimum values for the action and observation vectors.
    - ``version``: Environment version number.

    Vectorized observations have the following structure:

    <table>
        <tr>
            <td><strong>Index</strong></td>
            <td><strong>Description</strong></td>
            </tr>
        <tr>
            <td>``[0:nq]``</td>
            <td>Joint positions in [rad].</td>
        </tr>
        <tr>
            <td>``[nq:nq + nv]``</td>
            <td>Joint velocities in [rad] / [s].</td>
        </tr>
        <tr>
            <td>``[nq + nv:nq + 2 * nv]``</td>
            <td>Joint torques in [N] * [m].</td>
        </tr>
    </table>

    Vectorized actions have the following structure:

    <table>
        <tr>
            <td><strong>Index</strong></td>
            <td><strong>Description</strong></td>
            </tr>
        <tr>
            <td>``[0:nq]``</td>
            <td>Joint position commands in [rad].</td>
        </tr>
        <tr>
            <td>``[nq:nq + nv]``</td>
            <td>Joint velocity commands in [rad] / [s].</td>
        </tr>
        <tr>
            <td>``[nq + nv:nq + 2 * nv]``</td>
            <td>Joint torques in [N] * [m].</td>
        </tr>
    </table>

    The reward function is defined in @ref
    envs.pseudo_height_reward.PseudoHeightReward "PseudoHeightReward".

    See also @ref envs.upkie_base_env.UpkieBaseEnv "UpkieBaseEnv" for notes on
    using this environment.
    """

    reward: PseudoHeightReward
    robot: pin.RobotWrapper
    version: int = 0

    def __init__(
        self,
        config: Optional[dict] = LYING_CONFIG,
        fall_pitch: float = np.inf,
        frequency: float = None,
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
            frequency=frequency,
            shm_name=shm_name,
        )

        robot = upkie_description.load_in_pinocchio(root_joint=None)
        model = robot.model

        q_min, q_max = box_position_limits(model)
        v_max = box_velocity_limits(model)
        tau_max = box_torque_limits(model)
        state_dim = model.nq + 2 * model.nv + imu_dim
        state_max = np.hstack([q_max, v_max, tau_max, imu_max])
        state_min = np.hstack([q_min, -v_max, -tau_max, -imu_max])
        state_max = np.float32(state_max)
        state_min = np.float32(state_min)

        action_dim = 9
        action_max = np.hstack([q_max[:3], v_max[:3], tau_max[:3]])
        action_min = np.hstack([q_min[:3], -v_max[:3], -tau_max[:3]])
    
        action_max[wheels_pos_idx] = max_wheels_pos
        action_min[wheels_pos_idx] = -max_wheels_pos

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
        self.reward_range = PseudoHeightReward.get_range()

        # Class members
        self.__joints = list(robot.model.names)[1:]
        self.__last_positions = {}
        self.q_max = q_max
        self.q_min = q_min
        self.reward = PseudoHeightReward(
            imu_weight=10,
            joints_weight=0
        )
        self.robot = robot
        self.tau_max = tau_max
        self.v_max = v_max

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
        nq, nv = self.robot.model.nq, self.robot.model.nv
        model = self.robot.model
        obs = np.empty(nq + 2 * nv + imu_dim)
        for joint in self.__joints:
            i = model.getJointId(joint) - 1
            obs[i] = observation_dict["servo"][joint]["position"]
            obs[nq + i] = observation_dict["servo"][joint]["velocity"]
            obs[nq + nv + i] = observation_dict["servo"][joint]["torque"]
        obs[nq + 2 * nv : nq + 2 * nv + imu_dim] = observation_dict["imu"]["orientation"]
        return obs

    def dictionarize_action(self, action: np.ndarray) -> dict:
        """!
        Convert action vector into a spine action dictionary.

        @param action Action vector.
        @returns Action dictionary.
        """
        nq = self.robot.model.nq
        model = self.robot.model

        nq, nv = model.nq, model.nv

        action = unnormalize_vector(action, self.action_max, self.action_min)

        # TODO: rewrite this
        action_ = np.zeros(self.action_dim * 2)
        for i in range(3):
            idx = 6*i
            action_[idx:idx+3] = action[3*i:3*(i+1)]
            action_[idx+3:idx+6] = -action[3*i:3*(i+1)]

        action = action_

        action_dict = create_action_dict(
            position = action[:6],
            velocity = action[6:12],
            torque = action[12:]
        )

        return action_dict
