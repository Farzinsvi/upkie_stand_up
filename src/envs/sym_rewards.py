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

"""
    Important: these rewards can only be used with the environment UpkieStandUpSymEnv.
"""

from typing import Tuple

import numpy as np

# max height: is less than one; max pitch: pi/2
MAX_HEIGHT = 0.34299999999999997
MAX_PITCH = 1.5707963267948966
MAX_POSITION = 1.0

class StandingReward:

    """!
    This reward is the multiplication of a height factor and a pitch factor,
    encouraging the agent to optimize jointly the height and the pitch.

    This class has no attributes.
    """

    @staticmethod
    def get_range() -> Tuple[float, float]:
        """!
        Get range of the reward.

        This is part of the Gym API.
        """
        return (0.0, 1.0)

    def __init__(
        self
    ):
        """!
        Initialize reward
        """
        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """

        height = observation[9]
        pitch = observation[10]

        height_reward = ((height / MAX_HEIGHT) * (height > 0))
        pitch_reward = ((1.0 - np.abs(pitch) / MAX_PITCH) * (np.abs(pitch) < MAX_PITCH))
 
        return (height_reward * pitch_reward)


class StandingSymReward:

    """!
    This reward is the multiplication of a height factor, a pitch factor,
    and a position factor, encouraging the agent to optimize jointly the
    height and the pitch while staying close to a point. In the literature,
    this is said to be the most important difference between learning a
    standing up motion and walking.

    This class has no attributes.
    """

    @staticmethod
    def get_range() -> Tuple[float, float]:
        """!
        Get range of the reward.

        This is part of the Gym API.
        """
        return (0.0, 1.0)

    def __init__(
        self
    ):
        """!
        Initialize reward
        """
        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """

        height = observation[9]
        pitch = observation[10]
        position = observation[12]

        height_reward = ((height / MAX_HEIGHT) * (height > 0))
        pitch_reward = ((1.0 - np.abs(pitch) / MAX_PITCH) * (np.abs(pitch) < MAX_PITCH))
        position_reward = ((1.0 - np.abs(position) / MAX_POSITION) * (np.abs(position) < MAX_POSITION))
 
        return (height_reward * pitch_reward * position_reward)


class StandingSymRegReward:

    """!
    Same as StandUpAsymReward, but with an additive max-torque regularization term,
    encouraging the agent to keep the torques as small as possible. 

    This class has no attributes.
    """

    @staticmethod
    def get_range() -> Tuple[float, float]:
        """!
        Get range of the reward.

        This is part of the Gym API.
        """
        return (-1.0, 1.0)

    def __init__(
        self,
        tau_max
    ):
        """!
        Initialize reward
        """
        self.tau_max = tau_max
        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """

        height = observation[9]
        pitch = observation[10]
        position = observation[12]
        tau = observation[:3]

        height_reward = ((height / MAX_HEIGHT) * (height > 0))
        pitch_reward = ((1.0 - np.abs(pitch) / MAX_PITCH) * (np.abs(pitch) < MAX_PITCH))
        position_reward = ((1.0 - np.abs(position) / MAX_POSITION) * (np.abs(position) < MAX_POSITION))

        tau_regularization = np.linalg.norm(tau)/np.linalg.norm(self.tau_max)
 
        return (height_reward * pitch_reward * position_reward) - 1/4 * tau_regularization

