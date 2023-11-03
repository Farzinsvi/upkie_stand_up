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
    Important: these rewards can only be used with the environments UpkieAlphaEnv,
    UpkieBetaEnv and UpkieGammaEnv. Except GroundPositionReward and BalancingReward,
    that are to be used for UpkieWheelsNewEnv
"""

from typing import Tuple

import numpy as np

# max height: is less than one, max pitch: pi/2
MAX_HEIGHT = 0.34299999999999997
MAX_PITCH = 1.5707963267948966
MAX_GROUND_POSITION = 1.0

class TimeReward:
    """!
    Gives 1.0 or reward each steps.

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
        
        return 1.0

class HeightReward:
    """!
    Returns the normalized height of the robot.

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

        height = observation[6]
        
        return ((height / MAX_HEIGHT) * (height > 0))


class PitchReward:

    """!
    Returns the normalized pitch of the robot.

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

        pitch = observation[7]
        
        return ((1.0 - np.abs(pitch) / MAX_PITCH) * (np.abs(pitch) < MAX_PITCH))


class SumReward:
    """!
    Returns the normalized sum of the height and pitch rewards.

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
        self.r_height = HeightReward()
        self.r_pitch = PitchReward()

        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """
 
        return (self.r_height.get(observation) + self.r_pitch.get(observation)) / 2


class ProdReward:

    """!
    Returns the product of the height and pitch rewards.

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
        self.r_height = HeightReward()
        self.r_pitch = PitchReward()

        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """
 
        return (self.r_height.get(observation) * self.r_pitch.get(observation))


class MinReward:

    """!
    Returns the minimum between the height and pitch rewards.

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
        self.r_height = HeightReward()
        self.r_pitch = PitchReward()

        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """
 
        return min(self.r_height.get(observation), self.r_pitch.get(observation))

class BoundedSumReward:

    """!
    Returns the sum of bounded variants of the height and pitch rewards, where
    the reward is 0 unless height or pitch are above a certain threshold.

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
        self,
        min_height=0.15,
        min_pitch=0.75
    ):
        """!
        Initialize reward
        """
        self.r_height = HeightReward()
        self.r_pitch = PitchReward()
        self.min_height = min_height
        self.min_pitch = min_pitch

        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """

        r_h = self.r_height.get(observation)
        r_p = self.r_pitch.get(observation)
 
        return (r_h * (r_h > self.min_height) + r_p * (r_p > self.min_pitch))/2

class BoundedProdReward:

    """!
    Returns the product of bounded variants of the height and pitch rewards, where
    the reward is 0 unless height or pitch are above a certain threshold.

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
        self,
        min_height=0.15,
        min_pitch=0.75
    ):
        """!
        Initialize reward
        """
        self.r_height = HeightReward()
        self.r_pitch = PitchReward()
        self.min_height = min_height
        self.min_pitch = min_pitch

        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """

        r_h = self.r_height.get(observation)
        r_p = self.r_pitch.get(observation)
 
        return (r_h * (r_h > self.min_height) * r_p * (r_p > self.min_pitch))/2

class MinHeightProdReward:

    """!
    Returns the product of the height and pitch rewards, or 0 if the height is not
    above a certain threshold.

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
        self,
        min_height=MAX_HEIGHT * 0.5
    ):
        """!
        Initialize reward
        """
        self.r_height = HeightReward()
        self.r_pitch = PitchReward()
        self.min_height = min_height

        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """

        r_h = self.r_height.get(observation)
        r_p = self.r_pitch.get(observation)
 
        return r_h * r_p * (r_h > self.min_height)


class MinHeightSumReward:

    """!
    Returns the sum of the height and pitch rewards, or 0 if the height is not
    above a certain threshold.

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
        self,
        min_height=0.15
    ):
        """!
        Initialize reward
        """
        self.r_height = HeightReward()
        self.r_pitch = PitchReward()
        self.min_height = min_height

        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """

        r_h = self.r_height.get(observation)
        r_p = self.r_pitch.get(observation)
 
        return (r_h + r_p) * (r_h > self.min_height)



class GroundPositionReward:

    """!
    Returns the normalized ground position of the robot given by wheel odometry.

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

        ground_position = observation[8]
        
        return ((1.0 - np.abs(ground_position) / MAX_GROUND_POSITION) * (np.abs(ground_position) < MAX_GROUND_POSITION))


class BalancingReward:

    """!
    Returns the product between the height and ground position rewards.

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

        self.r_pitch = PitchReward()
        self.r_ground = GroundPositionReward()

        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """

        ground_position = observation[8]
        
        return self.r_pitch.get(observation) * self.r_ground.get(observation)

