#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria

from typing import Tuple

# from upkie_stand_up.tools.new_observators import compute_height

import numpy as np

# max height: is less than one, max pitch: pi/2
MAX_HEIGHT = 0.34299999999999997
# MAX_HEIGHT = 1.0
MAX_PITCH = 1.5707963267948966
MAX_GROUND_POSITION = 1.0

class TimeReward:

    """!
    TODO: explain

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
    TODO: explain

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
    TODO: explain

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
    TODO: explain

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
    TODO: explain

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
    TODO: explain

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
    TODO: explain

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
    TODO: explain

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
    TODO: explain

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
    TODO: explain

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
    TODO: explain

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
    TODO: explain

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

