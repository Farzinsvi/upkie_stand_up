#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria

from typing import Tuple

import numpy as np


class HeightReward:

    """!
    Reward function for standing up

    This class has the following attributes:

    """


    @staticmethod
    def get_range() -> Tuple[float, float]:
        """!
        Get range of the reward.

        This is part of the Gym API.
        """
        return (0, float("inf"))

    def __init__(
        self
    ):
        """!
        Initialize reward.
        """
        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """
        pitch = observation[0]
        ground_position = observation[1]
        ground_velocity = observation[2]
        angular_velocity = observation[3]

        T = self.lookahead_duration
        lookahead_pitch = pitch + T * angular_velocity
        lookahead_position = ground_position + T * ground_velocity
        normalized_lookahead_pitch = lookahead_pitch / self.max_pitch
        normalized_lookahead_position = lookahead_position / self.max_position
        return (
            1.0
            - self.pitch_weight * abs(normalized_lookahead_pitch)
            - self.position_weight * abs(normalized_lookahead_position)
        )
