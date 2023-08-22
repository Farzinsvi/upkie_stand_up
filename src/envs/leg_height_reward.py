#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria

from typing import Tuple

from upkie_stand_up.tools.observators import pseudo_height, pseudo_height_2

import numpy as np

class LegHeightReward:

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
        return (0.0, +float("inf"))

    def __init__(
        self
    ):
        """!
        Initialize reward.W
        """
        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """
        
        return pseudo_height_2(observation)
