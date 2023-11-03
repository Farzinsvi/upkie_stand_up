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

from typing import Tuple

from upkie_stand_up.tools.observators import pseudo_height, pseudo_height_2
from upkie.observers.base_pitch import compute_base_pitch_from_imu

import numpy as np

MAX_PITCH = 1.5707963267948966

class StandUpReward:

    """!
    TODO: explain

    - ``pitch_weight``: Weight of the pitch objective in the reward.
    - ``height_weight``: Weight of the height in the reward.
    """

    @staticmethod
    def get_range() -> Tuple[float, float]:
        """!
        Get range of the reward.

        This is part of the Gym API.
        """
        return (0.0, +float("inf"))

    def __init__(
        self,
        pitch_weight,
        height_weight
    ):
        """!
        Initialize reward.
        """
        self.pitch_weight = pitch_weight
        self.height_weight = height_weight

        return

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """

        pitch = observation[6]
        full_joints = np.hstack([observation[:3], -observation[:3]])
        
        return (
            1.0 +
            self.height_weight * pseudo_height_2(full_joints) +
            (-1.0) * self.pitch_weight * pitch / MAX_PITCH)
