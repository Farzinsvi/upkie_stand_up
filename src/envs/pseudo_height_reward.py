#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria

from typing import Tuple

from pyquaternion import Quaternion

import numpy as np

class PseudoHeightReward:

    """!
    Reward function for standing up from a lying down position. We cannot obtain
    directly the height of the robot, so this function uses the distances between
    the current joint and IMU values and the target values of the standing up position.

    The euclidian distances is used for joints values (wheels not included), and a 
    Quaternion distance is used for the IMUs.

    This reward function can only be used with the IMU-extended servos environment.

    This class has the following attributes:

    - ``lookahead_duration``: Length of the receding horizon, used to compute
      an internal divergent component of motion.
    - ``max_pitch``: Maximum pitch angle we expect to observe, in [rad].
    - ``max_position``: Maximum ground position we expect to observe, in [m].
    - ``pitch_weight``: Weight of the pitch objective in the reward.
    - ``position_weight``: Weight of the position objective in the reward.
    """

    imu_weight: float
    joints_weight: float

    @staticmethod
    def get_range() -> Tuple[float, float]:
        """!
        Get range of the reward.

        This is part of the Gym API.
        """
        return (-float("inf"), 1.0)

    def __init__(
        self,
        imu_weight: float = 1.0,
        joints_weight: float = 1.0,
    ):
        """!
        Initialize reward.

        @param imu_weight Weight of the difference with the target IMU
        @param joints_weight Weight of the difference with the target value of joints
        """
        self.imu_weight = imu_weight
        self.joints_weight = joints_weight

    def get(self, observation: np.ndarray) -> float:
        """!
        Get reward corresponding to an observation.

        @param observation Observation to compute reward from.
        @returns Reward.
        """
        joints = [0, 1, 3, 4]
        joints_value = observation[[0, 1, 3, 4]]
        imu_value = Quaternion(observation[18:])

        target_joints = np.array([0.0, 0.0, 0.0, 0.0])
        target_imu = Quaternion(np.array([0.0, 0.0, 0.0, 1.0]))
        
        return (
            1.0
            - self.imu_weight * Quaternion.absolute_distance(imu_value, target_imu)
            - self.joints_weight * np.linalg.norm(joints_value - target_joints)
        )
