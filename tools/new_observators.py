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

import pinocchio as pin
import upkie_description
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Tuple

robot = upkie_description.load_in_pinocchio(root_joint=None)

# Function taken from upkie utils 

def rotation_matrix_from_quaternion(
    quat: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Convert a unit quaternion to the matrix representing the same rotation.

    Args:
        quat: Unit quaternion to convert, in ``[w, x, y, z]`` format.

    Returns:
        Rotation matrix corresponding to this quaternion.

    See `Conversion between quaternions and rotation matrices`_.

    .. _`Conversion between quaternions and rotation matrices`:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Rotation_matrices
    """
    if abs(np.dot(quat, quat) - 1.0) > 1e-5:
        raise ValueError(f"Quaternion {quat} is not normalized")
    qw, qx, qy, qz = quat
    return np.array(
        [
            [
                1 - 2 * (qy ** 2 + qz ** 2),
                2 * (qx * qy - qz * qw),
                2 * (qw * qy + qx * qz),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx ** 2 + qz ** 2),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx ** 2 + qy ** 2),
            ],
        ]
    )

def compute_height(position, imu_orientation):
    """Compute de height of Upkie given its joint configuration and IMU orientation.

    This is computed as the height difference between the IMU device and the 

    Args:
        position (ndarray): 6-dim vector codyfing the position of each joint
        imu_orientation (ndarray): 4-dim IMU orientation

    Returns:
        float: float representing the height of Upkie
    """
    # Perform the forward kinematics over the kinematic tree
    pin.forwardKinematics(robot.model, robot.data, position)
    # Get IMU id
    imu_id = robot.model.getFrameId("imu")
    # Get IMU translation
    imu_translation = robot.data.oMi[imu_id].translation.T
    # Get IMU rotation
    imu_rotation = rotation_matrix_from_quaternion(imu_orientation)

    heights = [(imu_rotation @ (oMi.translation.T - imu_translation))[2]
        for oMi in robot.data.oMi][1:]

    return (0 - (heights[2] + heights[5])/2)


if __name__ == "__main__":
    position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    imu = np.array([0.0, 0.0, 0.0, 1.0])
    height = compute_height(position, imu)
    print(height)
    