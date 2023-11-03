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

import numpy as np

joints = {
    ('left_hip', 0),
    ('left_knee', 1),
    ('left_wheel', 2),
    ('right_hip', 3),
    ('right_knee', 4),
    ('right_wheel', 5)
}

def create_action_dict(
    position = None,
    velocity = None,
    torque = None
):
    """Creates an action dictionary with the given values of position,
    velocity and torque

    Args:
        position (ndarray, optional): 6-dim array of floats. Defaults to None.
        velocity (ndarray, optional): 6-dim array of floats. Defaults to None.
        torque (ndarray, optional): 6-dim array of floats. Defaults to None.

    Returns:
        dict: Action dictionary ready to use in environment
    """

    servo_dict = {
        joint : {
            'position': position[i],
            'velocity': velocity[i],
            'torque': torque[i]
            }
        for joint, i in joints
    }

    return {'servo': servo_dict}
