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

from prettytable import PrettyTable

def print_arguments_as_table(arguments):
    """Prints the arguments of a function in a table

    Args:
        arguments (object): Arguments of a function
    """
    table = PrettyTable()
    table.field_names = ['Argument', 'Value']

    for key, value in arguments.items():
        table.add_row([key, value])

    print(table)


# Given a list of configuration space points C = [c_1, ..., c_N] and
# list of steps between points M = [m_1, ..., m_N-1], generate the
# trajectory in the configuration space joining the points with the
# required number of inbetween points

def interpolate_two_points(x, y, n):
    trajectory = []
    for t in np.linspace(0, 1, n):
        trajectory.append(x + t * (y - x))
    return trajectory

def generate_trajectory(points, steps):
    trajectory = []
    for i in range(len(steps)):
        trajectory += interpolate_two_points(points[i], points[i+1], steps[i])
    return trajectory