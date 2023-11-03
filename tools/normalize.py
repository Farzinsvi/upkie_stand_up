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

def normalize_vector(vector, max_values, min_values):
    """
    Normalizes a vector between -1 and 1 based on maximum and minimum values.

    Args:
        vector (ndarray): The vector to be normalized.
        max_values (ndarray): The maximum values for each dimension.
        min_values (ndarray): The minimum values for each dimension.

    Returns:
        ndarray: The normalized vector.
    """
    normalized_vector = 2 * (vector - min_values) / (max_values - min_values) - 1
    return normalized_vector


def unnormalize_vector(normalized_vector, max_values, min_values):
    """
    Unnormalizes a vector between -1 and 1 back to its original subspace.

    Args:
        normalized_vector (ndarray): The normalized vector.
        max_values (ndarray): The maximum values for each dimension.
        min_values (ndarray): The minimum values for each dimension.

    Returns:
        ndarray: The unnormalized vector.
    """
    unnormalized_vector = ((normalized_vector + 1) * (max_values - min_values)) / 2 + min_values
    return unnormalized_vector