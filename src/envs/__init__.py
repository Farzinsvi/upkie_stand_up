#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
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

import gymnasium as gym

from upkie_stand_up.src.envs.upkie_base_env import UpkieBaseEnv
from upkie_stand_up.src.envs.upkie_alpha_env import UpkieAlphaEnv
from upkie_stand_up.src.envs.upkie_beta_env import UpkieBetaEnv
from upkie_stand_up.src.envs.upkie_gamma_env import UpkieGammaEnv 
from upkie_stand_up.src.envs.upkie_wheels_new_env import UpkieWheelsNewEnv
from upkie_stand_up.src.envs.upkie_stand_up_new_env import UpkieStandUpNewEnv
from upkie_stand_up.src.envs.upkie_stand_up_sym_env import UpkieStandUpSymEnv
from upkie_stand_up.src.envs.upkie_stand_up_asym_env import UpkieStandUpAsymEnv

standing_envs = ["UpkieAlphaEnv", "UpkieBetaEnv", "UpkieGammaEnv", "UpkieStandUpNewEnv",
                "UpkieStandUpSymEnv", "UpkieStandUpAsymEnv"]

balancing_envs = ["UpkieWheelsNewEnv"]

def register():
    for name in standing_envs:
        gym.envs.registration.register(
        id=f"{name}-v0",
        entry_point=f"upkie_stand_up.src.envs:{name}",
        max_episode_steps=1_000,
    )
    for name in balancing_envs:
        gym.envs.registration.register(
        id=f"{name}-v0",
        entry_point=f"upkie_stand_up.src.envs:{name}",
        max_episode_steps=2_500,
    )


__all__ = [
    "UpkieBaseEnv",
    "UpkieServosEnv",
    "UpkieWheelsEnv",
    "UpkieExtendedServosEnv",
    "UpkieNormExtServosEnv",
    "UpkieNormExtSymServosEnv",
    "UpkieStandUpEnv",
    "UpkieStandUpEnv2",
    "register",
] + standing_envs + balancing_envs
