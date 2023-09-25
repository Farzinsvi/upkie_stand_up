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
from upkie_stand_up.src.envs.upkie_servos_env import UpkieServosEnv
from upkie_stand_up.src.envs.upkie_wheels_env import UpkieWheelsEnv
from upkie_stand_up.src.envs.upkie_extended_servos_env import UpkieExtendedServosEnv
from upkie_stand_up.src.envs.upkie_norm_ext_servos_env import UpkieNormExtServosEnv
from upkie_stand_up.src.envs.upkie_norm_ext_sym_servos_env import UpkieNormExtSymServosEnv
from upkie_stand_up.src.envs.upkie_stand_up_env import UpkieStandUpEnv
from upkie_stand_up.src.envs.upkie_stand_up_2_env import UpkieStandUpEnv2
from upkie_stand_up.src.envs.upkie_alpha_env import UpkieAlphaEnv, UpkieAlphaEnv2, UpkieAlphaEnv3, UpkieAlphaEnv4, UpkieAlphaEnv5 
from upkie_stand_up.src.envs.upkie_beta_env import UpkieBetaEnv, UpkieBetaEnv2, UpkieBetaEnv3, UpkieBetaEnv4, UpkieBetaEnv5
from upkie_stand_up.src.envs.upkie_gamma_env import UpkieGammaEnv, UpkieGammaEnv2, UpkieGammaEnv3, UpkieGammaEnv4, UpkieGammaEnv5 
from upkie_stand_up.src.envs.upkie_wheels_new_env import UpkieWheelsNewEnv, UpkieWheelsNewEnv2, UpkieWheelsNewEnv3

final_envs = ["UpkieAlphaEnv", "UpkieAlphaEnv2", "UpkieAlphaEnv3", "UpkieAlphaEnv4", "UpkieAlphaEnv5", 
        "UpkieBetaEnv", "UpkieBetaEnv2", "UpkieBetaEnv3", "UpkieBetaEnv4", "UpkieBetaEnv5",
        "UpkieGammaEnv", "UpkieGammaEnv2", "UpkieGammaEnv3", "UpkieGammaEnv4", "UpkieGammaEnv5"]

wheels_envs = ["UpkieWheelsNewEnv", "UpkieWheelsNewEnv2", "UpkieWheelsNewEnv3"]

def register():
    gym.envs.registration.register(
        id=f"UpkieServosEnv-v{UpkieServosEnv.version}",
        entry_point="upkie_stand_up.src.envs:UpkieServosEnv",
        max_episode_steps=1_000_000_000,
    )
    gym.envs.registration.register(
        id=f"UpkieWheelsEnv-v{UpkieWheelsEnv.version}",
        entry_point="upkie_stand_up.src.envs:UpkieWheelsEnv",
        max_episode_steps=1_000_000_000,
    )
    gym.envs.registration.register(
        id=f"UpkieExtendedServosEnv-v{UpkieExtendedServosEnv.version}",
        entry_point="upkie_stand_up.src.envs:UpkieExtendedServosEnv",
        max_episode_steps=1_000_000_000,
    )
    gym.envs.registration.register(
        id=f"UpkieNormExtServosEnv-v{UpkieNormExtServosEnv.version}",
        entry_point="upkie_stand_up.src.envs:UpkieNormExtServosEnv",
        max_episode_steps=1_000,
    )
    gym.envs.registration.register(
        id=f"UpkieNormExtSymServosEnv-v{UpkieNormExtSymServosEnv.version}",
        entry_point="upkie_stand_up.src.envs:UpkieNormExtSymServosEnv",
        max_episode_steps=1_000,
    )
    gym.envs.registration.register(
        id=f"UpkieStandUpEnv-v{UpkieStandUpEnv.version}",
        entry_point="upkie_stand_up.src.envs:UpkieStandUpEnv",
        max_episode_steps=1_000,
    )
    # gym.envs.registration.register(
    #     id=f"UpkieStandUpEnv2-v{UpkieStandUpEnv2.version}",
    #     entry_point="upkie_stand_up.src.envs:UpkieStandUpEnv2",
    #     max_episode_steps=1_000,
    # )
    # gym.envs.registration.register(
    #     id=f"UpkieAlphaEnv-v{UpkieAlphaEnv.version}",
    #     entry_point="upkie_stand_up.src.envs:UpkieAlphaEnv",
    #     max_episode_steps=1_000,
    # )
    # gym.envs.registration.register(
    #     id=f"UpkieBetaEnv-v{UpkieBetaEnv.version}",
    #     entry_point="upkie_stand_up.src.envs:UpkieBetaEnv",
    #     max_episode_steps=1_000,
    # )
    # gym.envs.registration.register(
    #     id=f"UpkieGammaEnv-v{UpkieGammaEnv.version}",
    #     entry_point="upkie_stand_up.src.envs:UpkieGammaEnv",
    #     max_episode_steps=1_000,
    # )
    # gym.envs.registration.register(
    #     id=f"UpkieWheelsNewEnv-v{UpkieWheelsNewEnv.version}",
    #     entry_point="upkie_stand_up.src.envs:UpkieWheelsNewEnv",
    #     max_episode_steps=2_500,
    # )
    for name in final_envs:
        gym.envs.registration.register(
        id=f"{name}-v0",
        entry_point=f"upkie_stand_up.src.envs:{name}",
        max_episode_steps=1_000,
    )
    for name in wheels_envs:
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
]
