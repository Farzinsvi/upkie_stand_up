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

import argparse

from upkie_stand_up.src.agents.xpag.train import train_model

def parse_arguments():
    """Parse arguments to train an agent

    Returns:
        parser: Object containing the parser
    """
    parser = argparse.ArgumentParser(description='Train Upkie wheel balancer with SAC')

    parser.add_argument('--agent', type=str, choices=['SAC', 'TQC', 'TD3'], required=True,
                        help='SAC or TQC')
    
    parser.add_argument('--env', type=str, default='UpkieWheelsEnv-v3',
                        help='Training environment')
    
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Name of the experiment')

    parser.add_argument('--max_steps', type=int, default=4_000_000,
                        help='Maximum number of steps for training (default: 1_000_000_000)')
    
    parser.add_argument('--evaluate_every', type=int, default=5_000,
                        help='Evaluate the agent every x steps (default: 5_000)')
    
    parser.add_argument('--save_every', type=int, default=50_000,
                        help='Save the agent every x steps (default: 50_000)')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Train model
    train_model(
        agent_name=args.agent,
        env_name=args.env,
        exp_name=args.exp_name,
        max_steps=args.max_steps,
        evaluate_every=args.evaluate_every,
        save_every=args.save_every
    )

