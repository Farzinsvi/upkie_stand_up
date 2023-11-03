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

import pandas as pd
import matplotlib.pyplot as plt

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Experiment plotter')
parser.add_argument('--agent', type=str, choices=['SAC', 'TQC'], help='SAC or TQC', required=True)
parser.add_argument('--exp_name', type=str, help='Name of the experiment', required=True)

def plot_experiment(agent_name, exp_name):
    """Plots the training reward curve of a given experiment

    Args:
        agent_name (str): Agent name, namely SAC, TD3 or TQC
        exp_name (str): Experiment name
    """
    # Experiment path
    path = os.path.join(os.getcwd(), 'results', agent_name, exp_name, 'log.txt')

    # Read the CSV file
    data = pd.read_csv(path)

    # Extract the columns
    steps = data.iloc[:, 0]
    rewards = data.iloc[:, 2]

    # Plot the data
    plt.plot(steps, rewards)
    plt.xlabel('Steps')
    plt.ylabel('Episode reward')
    plt.title('Episode Rewards vs. Steps')
    plt.grid(True)
    plt.ylim(ymin=0, ymax=1100)
    plt.show()

# Set some global plot styles suitable for a thesis
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False  # Use LaTeX style for text

def plot_experiments(agent_names, exp_names, plot_range=None, window_size=100, title='Episode Rewards vs. Steps'):
    """
    Plot experiments with optional smoothing.
    
    :param agent_names: List of agent names.
    :param exp_names: List of experiment names.
    :param plot_range: Range of episodes/steps to plot. If None, plots all data.
    :param window_size: Size of the rolling window for smoothing.
    :param title: Title of the plot.
    """
    
    if len(agent_names) != len(exp_names):
        raise ValueError("agent_names and exp_names must have the same length")

    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(agent_names)))  # Using viridis color map
    
    for i, (agent_name, exp_name) in enumerate(zip(agent_names, exp_names)):
        path = os.path.join(os.getcwd(), 'results', agent_name, exp_name, 'log.txt')
        data = pd.read_csv(path)
        
        steps = data.iloc[:plot_range, 0]
        rewards = data.iloc[:plot_range, 2]
        
        smoothed_rewards = rewards.rolling(window=window_size).mean()

        ax.plot(steps, smoothed_rewards, label=exp_name, color=colors[i], linewidth=1.5)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Steps', fontsize=14)
    ax.set_ylabel('Episode Rewards', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12, loc='best')
    fig.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    
    plt.show()

# def plot_experiments(agent_names, exp_names, plot_range=None, title='Episode Rewards vs. Steps'):
    
#     # Experiment path
#     paths = []
#     for agent_name, exp_name in zip(agent_names, exp_names):
#         path = os.path.join(os.getcwd(), 'results', agent_name, exp_name, 'log.txt')

#         # Read the CSV file
#         data = pd.read_csv(path)

#         # Extract the columns
#         steps = data.iloc[:plot_range, 0]
#         rewards = data.iloc[:plot_range, 2]
    
#         print(steps)

#         # Plot the data
#         plt.plot(steps, rewards, label=exp_name)

#     plt.xlabel('Steps')
#     plt.ylabel('Episode reward')
#     plt.title(title)
#     plt.legend(loc='upper left')
#     plt.grid(True)
#     plt.ylim(ymin=0)
#     plt.savefig(f"{title}.png", dpi=150)
#     plt.show()

if __name__ == "__main__":
    # Parse arguments
    # args = parser.parse_args()

    # Plot experiment
    # plot_experiment(args.agent, args.exp_name)
    titles = [
        'SumReward',
        'ProdReward',
        'MinReward',
        'MinHeightProdReward(0.5)',
        'MinHeightProdReward(0.75)'
    ]
    agent_names = ['SAC']*3
    names = np.array(["UpkieAlphaEnv", "UpkieAlphaEnv2", "UpkieAlphaEnv3", "UpkieAlphaEnv4", "UpkieAlphaEnv5", 
        "UpkieBetaEnv", "UpkieBetaEnv2", "UpkieBetaEnv3", "UpkieBetaEnv4", "UpkieBetaEnv5",
        "UpkieGammaEnv", "UpkieGammaEnv2", "UpkieGammaEnv3", "UpkieGammaEnv4", "UpkieGammaEnv5",
        ])
    wheels_names = ["UpkieWheelsNewEnv", "UpkieWheelsNewEnv2", "UpkieWheelsNewEnv3"]
    for idx in range(5):
        exp_names = names[
                [idx + 5*i for i in range(3)]
        ]
        plot_experiments(agent_names, exp_names, plot_range=250, title=titles[idx])
    plot_experiments(['SAC']*3, wheels_names, plot_range=250, title='BalancingReward')
