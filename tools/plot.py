import pandas as pd
import matplotlib.pyplot as plt

import os
import argparse

parser = argparse.ArgumentParser(description='Experiment plotter')
parser.add_argument('--agent', type=str, choices=['SAC', 'TQC'], help='SAC or TQC', required=True)
parser.add_argument('--exp_name', type=str, help='Name of the experiment', required=True)

def plot_experiment(agent_name, exp_name):
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
    plt.ylim(ymin=0, ymax=5000)
    plt.show()

if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()

    # Plot experiment
    plot_experiment(args.agent, args.exp_name)
