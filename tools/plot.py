import pandas as pd
import matplotlib.pyplot as plt

import os
import argparse

parser = argparse.ArgumentParser(description='Experiment plotter')
parser.add_argument('exp_path', type=str, help='Path to the experiment')
args = parser.parse_args()
exp_path = args.exp_path

# Read the CSV file
data = pd.read_csv(os.path.join(exp_path, 'log.txt'))

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