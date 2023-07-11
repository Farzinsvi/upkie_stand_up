import numpy as np

from prettytable import PrettyTable

def print_arguments_as_table(arguments):
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