import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def draw_line(start, end, color='black'):
    plt.plot([start[0], end[0]], [start[1], end[1]], color=color)

def draw_rectangle(start, end, color='black'):
    fig, ax = plt.subplots()
    ax.plot([0, 10], [0, 10], color=color)
    ax.add_patch(plt.Rectangle(start, end[0], end[1], fill=True, color=color))

def pythagorean_tree(start, angle, length, depth):
    if depth == 0:
        return

    end_x = start[0] + length * torch.cos(angle)
    end_y = start[1] + length * torch.sin(angle)

    draw_line(start, (end_x, end_y))

    # Calculate the coordinates for the two child branches
    child_length = length / torch.sqrt(torch.tensor(2.0))
    left_angle = angle + torch.deg2rad(torch.tensor(45.0))
    right_angle = angle - torch.deg2rad(torch.tensor(45.0))

    left_child_start = (end_x, end_y)
    right_child_start = (end_x, end_y)

    left_child_start = (
        left_child_start[0] + child_length * torch.cos(left_angle),
        left_child_start[1] + child_length * torch.sin(left_angle)
    )

    right_child_start = (
        right_child_start[0] + child_length * torch.cos(right_angle),
        right_child_start[1] + child_length * torch.sin(right_angle)
    )

    # Recursively draw the child branches
    pythagorean_tree(left_child_start, left_angle, child_length, depth-1)
    pythagorean_tree(right_child_start, right_angle, child_length, depth-1)

if __name__ == '__main__':
    # Set up the plot
    plt.figure(figsize=(8, 8))
    plt.axis('equal')
    plt.axis('off')

    # Starting position and initial angle
    start_point = (0, 0)
    initial_angle = torch.deg2rad(torch.tensor(90.0))

    # Length of the initial branch and the depth of recursion
    initial_length = 100
    depth = 15

    # Draw the Pythagorean tree fractal
    pythagorean_tree(start_point, initial_angle, initial_length, depth)

    # Show the plot
    plt.show()