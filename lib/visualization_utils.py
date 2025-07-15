from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import copy
from typing import Any

import seaborn as sns
import time
import json
import pandas as pd
from ctypes import c_int32
from itertools import product


def plot_boxplot(dfs, col_y='normalized_error', y_label='normalized error', x_label="approach", 
                 figsize=(8, 6), log_scale=False, palette=None, rotate=False, target = None,  path_to_file=None):
    # Add 'approach' column if missing (assumes each df has a unique approach)
    for df in dfs:
        if 'approach' not in df.columns:
            raise ValueError("Each DataFrame must have an 'approach' column.")

    combined_df = pd.concat(dfs, ignore_index=True)
    plt.figure(figsize=figsize)
    sns.boxplot(x='approach', y=col_y, hue='approach', data=combined_df, palette=palette)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if target is not None:
        median_n_queries = dfs[-1]['query_regions'].median()
        plt.axhline(target*median_n_queries, color='orange', linestyle='-', linewidth=2, label='Target')
        plt.axhline(target, color='red', linestyle='--', linewidth=2, label='Target')
    # plt.title('Comparison of Normalized Error by Approach')
    plt.grid(True, axis='y', alpha=0.5, linestyle='--')
    plt.tight_layout()
    if log_scale:
        plt.yscale('log')
    if rotate:
        plt.xticks(rotation=-45)
    if path_to_file is not None:
        plt.savefig(path_to_file, bbox_inches='tight', pad_inches=0.05) 
    plt.show()

def visualize_badic_cover(b_adic_ranges, show_labels=False):
    """
    Visualize the minimal b-adic cover of a range.
    :param b_adic_ranges: A numpy array of BAdicRange objects.
    :param show_labels: Whether to display the limits of each range as labels.
    """
    if not len(b_adic_ranges):
        print("No ranges to visualize.")
        return

    # Assign colors to different levels
    levels = [r.level for r in b_adic_ranges]
    unique_levels = sorted(set(levels))
    level_colors = {level: plt.cm.viridis(i / len(unique_levels)) for i, level in enumerate(unique_levels)}

    fig, ax = plt.subplots(figsize=(10, 2))

    for i, r in enumerate(b_adic_ranges):
        color = level_colors[r.level]
        ax.barh(0, r.high - r.low, left=r.low, height=0.5, color=color, edgecolor='black', label=f'Level {r.level}' if i == levels.index(r.level) else "")
        if show_labels:
            ax.text((r.low + r.high) / 2, 0, f"[{r.low}, {r.high})", ha='center', va='center', fontsize=8, color='white')

    # Format the plot
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xlabel('Range')
    ax.set_title('Minimal b-Adic Cover Visualization')
    ax.legend(title="Levels", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_b_adic_cubes(cubes):
    """
    Plot BAdicCubes in 2D with colors representing their levels.
    :param cubes: An array of BAdicCubes, where each cube has 2 BAdicRanges and a level.
    """
    # Create a colormap for levels
    levels = sorted(set(cube.level for cube in cubes))
    cmap = plt.cm.get_cmap("tab10", len(levels))  # Adjust color map for number of levels
    level_to_color = {level: cmap(i) for i, level in enumerate(levels)}

    fig, ax = plt.subplots(figsize=(10, 10))

    for cube in cubes:
        # Access the ranges using the correct attribute
        x_range = cube.b_adic_ranges[0]  # First dimension
        y_range = cube.b_adic_ranges[1]  # Second dimension

        # Determine the color based on the cube's level
        color = level_to_color[cube.level]

        # Add a rectangle representing the cube to the plot
        rect = patches.Rectangle(
            (x_range.low, y_range.low),  # Bottom-left corner
            x_range.high - x_range.low,  # Width
            y_range.high - y_range.low,  # Height
            linewidth=1,
            edgecolor="black",
            facecolor=color,
            alpha=0.5
        )
        ax.add_patch(rect)

        # Optionally, add labels showing the bounds of each cube
        label = f"[{x_range.low}, {x_range.high})\n[{y_range.low}, {y_range.high})"
        ax.text(
            x_range.low + (x_range.high - x_range.low) / 2,
            y_range.low + (y_range.high - y_range.low) / 2,
            label,
            fontsize=8,
            color="black",
            ha="center",
            va="center"
        )

    # Set axis limits
    all_x = [range_.low for cube in cubes for range_ in cube.b_adic_ranges[:1]] + \
            [range_.high for cube in cubes for range_ in cube.b_adic_ranges[:1]]
    all_y = [range_.low for cube in cubes for range_ in cube.b_adic_ranges[1:]] + \
            [range_.high for cube in cubes for range_ in cube.b_adic_ranges[1:]]

    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_aspect('equal', adjustable='box')

    # Add a legend for the levels
    handles = [patches.Patch(color=level_to_color[level], label=f"Level {level}") for level in levels]
    ax.legend(handles=handles, title="Levels", loc="upper right")

    # Title and labels
    ax.set_title("B-Adic Cubes Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.show()

def plot_volume_distribution(cubes):
    """
    Plot the volume distribution of BAdicCubes across levels.
    :param cubes: Array of BAdicCubes.
    """
    level_volumes = {}
    for cube in cubes:
        volume = 1
        for r in cube.b_adic_ranges:
            volume *= (r.high - r.low)
        level_volumes[cube.level] = level_volumes.get(cube.level, 0) + volume

    levels = list(level_volumes.keys())
    volumes = list(level_volumes.values())

    plt.bar(levels, volumes, color="skyblue")
    plt.xlabel("Level")
    plt.ylabel("Total Volume")
    plt.title("Volume Distribution Across Levels")
    plt.show()

def plot_b_adic_cubes(cubes):
    """
    Plot BAdicCubes in 2D with colors representing their levels.
    :param cubes: An array of BAdicCubes, where each cube has 2 BAdicRanges and a level.
    """
    # Create a colormap for levels
    levels = sorted(set(cube.level for cube in cubes))
    cmap = plt.cm.get_cmap("tab10", len(levels))  # Adjust color map for number of levels
    level_to_color = {level: cmap(i) for i, level in enumerate(levels)}

    fig, ax = plt.subplots(figsize=(10, 10))

    for cube in cubes:
        # Access the ranges using the correct attribute
        x_range = cube.b_adic_ranges[0]  # First dimension
        y_range = cube.b_adic_ranges[1]  # Second dimension

        # Determine the color based on the cube's level
        color = level_to_color[cube.level]

        # Add a rectangle representing the cube to the plot
        rect = patches.Rectangle(
            (x_range.low, y_range.low),  # Bottom-left corner
            x_range.high - x_range.low,  # Width
            y_range.high - y_range.low,  # Height
            linewidth=1,
            edgecolor="black",
            facecolor=color,
            alpha=0.5
        )
        ax.add_patch(rect)

        # Optionally, add labels showing the bounds of each cube
        label = f"[{x_range.low}, {x_range.high})\n[{y_range.low}, {y_range.high})"
        ax.text(
            x_range.low + (x_range.high - x_range.low) / 2,
            y_range.low + (y_range.high - y_range.low) / 2,
            label,
            fontsize=8,
            color="black",
            ha="center",
            va="center"
        )

    # Set axis limits
    all_x = [range_.low for cube in cubes for range_ in cube.b_adic_ranges[:1]] + \
            [range_.high for cube in cubes for range_ in cube.b_adic_ranges[:1]]
    all_y = [range_.low for cube in cubes for range_ in cube.b_adic_ranges[1:]] + \
            [range_.high for cube in cubes for range_ in cube.b_adic_ranges[1:]]

    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_aspect('equal', adjustable='box')

    # Add a legend for the levels
    handles = [patches.Patch(color=level_to_color[level], label=f"Level {level}") for level in levels]
    ax.legend(handles=handles, title="Levels", loc="upper right")

    # Title and labels
    ax.set_title("B-Adic Cubes Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.show()