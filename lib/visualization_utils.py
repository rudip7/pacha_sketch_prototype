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
from numbers import Number
from matplotlib.ticker import MaxNLocator

# def plot_barplot_heuristics(
#     dfs,
#     col_y='runtime',
#     y_label='runtime (ms)',
#     palette=None, 

def plot_boxplot_heuristics(
    dfs, 
    col_y='normalized_error', 
    y_label='norm. error', 
    lower_move=0.09,
    figsize=(8, 6), 
    log_scale=False, 
    palette=None, 
    rotation=False, 
    path_to_file=None,
    show_legend=False,
    relative_entropies=None,
    label_prefix="w"
) -> plt.Figure:
    # Add 'approach' column if missing
    for df in dfs:
        if 'approach' not in df.columns:
            raise ValueError("Each DataFrame must have an 'approach' column.")

    combined_df = pd.concat(dfs, ignore_index=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=200.0)

    bp = sns.boxplot(
        x='approach',
        y=col_y,
        hue='approach',
        data=combined_df,
        palette=palette,
        ax=ax,
        showfliers=False
    )

    approaches = combined_df['approach'].unique()

    # Style patches depending on approach
    for patch, label in zip(ax.patches, approaches):
        if label.startswith('w') or label_prefix == None:
            patch.set_facecolor(patch.get_facecolor())
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)
            patch.set_hatch('')
        else:
            patch.set_edgecolor(patch.get_facecolor())
            patch.set_facecolor('white')
            patch.set_linewidth(1.5)
            patch.set_hatch('XX')

    tick_fontsize = ax.get_xticklabels()[0].get_fontsize() 

    rel_entropy_color = 'tab:red'

    # === ✅ Add relative entropy values above each box ===
    if relative_entropies is not None:
        if len(relative_entropies) != len(approaches):
            raise ValueError(
                f"Expected {len(approaches)} relative entropy values, "
                f"but got {len(relative_entropies)}."
            )

        # Create secondary y-axis
        ax2 = ax.twinx()
        ax2.set_ylabel("rel. entropy", color=rel_entropy_color)
        ax2.tick_params(axis='y', labelcolor=rel_entropy_color)
        ax2.spines['right'].set_color(rel_entropy_color)
        ax2.tick_params(axis='y', colors=rel_entropy_color)

        # Plot relative entropy as scatter on top of boxplots
        for i, value in enumerate(relative_entropies):
            x_pos = i  # box positions are integers in seaborn
            ax2.scatter(x_pos, value, color=rel_entropy_color, zorder=5, marker='x', alpha=0.8)
        
        # Optional: set secondary y-limits a bit higher for visibility
        ax2.set_ylim(0, max(relative_entropies) * 1.2)



    ax.set_xlabel('')
    ax.set_ylabel(y_label)
    ax.grid(True, axis='y', linestyle='--')

    if log_scale:
        ax.set_yscale('log')

    # === Centered dataset labels
    fig.canvas.draw()
    raw_tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    split_labels = [lbl.split('-', 1)[-1] if '-' in lbl else lbl for lbl in raw_tick_labels]

    datasets = []
    for s in split_labels:
        if not datasets or s != datasets[-1]:
            datasets.append(s)

    n_methods = int(len(raw_tick_labels) / len(datasets))
    xticks = ax.get_xticks()
    midpoints = []
    for i in range(len(datasets)):
        group = xticks[i * n_methods:(i + 1) * n_methods]
        midpoints.append(np.mean(group))

    ax.set_xticks(midpoints)
    ax.set_xticklabels(datasets, rotation=rotation, ha='center')

    
    if not show_legend and ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()

    if path_to_file:
        fig.savefig(path_to_file, bbox_inches='tight', pad_inches=0.05)

    return fig, ax


def plot_relative_error(
    results: list[pd.DataFrame],
    labels: list[str],
    x_label: str = '~ nr. rows',
    figsize: tuple[int, int] = (5, 3),
    color: str = 'tab:blue'
):
    fig, ax = plt.subplots(figsize=figsize)
    medians = [1 - df['relative_error'].median() for df in results]
    q25 = [1 - df['relative_error'].quantile(0.25) for df in results]
    q75 = [1 - df['relative_error'].quantile(0.75) for df in results]

    ax.axhline(1, color='red', linestyle='-', label='ground truth')
    ax.plot(range(len(medians)), medians, color=color, marker='o', label='median')
    ax.fill_between(range(len(medians)), q25, q75, color=color, alpha=0.2, label='Q25-Q75')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel('relative accuracy')
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    plt.grid(True, axis='y', alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()

    return fig

def bar_plot(
    results : list[Number],
    labels: list[str],
    y_label='relative entropy', 
    x_label="configuration", 
    figsize: tuple[int, int] = (5, 3),
    palette=None
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, axis='y', linestyle='--')

    ax.bar(
        range(len(labels)),
        results,
        color=[palette[label] for label in labels],
        capsize=5,
        edgecolor='black',
        alpha=0.8
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.legend()

    # handles, labels = plt.gca().get_legend_handles_labels()
    plt.tight_layout()
    return fig

def bar_plot_no_outliers(
    results : list,
    labels: list[str],
    y_label='relative entropy',
    lower_move=0.1, 
    figsize: tuple[int, int] = (5, 3),
    palette=None
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, axis='y', linestyle='--')

    ax.bar(
        range(len(labels)),
        results,
        color=[palette[label] for label in labels],
        capsize=5,
        edgecolor='black',
        width=0.8
    )

    for patch, label in zip(ax.patches, labels):
        if label.startswith('W'):
            patch.set_facecolor(patch.get_facecolor())  # keep color
            patch.set_linewidth(1.0)
            patch.set_hatch('')
            patch.set_alpha(0.8)
        else:  # baseline 'o-'
            patch.set_facecolor(patch.get_facecolor())  # keep color
            patch.set_linewidth(1.0)
            patch.set_hatch('')
            patch.set_alpha(0.4)

    # --- Determine dataset groups ---
    tick_positions = np.arange(len(labels))
    split_labels = [lbl.split('-', 1)[-1] if '-' in lbl else lbl for lbl in labels]
    datasets = []
    for s in split_labels:
        if not datasets or s != datasets[-1]:
            datasets.append(s)

    n_methods = int(len(labels) / len(datasets))
    midpoints = []
    for i in range(len(datasets)):
        group = tick_positions[i * n_methods:(i + 1) * n_methods]
        midpoints.append(np.mean(group))

    ax.set_xticks(midpoints)
    ax.set_xticklabels(datasets, ha='center')
    # ax.set_xlabel('dataset')

    # --- Add P/O labels in axes coordinates ---
    tick_fontsize = ax.get_xticklabels()[0].get_fontsize()
    smaller_fontsize = tick_fontsize * 0.8
    for i, tick in enumerate(tick_positions):
        if n_methods == 2:
            label = "w" if i % 2 == 0 else "n"
        ax.text(
            tick,
            -lower_move,  # 5% below the x-axis
            label,
            ha='center',
            va='top',
            color='black', fontweight='bold',
            fontsize=smaller_fontsize ,
            transform=ax.get_xaxis_transform()  # <-- use axis coordinate system
        )

    fig.subplots_adjust(bottom=0.18)
    ax.set_ylabel(y_label)
    # ax.legend()

    # handles, labels = plt.gca().get_legend_handles_labels()
    plt.tight_layout()
    return fig


def legend_bar_plot_p_vs_o(
    results : list,
    labels: list[str],
    y_label='throughput (updt./s)',
    lower_move=0.1, 
    figsize: tuple[int, int] = (5, 3),
    palette=None
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, axis='y', linestyle='--')

    ax.bar(
        range(len(labels)),
        results,
        color=[palette[label] for label in labels],
        capsize=5,
        edgecolor='black',
        width=0.8
    )

    for patch, label in zip(ax.patches, labels):
        if label.startswith('Ps'):
            patch.set_facecolor(patch.get_facecolor())  # keep color
            patch.set_linewidth(1.0)
            patch.set_hatch('')
            patch.set_alpha(0.6)
        elif label.startswith('P'):
            patch.set_facecolor(patch.get_facecolor())  # keep color
            patch.set_linewidth(1.0)
            patch.set_hatch('')
            patch.set_alpha(0.8)
        else:  # baseline 'o-'
            patch.set_edgecolor(patch.get_facecolor())
            patch.set_facecolor('white')
            patch.set_linewidth(1.5)
            patch.set_hatch('///')

    fig.subplots_adjust(bottom=0.18)
    ax.set_ylabel(y_label)
    # ax.legend()

    # handles, labels = plt.gca().get_legend_handles_labels()
    plt.tight_layout()
    return fig, ax

def bar_plot_p_vs_o(
    results : list,
    labels: list[str],
    y_label='throughput (up./s)',
    lower_move=0.1, 
    figsize: tuple[int, int] = (5, 3),
    palette=None
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, axis='y', linestyle='--')

    ax.bar(
        range(len(labels)),
        results,
        color=[palette[label] for label in labels],
        capsize=5,
        edgecolor='black',
        width=0.8
    )

    for patch, label in zip(ax.patches, labels):
        if label.startswith('Ps'):
            patch.set_facecolor(patch.get_facecolor())  # keep color
            patch.set_linewidth(1.0)
            patch.set_hatch('')
            patch.set_alpha(0.6)
        elif label.startswith('P'):
            patch.set_facecolor(patch.get_facecolor())  # keep color
            patch.set_linewidth(1.0)
            patch.set_hatch('')
            patch.set_alpha(0.8)
        else:  # baseline 'o-'
            patch.set_edgecolor(patch.get_facecolor())
            patch.set_facecolor('white')
            patch.set_linewidth(1.5)
            patch.set_hatch('///')

    # --- Determine dataset groups ---
    tick_positions = np.arange(len(labels))
    split_labels = [lbl.split('-', 1)[-1] if '-' in lbl else lbl for lbl in labels]
    datasets = []
    for s in split_labels:
        if not datasets or s != datasets[-1]:
            datasets.append(s)

    n_methods = int(len(labels) / len(datasets))
    midpoints = []
    for i in range(len(datasets)):
        group = tick_positions[i * n_methods:(i + 1) * n_methods]
        midpoints.append(np.mean(group))

    ax.set_xticks(midpoints)
    ax.set_xticklabels(datasets, ha='center')
    # ax.set_xlabel('dataset')

    # --- Add P/O labels in axes coordinates ---
    tick_fontsize = ax.get_xticklabels()[0].get_fontsize()
    smaller_fontsize = tick_fontsize * 0.8
    for i, tick in enumerate(tick_positions):
        if n_methods == 2:
            label = "PS" if i % 2 == 0 else "OS"
        else:
            label = "PS" if i % 3 == 0 else ("PSs" if i % 3 == 1 else "OS")
        ax.text(
            tick,
            -lower_move,  # 5% below the x-axis
            label,
            ha='center',
            va='top',
            color='black', fontweight='bold',
            fontsize=smaller_fontsize ,
            transform=ax.get_xaxis_transform()  # <-- use axis coordinate system
        )

    fig.subplots_adjust(bottom=0.18)
    ax.set_ylabel(y_label)
    ax.set_yticks([10e3, 20e3, 30e3])
    ax.set_yticklabels([10, 20,30])
    # ax.legend()

    # handles, labels = plt.gca().get_legend_handles_labels()
    plt.tight_layout()
    return fig

def plot_scatter(
    results : list[pd.DataFrame],
    x_col = 'query_regions',
    y_col = 'normalized_error',
    x_max = None,
    figsize: tuple[int, int] = (5, 3),
    x_label ='nr query regions',
    y_label = 'normalized error',
    step=None,
    palette=None,
    markers=None
):

    fig, ax = plt.subplots(figsize=figsize)
    if markers is None:
        markers = ['o'] * len(results)

    for i, df in enumerate(results):
        if x_col == "total_sketch_queries":
            df["total_sketch_queries"] = df[
                ["relevant_nodes", "b_adic_cubes", "candidate_regions", "query_regions"]
            ].sum(axis=1)

        df = df[df['forced'] == -1] 
        if x_max is not None:
            df = df[df[x_col] < x_max] 

        label = df['approach'].iloc[0]
        ax.scatter(
            df[x_col],
            df[y_col],
            alpha=0.5,
            label=label,
            color=palette[label],
            marker=markers[i]
        )

    if step is not None:
        ax.set_xticks(range(0, x_max + 1, step))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.legend()
    ax.grid(True, linestyle='--')
    plt.tight_layout()

    return fig


def error_bar_plot_p_vs_o(
    results: list[pd.DataFrame],
    col_y='normalized_error',
    y_label='runtime (ms)', 
    figsize: tuple[int, int] = (5, 3),
    lower_bound: float = 0.0,
    log_scale=True,
    lower_move=0.1,
    palette=None
):
    fig, ax = plt.subplots(figsize=figsize)
    medians = [df[col_y].median() for df in results]
    q25 = [df[col_y].quantile(0.25) for df in results]
    q75 = [df[col_y].quantile(0.75) for df in results]

    lower_err = np.array(medians) - np.array(q25)
    upper_err = np.array(q75) - np.array(medians)

    # combined_df = pd.concat(results, ignore_index=True)
    labels = [df['approach'].iloc[0] for df in results]

    yerr = np.array([lower_err, upper_err])
    bars = ax.bar(
        range(len(labels)),
        medians,
        yerr=yerr,
        align='center',
        color=[palette[label] for label in labels],
        capsize=1,
        edgecolor='black',
        width=0.8
    )

    # --- Style bars ---
    for patch, label in zip(bars, labels):
        if label.startswith('P'):
            patch.set_alpha(0.8)
        else:  # baseline 'O-'
            patch.set_edgecolor(patch.get_facecolor())
            patch.set_facecolor('white')
            patch.set_hatch('///')
            patch.set_linewidth(1.5)

    # --- Determine dataset groups ---
    tick_positions = np.arange(len(labels))
    split_labels = [lbl.split('-', 1)[-1] if '-' in lbl else lbl for lbl in labels]
    datasets = []
    for s in split_labels:
        if not datasets or s != datasets[-1]:
            datasets.append(s)

    n_methods = int(len(labels) / len(datasets))
    midpoints = []
    for i in range(len(datasets)):
        group = tick_positions[i * n_methods:(i + 1) * n_methods]
        midpoints.append(np.mean(group))

    ax.set_xticks(midpoints)
    ax.set_xticklabels(datasets, ha='center')
    # ax.set_xlabel('dataset')

    # --- Add P/O labels in axes coordinates ---
    tick_fontsize = ax.get_xticklabels()[0].get_fontsize()
    smaller_fontsize = tick_fontsize * 0.8
    for i, tick in enumerate(tick_positions):
        label = "PS" if i % 2 == 0 else "OS"
        ax.text(
            tick,
            -lower_move,
            label,
            ha='center',
            va='top',
            color='black',
            fontsize=smaller_fontsize ,
            transform=ax.get_xaxis_transform()  # <-- use axis coordinate system
        )

    fig.subplots_adjust(bottom=0.18)

    if log_scale:
        ax.set_yscale('log')
    else:
        ax.set_ylim(lower_bound, None)

    ax.set_ylabel(y_label)
    
    # ax.legend()
    ax.grid(True, axis='y', linestyle='--')
    plt.tight_layout()
    return fig

def plot_relative_accuracy_p_vs_o(
    results: list[pd.DataFrame],
    figsize: tuple[int, int] = (5, 3),
    lower_bound: float = 0.0,
    lower_move=0.1,
    palette=None
):
    fig, ax = plt.subplots(figsize=figsize)
    medians = [1 - df['relative_error'].median() for df in results]
    q25 = [1 - df['relative_error'].quantile(0.25) for df in results]
    q75 = [1 - df['relative_error'].quantile(0.75) for df in results]

    lower_err = np.array(medians) - np.array(q75)
    upper_err = np.array(q25) - np.array(medians)

    combined_df = pd.concat(results, ignore_index=True)
    labels = combined_df['approach'].unique()

    ax.axhline(1, color='red', linestyle='-', label='ground truth')
    bars = ax.bar(
        range(len(labels)),
        medians,
        yerr=[lower_err, upper_err],
        align='center',
        color=[palette[label] for label in labels],
        capsize=1,
        edgecolor='black',
        width=0.8
    )

    # --- Style bars ---
    for patch, label in zip(bars, labels):
        if label.startswith('P'):
            patch.set_alpha(0.8)
        else:  # baseline 'O-'
            patch.set_edgecolor(patch.get_facecolor())
            patch.set_facecolor('white')
            patch.set_linewidth(1.5)
            patch.set_hatch('///')

    # --- Determine dataset groups ---
    tick_positions = np.arange(len(labels))
    split_labels = [lbl.split('-', 1)[-1] if '-' in lbl else lbl for lbl in labels]
    datasets = []
    for s in split_labels:
        if not datasets or s != datasets[-1]:
            datasets.append(s)

    n_methods = int(len(labels) / len(datasets))
    midpoints = []
    for i in range(len(datasets)):
        group = tick_positions[i * n_methods:(i + 1) * n_methods]
        midpoints.append(np.mean(group))

    ax.set_xticks(midpoints)
    ax.set_xticklabels(datasets, ha='center')
    # ax.set_xlabel('dataset')

    # --- Add P/O labels in axes coordinates ---
    tick_fontsize = ax.get_xticklabels()[0].get_fontsize()
    smaller_fontsize = tick_fontsize * 0.8
    for i, tick in enumerate(tick_positions):
        label = "PS" if i % 2 == 0 else "OS"
        ax.text(
            tick,
            -lower_move,  # 5% below the x-axis
            label,
            ha='center',
            va='top',
            color='black',
            fontsize=smaller_fontsize ,
            transform=ax.get_xaxis_transform()  # <-- use axis coordinate system
        )

    fig.subplots_adjust(bottom=0.18)

    ax.set_ylabel('relative accuracy')
    ax.set_ylim(lower_bound, 1.05)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--')
    plt.tight_layout()
    return fig


def plot_violinplot_p_vs_o(
    dfs, 
    col_y='normalized_error', 
    y_label='normalized error', 
    lower_move=0.09,
    figsize=(8, 6), 
    log_scale=False, 
    palette=None, 
    rotation=False, 
    path_to_file=None,
    show_legend=False
) -> plt.Figure:
    # Add 'approach' column if missing (assumes each df has a unique approach)
    for df in dfs:
        if 'approach' not in df.columns:
            raise ValueError("Each DataFrame must have an 'approach' column.")

    combined_df = pd.concat(dfs, ignore_index=True)
    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        x='approach',
        y=col_y, 
        hue='approach', 
        data=combined_df, 
        palette=palette, 
        ax=ax,
        cut=0,          # avoid extending beyond observed range
        inner='quartile', # draw box-style quartile lines inside violin
        # linewidth=1.5
    )

    approaches = combined_df['approach'].unique()
    for i, c in enumerate(ax.collections):
        fc = c.get_facecolor()
        if i % 2 == 0:
            c.set_alpha(0.8)
            c.set_edgecolor('black')
            # c.set_linewidth(1.2)
        else:
            c.set_facecolor('white')     # white fill
            c.set_edgecolor(fc)
            c.set_linewidth(1.5)

    ax.set_xlabel('')
    ax.set_ylabel(y_label)

    ax.grid(True, axis='y', linestyle='--')

    if log_scale:
        ax.set_yscale('log')

       # === Centered dataset labels (as before) ===
    fig.canvas.draw()
    raw_tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    split_labels = [lbl.split('-', 1)[-1] if '-' in lbl else lbl for lbl in raw_tick_labels]
    datasets = []
    for s in split_labels:
        if not datasets or s != datasets[-1]:
            datasets.append(s)

    n_methods = int(len(raw_tick_labels) / len(datasets))
    xticks = ax.get_xticks()
    midpoints = []
    for i in range(len(datasets)):
        group = xticks[i * n_methods:(i + 1) * n_methods]
        midpoints.append(np.mean(group))

    ax.set_xticks(midpoints)
    ax.set_xticklabels(datasets, rotation=rotation, ha='center')

    # === Add method labels ("P", "O") under each dataset ===
    ylim = ax.get_ylim()
    if ax.get_yscale() == 'log':
        # multiplicative offset for log scale (move slightly below lower limit)
        y_pos = ylim[0] / (ylim[1] / ylim[0]) ** lower_move
    else:
        # additive offset for linear scale
        y_pos = ylim[0] - (ylim[1] - ylim[0]) * 0.05

    for i, tick in enumerate(xticks):
        # alternate between P and O (assuming 2 per dataset)
        label = "PS" if i % 2 == 0 else "OS"
        ax.text(
            tick, y_pos, label,
            ha='center', va='top',
            color='black'
        )

    # adjust limits so labels fit
    # ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * 0.1, ylim[1])

    if rotation:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='center')

    if not show_legend and ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()

    if path_to_file:
        fig.savefig(path_to_file, bbox_inches='tight', pad_inches=0.05)

    return fig

def plot_boxplot_p_vs_s(
    dfs, 
    col_y='relative_error', 
    y_label='relative error', 
    lower_move=0.09,
    figsize=(8, 6), 
    log_scale=False, 
    palette=None, 
    rotation=False, 
    path_to_file=None,
    show_legend=False,
    relative_entropies=None,
    plot_sub_title = True
) -> plt.Figure:
    # Add 'approach' column if missing
    for df in dfs:
        if 'approach' not in df.columns:
            raise ValueError("Each DataFrame must have an 'approach' column.")

    combined_df = pd.concat(dfs, ignore_index=True)
    fig, ax = plt.subplots(figsize=figsize)

    bp = sns.boxplot(
        x='approach',
        y=col_y,
        hue='approach',
        data=combined_df,
        palette=palette,
        ax=ax,
        showfliers=False
    )

    approaches = combined_df['approach'].unique()

    # Style patches depending on approach
    for patch, label in zip(ax.patches, approaches):
        if 'P' in label:
            patch.set_facecolor(patch.get_facecolor())
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)
            patch.set_hatch('')
        else:
            patch.set_edgecolor(patch.get_facecolor())
            patch.set_facecolor('white')
            patch.set_linewidth(1.5)
            patch.set_hatch('XXXX')

    tick_fontsize = ax.get_xticklabels()[0].get_fontsize() 

    rel_entropy_color = 'tab:red'

    # === ✅ Add relative entropy values above each box ===
    if relative_entropies is not None:
        if len(relative_entropies) != len(approaches):
            raise ValueError(
                f"Expected {len(approaches)} relative entropy values, "
                f"but got {len(relative_entropies)}."
            )

        # Create secondary y-axis
        ax2 = ax.twinx()
        ax2.set_ylabel("relative entropy", color=rel_entropy_color)
        ax2.tick_params(axis='y', labelcolor=rel_entropy_color)
        ax2.spines['right'].set_color(rel_entropy_color)
        ax2.tick_params(axis='y', colors=rel_entropy_color)

        # Plot relative entropy as scatter on top of boxplots
        for i, value in enumerate(relative_entropies):
            x_pos = i  # box positions are integers in seaborn
            ax2.scatter(x_pos, value, color=rel_entropy_color, zorder=5, marker='x', alpha=0.8)
        
        # Optional: set secondary y-limits a bit higher for visibility
        ax2.set_ylim(0, max(relative_entropies) * 1.2)

    ax.set_xlabel('')
    ax.set_ylabel(y_label)
    ax.grid(True, axis='y', linestyle='--')

    if log_scale:
        ax.set_yscale('log')

    # === Centered dataset labels
    fig.canvas.draw()
    raw_tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    split_labels = [lbl.split('-', 1)[0] if '-' in lbl else lbl for lbl in raw_tick_labels]

    aggregate = []
    for s in split_labels:
        if not aggregate or s != aggregate[-1]:
            aggregate.append(s)

    n_methods = int(len(raw_tick_labels) / len(aggregate))
    xticks = ax.get_xticks()
    midpoints = []
    for i in range(len(aggregate)):
        group = xticks[i * n_methods:(i + 1) * n_methods]
        midpoints.append(np.mean(group))

    ax.set_xticks(midpoints)
    ax.set_xticklabels(aggregate, rotation=rotation, ha='center')

    # === Add method labels ("P", "O") under each dataset
    if plot_sub_title:
        ylim = ax.get_ylim()
        if ax.get_yscale() == 'log':
            y_pos = ylim[0] / (ylim[1] / ylim[0]) ** lower_move
        else:
            y_pos = ylim[0] - (ylim[1] - ylim[0]) * 0.05

        tick_fontsize = ax.get_xticklabels()[0].get_fontsize()
        smaller_fontsize = tick_fontsize * 0.8

        for i, tick in enumerate(xticks):
            label = "PS" if i % 2 == 0 else "S"
            ax.text(
                tick, y_pos, label,
                ha='center', va='top',
                color='black',
                fontsize=smaller_fontsize
            )

    if rotation:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='center')

    if not show_legend and ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()

    if path_to_file:
        fig.savefig(path_to_file, bbox_inches='tight', pad_inches=0.05)

    return fig


def plot_boxplot_p_vs_o(
    dfs, 
    col_y='normalized_error', 
    y_label='norm. error', 
    lower_move=0.09,
    figsize=(8, 6), 
    log_scale=False, 
    palette=None, 
    rotation=False, 
    path_to_file=None,
    show_legend=False,
    relative_entropies=None
) -> plt.Figure:
    # Add 'approach' column if missing
    for df in dfs:
        if 'approach' not in df.columns:
            raise ValueError("Each DataFrame must have an 'approach' column.")

    combined_df = pd.concat(dfs, ignore_index=True)
    fig, ax = plt.subplots(figsize=figsize)

    bp = sns.boxplot(
        x='approach',
        y=col_y,
        hue='approach',
        data=combined_df,
        palette=palette,
        ax=ax,
        showfliers=False
    )

    approaches = combined_df['approach'].unique()

    # Style patches depending on approach
    for patch, label in zip(ax.patches, approaches):
        if label.startswith('P'):
            patch.set_facecolor(patch.get_facecolor())
            patch.set_linewidth(1.0)
            patch.set_alpha(0.8)
            patch.set_hatch('')
        else:
            patch.set_edgecolor(patch.get_facecolor())
            patch.set_facecolor('white')
            patch.set_linewidth(1.5)
            patch.set_hatch('////')

    tick_fontsize = ax.get_xticklabels()[0].get_fontsize() 

    rel_entropy_color = 'tab:red'

    # === ✅ Add relative entropy values above each box ===
    if relative_entropies is not None:
        if len(relative_entropies) != len(approaches):
            raise ValueError(
                f"Expected {len(approaches)} relative entropy values, "
                f"but got {len(relative_entropies)}."
            )

        # Create secondary y-axis
        ax2 = ax.twinx()
        ax2.set_ylabel("rel. entropy", color=rel_entropy_color)
        ax2.tick_params(axis='y', labelcolor=rel_entropy_color)
        ax2.spines['right'].set_color(rel_entropy_color)
        ax2.tick_params(axis='y', colors=rel_entropy_color)

        # Plot relative entropy as scatter on top of boxplots
        for i, value in enumerate(relative_entropies):
            x_pos = i  # box positions are integers in seaborn
            ax2.scatter(x_pos, value, color=rel_entropy_color, zorder=5, marker='x', alpha=0.8)
        
        # Optional: set secondary y-limits a bit higher for visibility
        ax2.set_ylim(0, max(relative_entropies) * 1.2)



    ax.set_xlabel('')
    ax.set_ylabel(y_label)
    ax.grid(True, axis='y', linestyle='--')

    if log_scale:
        ax.set_yscale('log')

    # === Centered dataset labels
    fig.canvas.draw()
    raw_tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    split_labels = [lbl.split('-', 1)[-1] if '-' in lbl else lbl for lbl in raw_tick_labels]

    datasets = []
    for s in split_labels:
        if not datasets or s != datasets[-1]:
            datasets.append(s)

    n_methods = int(len(raw_tick_labels) / len(datasets))
    xticks = ax.get_xticks()
    midpoints = []
    for i in range(len(datasets)):
        group = xticks[i * n_methods:(i + 1) * n_methods]
        midpoints.append(np.mean(group))

    ax.set_xticks(midpoints)
    ax.set_xticklabels(datasets, rotation=rotation, ha='center')

    # === Add method labels ("P", "O") under each dataset
    ylim = ax.get_ylim()
    if ax.get_yscale() == 'log':
        y_pos = ylim[0] / (ylim[1] / ylim[0]) ** lower_move
    else:
        y_pos = ylim[0] - (ylim[1] - ylim[0]) * 0.05

    tick_fontsize = ax.get_xticklabels()[0].get_fontsize()
    smaller_fontsize = tick_fontsize * 0.8

    for i, tick in enumerate(xticks):
        label = "PS" if i % 2 == 0 else "OS"
        ax.text(
            tick, y_pos, label,
            ha='center', va='top',
            color='black',
            fontsize=smaller_fontsize
        )

    if rotation:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='center')

    if not show_legend and ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()

    if path_to_file:
        fig.savefig(path_to_file, bbox_inches='tight', pad_inches=0.05)

    return fig


def plot_lineplot_p_vs_o(
    dfs, 
    col_y='normalized_error', 
    y_label='norm. error', 
    figsize=(8, 6), 
    log_scale=False, 
    palette=None, 
    rotation=False, 
    path_to_file=None,
    x_label='nr. predicates',
    show_legend=False
) -> plt.Figure:
    # Add 'approach' column if missing (assumes each df has a unique approach)
    for df in dfs:
        if 'approach' not in df.columns:
            raise ValueError("Each DataFrame must have an 'approach' column.")

    fig, ax = plt.subplots(figsize=figsize)

    medians = np.asarray([df[col_y].median() for df in dfs])
    q25 = np.asarray([df[col_y].quantile(0.25) for df in dfs])
    q75 = np.asarray([df[col_y].quantile(0.75) for df in dfs])

    pacha_indices = np.arange(0, len(dfs), 2)
    omni_indices = np.arange(1, len(dfs), 2)

    combined_df = pd.concat(dfs, ignore_index=True)
    approaches = combined_df['approach'].unique()
    split_labels = [lbl.split('-', 1)[-1] if '-' in lbl else lbl for lbl in approaches]
    labels = []
    for s in split_labels:
        if s not in labels:
            labels.append(s)

    ax.plot(labels, medians[pacha_indices], label='Pacha Sketch', color=palette['pacha'])
    # ax.plot(labels, medians[pacha_indices], marker='o', label='Pacha Sketch', color=palette['pacha'])
    ax.fill_between(labels, q25[pacha_indices], q75[pacha_indices], color=palette['pacha'], alpha=0.2)

    ax.plot(labels, medians[omni_indices],  label='Omni Sketch', linestyle='--', color=palette['omni'])
    # ax.plot(labels, medians[omni_indices], marker='x', label='Omni Sketch', color=palette['omni'])
    ax.fill_between(labels, q25[omni_indices], q75[omni_indices], color=palette['omni'], alpha=0.2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.ticklabel_format(axis='y', style='sci')

    ax.grid(True, axis='y', linestyle='--')

    if log_scale:
        ax.set_yscale('log')

    if rotation:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='center')

    if not show_legend and ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()

    if path_to_file:
        fig.savefig(path_to_file, bbox_inches='tight', pad_inches=0.05)

    return fig


def plot_selectivities(
        dataset_name, 
        base_results_path = '../results/experiments_results/',
        figsize = (8, 6),
        lower_move=0.20):
    # selectivities = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64])
    datasets_palette = {
            'tpch' : 'tab:blue',
            'retail' : 'tab:orange',
            'census' : 'tab:purple',
            'bank' : 'tab:green'
        }
    
    selectivities = np.array([0.01, 0.04, 0.16, 0.64])
    approaches = ['pacha', 'omni'] 

    results = []
    for appoach in approaches:
        for sel in selectivities:
            result_df = pd.read_csv(f"{base_results_path}{appoach}/{dataset_name}/selectivities/{dataset_name}_sel_{sel}.csv")
            result_df['approach'] = f'{appoach[0].upper()}-'+str(sel)
            results.append(result_df)

    n_measurements = len(selectivities)
    index_array = [i + j * n_measurements for i in range(n_measurements) for j in range(2)]
    results = [results[i] for i in index_array]

    combined_df = pd.concat(results, ignore_index=True)
    labels = combined_df['approach'].unique()
    # custom_palette = {}
    # for label in labels:
    #     custom_palette[label] = datasets_palette[dataset_name]

    custom_palette = {"pacha" : "tab:blue", "omni" : "tab:blue"}

    return plot_lineplot_p_vs_o(results, x_label='selectivity', log_scale=True, figsize=figsize, palette=custom_palette)
    # return plot_boxplot_p_vs_o(results, lower_move=lower_move, log_scale=True, figsize=figsize, palette=custom_palette)

def plot_predicates(
        dataset_name, 
        predicates_type:str,
        base_results_path = '../results/experiments_results/',
        figsize = (8, 6),
        lower_move=0.20):
    
    datasets_palette = {
            'tpch' : 'tab:blue',
            'retail' : 'tab:orange',
            'census' : 'tab:purple',
            'bank' : 'tab:green'
        }
    
    n_num_dims = {
        'tpch': 5,
        'retail': 3,
        'census': 3,
        'bank': 4
    }

    n_cat_dims = {
        'tpch': 5,
        'retail': 3,
        'census': 7,
        'bank': 6
    }

    
    if predicates_type == 'numerical':
        n_dims = np.arange(1, n_num_dims[dataset_name] + 1)
        labels = n_dims.copy()
    elif predicates_type == 'categorical':
        n_dims = np.arange(1, n_cat_dims[dataset_name] + 1)
        labels = n_dims.copy()
    elif predicates_type == 'mixed':
        dom_dims = n_cat_dims[dataset_name]
        n_dims = np.arange(1, dom_dims + 1)
        labels = n_dims.copy()
        n = 1
        for i in n_dims - 1:
            labels[i] += n
            if n < n_num_dims[dataset_name]:
                n += 1
        
    else:
        raise ValueError("predicates_type must be 'numerical', 'categorical', or 'mixed'")

    short_name = predicates_type.replace('numerical', 'num').replace('categorical', 'cat').replace('mixed', 'mix')
    
    results = []
    approaches = ['pacha', 'omni']
    for appoach in approaches: 
        for i, n in enumerate(n_dims):
            result_df = pd.read_csv(f"{base_results_path}{appoach}/{dataset_name}/{predicates_type}/{dataset_name}_{short_name}_{n}.csv")
            result_df['approach'] = f'{appoach[0].upper()}-'+str(labels[i])
            # print(result_df['normalized_error'].median())
            results.append(result_df)
    
    n_measurements = len(n_dims)
    index_array = [i + j * n_measurements for i in range(n_measurements) for j in range(2)]
    results = [results[i] for i in index_array]

    combined_df = pd.concat(results, ignore_index=True)
    labels = combined_df['approach'].unique()
    # custom_palette = {}
    # for n in labels:
    #     custom_palette[n] = datasets_palette[dataset_name]

    custom_palette = {"pacha" : "tab:blue", "omni" : "tab:blue"}

    return plot_lineplot_p_vs_o(results, log_scale=True, figsize=figsize, palette=custom_palette)
    
    # return plot_boxplot_p_vs_o(results, lower_move=lower_move, log_scale=True, figsize=figsize, palette=custom_palette)

def plot_boxplot(
    dfs, 
    col_y='normalized_error', 
    y_label='norm. error', 
    x_label="approach", 
    figsize=(8, 6), 
    log_scale=False, 
    palette=None, 
    rotation=False,
    scale_x_ticks=None, 
    target=None,  
    path_to_file=None,
    show_legend=False,
    relative_entropies=None,
    rel_entropies_no_outliers=None
) -> plt.Figure:
    # Add 'approach' column if missing (assumes each df has a unique approach)
    for df in dfs:
        if 'approach' not in df.columns:
            raise ValueError("Each DataFrame must have an 'approach' column.")

    combined_df = pd.concat(dfs, ignore_index=True)
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(x='approach', y=col_y, hue='approach', data=combined_df, palette=palette, ax=ax,
                flierprops=dict(markersize=1, markeredgecolor=(0, 0, 0, 0.3)))

    for patch in ax.patches:
        patch.set_alpha(0.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    approaches = combined_df['approach'].unique()
    if target is not None:
        if isinstance(target, (list, np.ndarray, pd.Series)):
            if len(target) == len(approaches):
                median_n_queries = [df['query_regions'].max() for df in dfs]
                x_min, x_max = -0.5, len(approaches) - 0.5
                median_errors = np.array(median_n_queries) * np.array(target)
                ax.plot([x_min, x_max], [median_errors[0], median_errors[-1]], color='orange', linestyle='--', linewidth=2, label='Target (Median)')
                # ax.plot([x_min, x_max], [target[0], target[-1]], color='red', linestyle='--', linewidth=2, label='Target')
            else:
                raise ValueError("Length of target values must match number of approaches.")
        else:
            median_n_queries = dfs[-1]['query_regions'].max()
            ax.axhline(target * median_n_queries, color='orange', linestyle='--', linewidth=2, label='Target')
            # ax.axhline(target, color='red', linestyle='--', linewidth=2, label='Target')


    rel_entropy_color = 'tab:red'

    # === ✅ Add relative entropy values above each box ===
    if relative_entropies is not None:
        if len(relative_entropies) != len(approaches):
            raise ValueError(
                f"Expected {len(approaches)} relative entropy values, "
                f"but got {len(relative_entropies)}."
            )

        # Create secondary y-axis
        ax2 = ax.twinx()
        ax2.set_ylabel("rel. entropy", color=rel_entropy_color)
        ax2.tick_params(axis='y', labelcolor=rel_entropy_color)
        ax2.spines['right'].set_color(rel_entropy_color)
        ax2.tick_params(axis='y', colors=rel_entropy_color)

        x_positions = list(range(len(relative_entropies)))
        ax2.plot(
            x_positions,
            relative_entropies,
            color=rel_entropy_color,
            zorder=5,
            alpha=0.8,
            linestyle='-'
        )

        # Optional: set secondary y-limits a bit higher for visibility
        ax2.set_ylim(0, max(relative_entropies) * 1.2)

        if rel_entropies_no_outliers is not None:
            ax2.plot(
                x_positions,
                rel_entropies_no_outliers,
                color=rel_entropy_color,
                zorder=5,
                alpha=0.8,
                linestyle='--',
            )
            # ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    ax.grid(True, axis='y', linestyle='--')

    if log_scale:
        ax.set_yscale('log')

    if scale_x_ticks:
        tick_fontsize = ax.get_xticklabels()[0].get_fontsize()
        smaller_fontsize = tick_fontsize * scale_x_ticks
        for tick in ax.get_xticklabels():
            tick.set_fontsize(smaller_fontsize)

    if rotation:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='center')

    if not show_legend and ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()

    if path_to_file:
        fig.savefig(path_to_file, bbox_inches='tight', pad_inches=0.05)

    return fig

def plot_violinplot(
    dfs, 
    col_y='normalized_error', 
    y_label='normalized error', 
    x_label="approach", 
    figsize=(8, 6), 
    log_scale=False, 
    palette=None, 
    rotation=False, 
    target=None,  
    path_to_file=None,
    show_legend=False
) -> plt.Figure:
    # Add 'approach' column if missing (assumes each df has a unique approach)
    for df in dfs:
        if 'approach' not in df.columns:
            raise ValueError("Each DataFrame must have an 'approach' column.")

    combined_df = pd.concat(dfs, ignore_index=True)
    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        x='approach',
        y=col_y, 
        hue='approach', 
        data=combined_df, 
        palette=palette, 
        ax=ax,
        cut=0,          # avoid extending beyond observed range
        inner='quartile', # draw box-style quartile lines inside violin
        alpha=0.8
        # linewidth=1.5
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if target is not None:
        if isinstance(target, (list, np.ndarray, pd.Series)):
            approaches = combined_df['approach'].unique()
            if len(target) == len(approaches):
                median_n_queries = [df['query_regions'].max() for df in dfs]
                x_min, x_max = -0.5, len(approaches) - 0.5
                median_errors = np.array(median_n_queries) * np.array(target)
                ax.plot([x_min, x_max], [median_errors[0], median_errors[-1]], color='red', linestyle='--', linewidth=2, label='Target (Median)')
                # ax.plot([x_min, x_max], [target[0], target[-1]], color='red', linestyle='--', linewidth=2, label='Target')
            else:
                raise ValueError("Length of target values must match number of approaches.")
        else:
            median_n_queries = dfs[-1]['query_regions'].max()
            ax.axhline(target * median_n_queries, color='red', linestyle='--', linewidth=2, label='Target')
            # ax.axhline(target, color='red', linestyle='--', linewidth=2, label='Target')

    ax.grid(True, axis='y', linestyle='--')

    if log_scale:
        ax.set_yscale('log')

    if rotation:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='center')

    if not show_legend and ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()

    if path_to_file:
        fig.savefig(path_to_file, bbox_inches='tight', pad_inches=0.05)

    return fig

    

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


def plot_estimate_distribution(df: pd.DataFrame, col="true_sums"):
    order = np.argsort(df[col].values)
    est_sorted = df["estimates"].values[order]
    true_sorted = df[col].values[order]

    # order = np.argsort(df_sums["estimates"].values)
    # est_sorted = df_sums["estimates"].values
    # true_sorted = df_sums["true_sums"].values

    def minmax(x):
        return (x - x.min()) / (x.max() - x.min())

    est_norm = minmax(est_sorted)
    true_norm = minmax(true_sorted)

    plt.figure(figsize=(8,4))
    plt.plot(est_norm, label="estimates (min-max)", marker="o")
    plt.plot(true_norm, label=f"{col} (min-max)", marker="o")
    plt.legend()
    plt.xlabel("sorted index by estimate")
    plt.ylabel("normalized value")
    plt.tight_layout()
    plt.show()