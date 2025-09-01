#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 10:46 on 25/7/25
# Title: utils.py
# Explanation: Plotting functions

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from celluloid import Camera
import numpy as np
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import itertools
import ruptures as rpt
import warnings
import imageio

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)


def makeGif(all_histories: list, filename_prefix: str, attributes: list):
    """
    Generates a GIF of the v_relation learning process, averaged across all agents.
    """
    print(f"--- Generating averaged '{filename_prefix}' GIF for {len(all_histories)} agents ---")

    if not all_histories:
        print("Warning: No history data provided to makeGif. Aborting.")
        return

    histories_array = np.array(all_histories)
    avg_history = np.mean(histories_array, axis=0)

    filenames = []
    num_trials = len(avg_history)

    vmin = np.min(avg_history) if avg_history.size > 0 else 0
    vmax = np.max(avg_history) if avg_history.size > 0 else 1

    for i in range(num_trials):
        avg_v_relation_matrix = avg_history[i]
        matrix_to_plot = avg_v_relation_matrix.T

        fig, ax = plt.subplots(figsize=(5, 6), dpi=100)
        im = ax.imshow(matrix_to_plot, cmap='viridis', vmin=vmin, vmax=vmax)

        ax.set_xticks([])
        ax.set_yticks(np.arange(3))
        ax.set_yticklabels(['Same', 'Error', 'Diff'], fontsize=14)

        for y_index in range(matrix_to_plot.shape[0]):
            for x_index in range(matrix_to_plot.shape[1]):
                attribute_label = attributes[x_index]
                ax.text(x_index, y_index, attribute_label,
                        ha="center", va="center", color="black", fontsize=14)

        if i < 30:
            phase_label = r"$\bf{TRAIN}$"
            title_color = 'blue'
        else:
            phase_label = r"$\bf{TEST}$"
            title_color = 'red'

        title_text = f"Internal Values (trial {i}) - {phase_label}"
        ax.set_title(title_text, fontsize=16, color=title_color)

        fig.tight_layout()

        frame_filename = f"{output_dir}/_frame_{i}.png"
        filenames.append(frame_filename)
        plt.savefig(frame_filename)
        plt.close()

    gif_path = f"{output_dir}/{filename_prefix}_averaged.gif"

    with imageio.get_writer(gif_path, mode='I', duration=100, unit='ms', loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in set(filenames):
        os.remove(filename)

    print(f"Successfully created averaged GIF: {gif_path}")

def get_significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


# Calculates smoothed mean and SEM
def _get_smoothed_stats(stats_df: pd.DataFrame, window_size: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mean = stats_df['mean']
    sem = stats_df['sem']
    smoothed_mean = mean.rolling(window=window_size, center=True, min_periods=1).mean()
    smoothed_lower = (mean - sem).rolling(window=window_size, center=True, min_periods=1).mean()
    smoothed_upper = (mean + sem).rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed_mean, smoothed_lower, smoothed_upper


# Performance plots with a shaded SEM area for both agent and human accuracy.
def plot_results(agent_full_df: pd.DataFrame, attributes: List[str], human_acc_stats_df: pd.DataFrame,
                 human_conf_stats_df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Cognitive Agent Simulation Results (Averaged)', fontsize=16, y=0.99)
    smoothing_window_size = 5

    # --- Plot 1: Overall Accuracy Over Time with SEM for Agent and Human ---
    # Agent accuracy data
    agent_acc_stats = agent_full_df.groupby('trial_num')['was_correct'].agg(['mean', 'sem']).reset_index()
    agent_smoothed_mean, agent_smoothed_lower, agent_smoothed_upper = _get_smoothed_stats(agent_acc_stats,
                                                                                          smoothing_window_size)
    axes[0].plot(agent_acc_stats['trial_num'], agent_smoothed_mean, label='Agent Avg. Accuracy', color='b', linewidth=2)
    axes[0].fill_between(agent_acc_stats['trial_num'], agent_smoothed_lower, agent_smoothed_upper,
                         color='b', alpha=0.2, label='Agent Accuracy SEM')

    # Agent confidence data (Dashed Cyan)
    agent_conf_stats = agent_full_df.groupby('trial_num')['confidence'].agg(['mean', 'sem']).reset_index()
    conf_smoothed_mean, conf_smoothed_lower, conf_smoothed_upper = _get_smoothed_stats(agent_conf_stats,
                                                                                       smoothing_window_size)
    axes[0].plot(agent_conf_stats['trial_num'], conf_smoothed_mean, label='Agent Avg. Confidence', color='c',
                 linestyle='--', linewidth=2)
    axes[0].fill_between(agent_conf_stats['trial_num'], conf_smoothed_lower, conf_smoothed_upper, color='c', alpha=0.15)

    # Human accuracy data (Solid Red)
    human_acc_smoothed_mean, human_acc_smoothed_lower, human_acc_smoothed_upper = _get_smoothed_stats(
        human_acc_stats_df,
        smoothing_window_size)
    axes[0].plot(human_acc_stats_df['trial_num'], human_acc_smoothed_mean, label='Human Avg. Accuracy', color='r',
                 linestyle='-', linewidth=2)
    axes[0].fill_between(human_acc_stats_df['trial_num'], human_acc_smoothed_lower, human_acc_smoothed_upper,
                         color='r', alpha=0.2)

    # Human confidence data (Dashed Coral)
    human_conf_smoothed_mean, human_conf_smoothed_lower, human_conf_smoothed_upper = _get_smoothed_stats(
        human_conf_stats_df, smoothing_window_size)
    axes[0].plot(human_conf_stats_df['trial_num'], human_conf_smoothed_mean, label='Human Avg. Confidence',
                 color='coral', linestyle='--', linewidth=2)
    axes[0].fill_between(human_conf_stats_df['trial_num'], human_conf_smoothed_lower, human_conf_smoothed_upper,
                         color='coral', alpha=0.15)

    axes[0].axhline(y=0.1, color='gray', linestyle=':', label='Chance Level (10%)')
    axes[0].set_title('Agent vs. Human Performance & Confidence')
    axes[0].set_ylabel('Accuracy / Confidence')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[0].grid(True)

    # --- Plot 2: Accuracy per Attribute ---
    agent_averaged_df = agent_full_df.groupby('trial_num').mean().reset_index()
    att_map = {'C': 'Color', 'F': 'Filling', 'S': 'Shape', 'B': 'Background'}
    for i, att in enumerate(attributes):
        source_col_name = f'cho{att}_acc'
        smoothed_col_name = f'{source_col_name}_smoothed'

        # Apply smoothing to the mean accuracy data from the results_df
        agent_averaged_df[smoothed_col_name] = agent_averaged_df[source_col_name].rolling(
            window=smoothing_window_size, center=True, min_periods=1
        ).mean()
        axes[1].plot(agent_averaged_df['trial_num'], agent_averaged_df[smoothed_col_name],
                     label=f'Agent - {att_map[att]}')

    axes[1].axvline(x=30, color='k', linestyle='--', label='Train/Test Split')
    axes[1].set_title('Agent\'s Correct Feature Choice Accuracy (Same/Different vs. Pair)')
    axes[1].set_ylabel('Proportion Correct Choices')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True)

    # --- Plot 3: Learned strategic values ---
    for att in attributes:
        axes[2].plot(agent_averaged_df['trial_num'], agent_averaged_df[f'V_att_{att}'], label=f"V({att})")
    axes[2].set_title('Learned Strategic Priority of Attributes Over Time')
    axes[2].set_xlabel('Trial Number')
    axes[2].set_ylabel('Learned Strategic Value')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(os.path.join(output_dir, 'simulation_summary_plot_with_mcts_averaged.png'))
    plt.close(fig)
    print("\nGenerated 'simulation_summary_plot_with_mcts_averaged.png'")


# Bar plot by block with ANOVA, Tukey HSD tests, and significance annotations
def _plot_accuracy_bar_chart(block_data: pd.DataFrame, id_col: str, value_col: str, title: str, filename: str,
                             data_name: str):
    block_data = block_data[block_data['block'].isin([1, 2, 3])]
    block_means_per_id = block_data.groupby([id_col, 'block'])[value_col].mean().reset_index()

    if block_means_per_id.empty:
        print(f"Warning: No data found for blocks 1, 2, or 3 in '{filename}'. Skipping plot.")
        return

    pivoted_data = block_means_per_id.pivot(index=id_col, columns='block', values=value_col)
    means = pivoted_data.mean()
    sems = pivoted_data.sem()

    fig, ax = plt.subplots(figsize=(8, 7))
    bar_colors = ['white', 'lightgray', 'lightgray']
    ax.bar(means.index, means, yerr=sems, capsize=5, color=bar_colors, edgecolor='black', alpha=0.8)

    for i in range(len(pivoted_data)):
        ax.plot(pivoted_data.columns, pivoted_data.iloc[i], marker='o', color='gray',
                linestyle='-', linewidth=0.5, alpha=0.4, markersize=4)

    if block_data['block'].nunique() > 1:
        print(f"\n--- [{data_name}] Block Accuracy Stats ---")
        aov = AnovaRM(data=block_means_per_id, depvar=value_col, subject=id_col, within=['block'])
        res = aov.fit()
        print("Repeated Measures ANOVA results:")
        print(res.summary())

        tukey_results = pairwise_tukeyhsd(endog=block_means_per_id[value_col], groups=block_means_per_id['block'],
                                          alpha=0.05)
        print("\nTukey HSD post-hoc tests:")
        print(tukey_results)

        y_max = pivoted_data.max().max()
        bracket_y = y_max + 0.1
        bracket_height = 0.01
        star_y_offset = 0.01
        bracket_spacing = 0.07

        tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:],
                                columns=tukey_results._results_table.data[0])

        for index, row in tukey_df.iterrows():
            star_string = get_significance_stars(row['p-adj'])
            if star_string:
                group1, group2 = row['group1'], row['group2']
                x1, x2 = group1, group2
                ax.plot([x1, x1, x2, x2],
                        [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
                        lw=1.5, c='k')
                ax.text((x1 + x2) * 0.5, bracket_y + star_y_offset, star_string, ha='center', va='bottom', fontsize=16)
                bracket_y += bracket_spacing

        ax.set_ylim(0, bracket_y + 0.05)
    else:
        ax.set_ylim(0, 1.05)

    ax.set_title(title)
    ax.set_ylabel('Accuracy')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Block 1', 'Block 2', 'Block 3'])

    train_patch = plt.Rectangle((0, 0), 1, 1, fc="white", ec='black')
    test_patch = plt.Rectangle((0, 0), 1, 1, fc="lightgray", ec='black')
    ax.legend([train_patch, test_patch], ["Train", "Test"], loc='best')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"\nGenerated '{filename}'")


# Generates a bar plot of accuracy across blocks (agents)
def plot_agent_bar_accuracy(block_data: pd.DataFrame):
    _plot_accuracy_bar_chart(
        block_data,
        id_col='sim_run',
        value_col='was_correct',
        title='[Agent] Accuracy across blocks',
        filename=os.path.join(output_dir, 'agent_accuracy_bar_plot.png'),
        data_name='AGENT'
    )


# Generates a bar plot of accuracy across blocks (humans)
def plot_human_bar_accuracy(block_data: pd.DataFrame):
    _plot_accuracy_bar_chart(
        block_data,
        id_col='prolific_id',
        value_col='accuracy',
        title='[Human] Accuracy across blocks',
        filename=os.path.join(output_dir, 'human_accuracy_bar_plot.png'),
        data_name='HUMAN'
    )


# Generates a bar plot of the agent's feature choice accuracy, binned every 10 trials.
def plot_feature_choice_accuracy(df: pd.DataFrame, attributes: List[str]):
    # Define 9 blocks of 10 trials each
    df['ten_trial_block'] = (df['trial_num'] - 1) // 10 + 1

    # Calculate the mean accuracy for each feature in each block
    acc_cols = [f'cho{att}_acc' for att in attributes]
    block_means = df.groupby('ten_trial_block')[acc_cols].mean()

    colors = [
        [0, 0.4470, 0.7410],  # Blue (Color)
        [0.8500, 0.3250, 0.0980],  # Orange (Filling)
        [0.9290, 0.6940, 0.1250],  # Yellow (Shape)
        [0.4940, 0.1840, 0.5560]  # Purple (Background)
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    block_means.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', color=colors)
    att_map = {'C': 'Color', 'F': 'Filling', 'S': 'Shape', 'B': 'Background'}

    ax.set_title('[Agent] Average Feature Choice Accuracy Every 10 Trials')
    ax.set_xlabel('Block Number (Averaged every 10 trials)')
    ax.set_ylabel('Average Accuracy (1=Same/Diff, 0=Pair)')
    ax.set_ylim(0, 1)
    ax.legend([f'acc({att_map[att]})' for att in attributes])
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agent_feature_choice_accuracy.png'))
    plt.close(fig)
    print("\nGenerated 'agent_feature_choice_accuracy.png'")


# Calculates and plots the difference between the max and min feature choice accuracy for each 10-trial block.
def plot_accuracy_difference(df: pd.DataFrame, attributes: List[str]):
    # Define 9 blocks of 10 trials each
    df['ten_trial_block'] = (df['trial_num'] - 1) // 10 + 1

    # Calculate average feature choice accuracy for each block
    acc_cols = [f'cho{att}_acc' for att in attributes]
    block_accuracies = df.groupby('ten_trial_block')[acc_cols].mean()

    # Calculate the difference (max - min) for each block
    max_acc = block_accuracies.max(axis=1)
    min_acc = block_accuracies.min(axis=1)
    difference = max_acc - min_acc

    # Calculate the standard deviation across the four feature accuracies for each block
    std_dev = block_accuracies.std(axis=1)

    # Define bounds for the shaded area
    upper_bound = difference + std_dev
    lower_bound = difference - std_dev
    lower_bound[lower_bound < 0] = 0  # Difference cannot be negative

    # Find the minimum difference point
    min_diff_value = difference.min()
    min_diff_block = difference.idxmin()

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the shaded area for standard deviation
    ax.fill_between(block_accuracies.index, lower_bound, upper_bound,
                    color='gray', alpha=0.15, label='Std. Dev. of Blocks')

    # Plot the main difference line
    ax.plot(block_accuracies.index, difference, marker='o', linestyle='-',
            color='blue', label='Difference (Max - Min)', linewidth=2)

    # Plot and annotate the minimum difference point
    ax.plot(min_diff_block, min_diff_value, marker='o', markersize=12,
            markerfacecolor='red', markeredgecolor='white', label='Min Difference')
    ax.text(min_diff_block, min_diff_value + 0.01, f'Min Diff\nBlock {min_diff_block} ({min_diff_value:.4f})',
            ha='center', va='bottom', color='red', fontweight='bold')

    ax.set_title('[Agent] Difference of Average Accuracy by Block')
    ax.set_xlabel('Block Number')
    ax.set_ylabel('Difference (Max - Min Accuracy)')
    ax.set_xticks(range(1, 10))
    ax.set_ylim(0, max(upper_bound.max() * 1.1, 0.1))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agent_accuracy_difference_plot.png'))
    plt.close(fig)
    print("\nGenerated 'agent_accuracy_difference_plot.png'")


def plot_grouped_boxplot(df: pd.DataFrame, attributes: List[str]):
    print("\n--- Generating Grouped Box Plot ---")

    # Prepare data
    acc_cols = [f'cho{att}_acc' for att in attributes]
    agent_block_means = df.groupby(['sim_run', 'block'])[acc_cols].mean().reset_index()
    long_df = agent_block_means.melt(id_vars=['sim_run', 'block'], value_vars=acc_cols,
                                     var_name='Dimension', value_name='Accuracy')
    long_df['Dimension'] = long_df['Dimension'].str.replace('cho|_acc', '', regex=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 9))
    sns.boxplot(x='block', y='Accuracy', hue='Dimension', data=long_df, ax=ax, hue_order=attributes)

    # Only run statistical tests if there is more than one simulation run
    if df['sim_run'].nunique() > 1:
        # Statistical analysis
        all_p_values = {}
        pairs = list(itertools.combinations(attributes, 2))
        for block_num in [1, 2, 3]:
            block_data = agent_block_means[agent_block_means['block'] == block_num]
            p_vals_raw = [ttest_rel(block_data[f'cho{p[0]}_acc'], block_data[f'cho{p[1]}_acc']).pvalue for p in pairs]
            reject, p_vals_corrected, _, _ = multipletests(p_vals_raw, alpha=0.05, method='bonferroni')
            all_p_values[block_num] = {pair: p for pair, p in zip(pairs, p_vals_corrected)}

        print("Corrected p-values for within-block comparisons:")
        print(all_p_values)

        y_max = long_df['Accuracy'].max()
        # Start brackets slightly above the highest data point
        bracket_y_start = y_max + 0.05
        # Keep track of the highest y-position needed across ALL blocks
        plot_y_limit = bracket_y_start

        for block_num in [1, 2, 3]:
            # Reset the starting y-position for each new block's set of annotations
            y_pos = bracket_y_start

            # Sort pairs to ensure a consistent drawing order
            sorted_pairs = sorted(all_p_values[block_num].items(),
                                  key=lambda item: (attributes.index(item[0][0]), attributes.index(item[0][1])))

            for pair, p_val in sorted_pairs:
                stars = get_significance_stars(p_val)
                if stars:
                    att1, att2 = pair
                    # Calculate x-positions for the bracket ends
                    x1 = (block_num - 1) + (attributes.index(att1) - 1.5) / 4
                    x2 = (block_num - 1) + (attributes.index(att2) - 1.5) / 4

                    # Draw the bracket and text
                    ax.plot([x1, x1, x2, x2], [y_pos, y_pos + 0.01, y_pos + 0.01, y_pos], lw=1.5, c='k')
                    ax.text((x1 + x2) * 0.5, y_pos + 0.015, stars, ha='center', va='bottom', fontsize=14)

                    # Increment y_pos for the next bracket
                    y_pos += 0.08

                    # Update the overall plot limit if the current block's annotations went higher
            plot_y_limit = max(plot_y_limit, y_pos)

        # Set the final plot limit based on the highest annotation drawn
        plt.ylim(top=plot_y_limit)

    ax.set_title('Agent Feature Choice Accuracy by Block across Dimensions')
    ax.set_xlabel('Block')
    ax.set_ylabel('Mean Accuracy (1=Same/Diff, 0=Pair)')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Block 1', 'Block 2', 'Block 3'])
    ax.legend(title='Attribute')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agent_grouped_boxplot.png'))
    plt.close(fig)
    print("\nGenerated 'agent_grouped_boxplot.png'")


def plot_accuracy_distributions(human_df: pd.DataFrame, agent_df: pd.DataFrame):
    print("Generating comparison plot for human and agent accuracy distributions...")

    # --- Prepare Data ---
    # 1. Calculate final accuracy for each human participant
    human_final_accuracies = human_df.groupby('prolific_id')['accuracy'].mean()

    # 2. Calculate final accuracy for each agent
    agent_final_accuracies = agent_df.groupby('sim_run')['was_correct'].mean()

    # --- Create Plots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Subplot 1: Human Accuracies
    sns.histplot(human_final_accuracies, bins=15, kde=True, ax=ax1, color='skyblue')
    human_mean_acc = human_final_accuracies.mean()
    ax1.axvline(human_mean_acc, color='red', linestyle='--', label=f'Mean Acc: {human_mean_acc:.2f}')
    ax1.set_title('Distribution of Final Accuracies Across Human Participants', fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Number of Humans')
    ax1.legend()

    # Subplot 2: Agent Accuracies
    sns.histplot(agent_final_accuracies, bins=15, kde=True, ax=ax2, color='lightgreen')
    agent_mean_acc = agent_final_accuracies.mean()
    ax2.axvline(agent_mean_acc, color='red', linestyle='--', label=f'Mean Acc: {agent_mean_acc:.2f}')
    ax2.set_title('Distribution of Final Accuracies Across 57 Agents', fontweight='bold')
    ax2.set_xlabel('Final Average Accuracy')
    ax2.set_ylabel('Number of Agents')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_distributions_comparison.png'))
    print("Saved comparison plot to 'accuracy_distributions_comparison.png'")


def plot_accuracy_distributions_standardized(human_df: pd.DataFrame, agent_df: pd.DataFrame):
    """
    Plots the distribution of final accuracies for humans and agents using the same, standardized bins.
    """
    print("\n--- Generating standardized accuracy distribution plot ---")

    human_final_acc = human_df.groupby('prolific_id')['accuracy'].mean()
    agent_final_acc = agent_df.groupby('prolific_id')['accuracy'].mean()

    num_humans = human_df['prolific_id'].nunique()
    num_agents = agent_df['prolific_id'].nunique()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)

    # Define a common set of bins for both plots for a true comparison
    min_val = min(human_final_acc.min(), agent_final_acc.min())
    max_val = max(human_final_acc.max(), agent_final_acc.max())
    bins = np.linspace(min_val, max_val + 1e-9, 15)

    # Human Plot
    sns.histplot(human_final_acc, ax=ax1, color='skyblue', kde=True, bins=bins)
    ax1.axvline(human_final_acc.mean(), color='red', linestyle='--', label=f'Mean Acc: {human_final_acc.mean():.2f}')
    ax1.set_title(f'Distribution of Final Accuracies Across {num_humans} Human Participants', fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Number of Humans')
    ax1.legend()

    # Agent Plot
    sns.histplot(agent_final_acc, ax=ax2, color='lightgreen', kde=True, bins=bins)
    ax2.axvline(agent_final_acc.mean(), color='red', linestyle='--', label=f'Mean Acc: {agent_final_acc.mean():.2f}')
    ax2.set_title(f'Distribution of Final Accuracies Across {num_agents} Agents', fontweight='bold')
    ax2.set_xlabel('Final Average Accuracy')
    ax2.set_ylabel('Number of Agents')
    ax2.legend()

    save_path = os.path.join(output_dir, 'accuracy_distributions_standardized.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved standardized accuracy distribution plot to '{save_path}'")
def plot_avg_simulation_dist(agent_df: pd.DataFrame):
    print("Generating distribution plot for average simulations per agent...")

    # Calculate the average number of simulations for each agent
    avg_sims_per_agent = agent_df.groupby('sim_run')['num_mcts_simulations'].mean()

    # --- Create Plot ---
    plt.figure(figsize=(12, 6))
    sns.histplot(avg_sims_per_agent, bins=15, kde=True)

    mean_val = avg_sims_per_agent.mean()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean of Averages: {mean_val:.2f}')

    plt.title('Distribution of Average MCTS Simulations per Agent', fontweight='bold')
    plt.xlabel('Average Total Simulations per Trial')
    plt.ylabel('Number of Agents')
    plt.legend()

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the figure
    save_path = os.path.join(output_dir, 'avg_mcts_sims_per_agent_dist.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to '{save_path}'")


def plot_rt_over_blocks(human_df: pd.DataFrame):
    print("Generating plot for reaction times over blocks...")

    # --- 1. Data Preparation ---
    if not all(col in human_df.columns for col in ['ThreeRespRT', 'response_time_problem_solving_keyboard']):
        print("Missing required RT columns. Skipping plot.")
        return

    trials_per_block = 10
    total_trials = human_df['trial_num'].max()
    num_blocks = total_trials // trials_per_block

    df = human_df.copy()
    df['block'] = (df['trial_num'] - 1) // trials_per_block + 1

    # --- 2. Calculate Block-Averaged Stats ---
    rt_stats = df.groupby('block').agg(
        mean_3resp=('ThreeRespRT', 'mean'),
        sem_3resp=('ThreeRespRT', 'sem'),
        mean_ps=('response_time_problem_solving_keyboard', 'mean'),
        sem_ps=('response_time_problem_solving_keyboard', 'sem')
    ).reset_index()

    # Calculate the difference in mean RTs
    rt_stats['diff'] = rt_stats['mean_3resp'] - rt_stats['mean_ps']

    # --- 3. Changepoint Analysis ---
    def find_changepoint(data_series):
        try:
            points = data_series.dropna().to_numpy()
            if len(points) > 1:
                algo = rpt.Pelt(model="rbf").fit(points)
                result = algo.predict(n_bkps=1)
                if result and result[0] < len(points):
                    return result[0]
        except ImportError:
            warnings.warn("'ruptures' library not installed. Skipping changepoint analysis.")
        except Exception as e:
            print(f"Could not perform changepoint analysis: {e}")
        return None

    changepoint_3resp = find_changepoint(rt_stats['mean_3resp'])
    changepoint_diff = find_changepoint(rt_stats['diff'])
    print(f"Changepoint for 3-Resp RT is at BLOCK: {changepoint_3resp}")
    print(f"Changepoint for RT Difference is at BLOCK: {changepoint_diff}")

    # --- 4. Plotting ---
    # Create a figure with two subplots (2 rows, 1 column), sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # --- PLOT 1: REACTION TIMES ---
    line_color_3resp, shade_color_3resp = ((0.9, 0.4, 0.1), (0.95, 0.7, 0.5))
    line_color_ps, shade_color_ps = ((0.2, 0.6, 0.2), (0.6, 0.9, 0.6))

    # Plot ThreeRespRT
    ax1.plot(rt_stats['block'], rt_stats['mean_3resp'], color=line_color_3resp, marker='s',
             label='Mean Final Decision RT')
    ax1.fill_between(rt_stats['block'], rt_stats['mean_3resp'] - rt_stats['sem_3resp'],
                     rt_stats['mean_3resp'] + rt_stats['sem_3resp'],
                     color=shade_color_3resp, alpha=0.3, label='Final Decision RT ± SEM')

    # Plot Problem-Solving RT
    ax1.plot(rt_stats['block'], rt_stats['mean_ps'], color=line_color_ps, marker='o', label='Mean First Choice RT')
    ax1.fill_between(rt_stats['block'], rt_stats['mean_ps'] - rt_stats['sem_ps'],
                     rt_stats['mean_ps'] + rt_stats['sem_ps'],
                     color=shade_color_ps, alpha=0.3, label='First Choice RT ± SEM')

    ax1.axvline(3, color='gray', linestyle='--', label='Reward Removed')
    if changepoint_3resp:
        ax1.axvline(changepoint_3resp + 0.5, color=line_color_3resp, linestyle='--',
                    label=f'Changepoint at Block {changepoint_3resp}')

    ax1.set_title('Average Human Deliberation Time Per 10-Trial Block')
    ax1.set_ylabel('Reaction Time (ms)')
    ax1.legend(loc='best')

    # --- PLOT 2: DIFFERENCE IN REACTION TIMES ---
    ax2.plot(rt_stats['block'], rt_stats['diff'], color='black', marker='s', label='Difference (Final - First)')
    if changepoint_diff:
        ax2.axvline(changepoint_diff + 0.5, color='magenta', linestyle='--',
                    label=f'Difference Changepoint at Block {changepoint_diff}')

    ax2.axhline(0, color='gray', linestyle='-')  # Add a zero line
    ax2.set_title('Difference in Block-Averaged RTs (Final Decision RT minus First Choice RT)')
    ax2.set_xlabel('Block Number')
    ax2.set_ylabel('Difference in RT (ms)')
    ax2.legend(loc='best')

    # Final figure adjustments
    plt.xticks(range(1, int(num_blocks) + 1))
    plt.xlim(0.5, num_blocks + 0.5)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'human_deliberation_time_full.png')
    plt.savefig(save_path)
    print(f"Saved two-panel plot to '{save_path}'")


def plot_agent_deliberation_over_blocks(agent_df: pd.DataFrame):
    print("Generating plot for agent deliberation over blocks...")

    # --- 1. Data Preparation ---
    # Convert trial number to block number
    agent_df['block'] = (agent_df['trial_num'] - 1) // 10 + 1

    # --- 2. Calculate Block-Averaged Stats ---
    deliberation_stats = agent_df.groupby('block')['num_mcts_simulations'].agg(['mean', 'sem']).reset_index()

    # --- 3. Plotting ---
    plt.figure(figsize=(14, 7))
    # sns.set_theme(style="whitegrid")

    line_color = (0.2, 0.4, 0.8)  # Blue
    shade_color = (0.6, 0.8, 1.0)

    # Calculate upper and lower bounds for the shaded SEM area
    deliberation_stats['upper'] = deliberation_stats['mean'] + deliberation_stats['sem']
    deliberation_stats['lower'] = deliberation_stats['mean'] - deliberation_stats['sem']
    deliberation_stats['lower'] = deliberation_stats['lower'].clip(lower=0)

    # Plot the mean deliberation line
    plt.plot(deliberation_stats['block'], deliberation_stats['mean'], color=line_color, linewidth=2,
             marker='o', markersize=6, label='Mean Agent Deliberation')

    # Plot the shaded SEM area
    plt.fill_between(deliberation_stats['block'], deliberation_stats['lower'], deliberation_stats['upper'],
                     color=shade_color, alpha=0.3, label='Mean ± SEM')

    # Add a vertical line for the train/test split
    train_test_split_block = 3  # Reward is removed after block 1 (30 trials)
    plt.axvline(train_test_split_block, color='gray', linestyle='--', linewidth=1.5, label='Reward Removed')

    # Final plot styling
    plt.title('Average Agent Deliberation (MCTS Simulations) Per Block')
    plt.xlabel('Block Number (30 trials each)')
    plt.ylabel('Average Number of MCTS Simulations')
    plt.xticks(range(1, int(deliberation_stats['block'].max()) + 1))
    plt.xlim(0.5, deliberation_stats['block'].max() + 0.5)
    plt.legend(loc='best')

    save_path = os.path.join(output_dir, 'agent_deliberation_over_blocks.png')
    plt.savefig(save_path)
    print(f"Saved agent deliberation plot to '{save_path}'")


def plot_insight_comparison(human_df: pd.DataFrame, agent_df: pd.DataFrame, output_dir: str = 'plots'):
    """
    Plots a direct comparison of heuristic (MF) / insight-based choices over trials
    using the final human and agent dataframes.

    Args:
        human_df: DataFrame with the human experimental data.
        agent_df: DataFrame with the simulated agent data.
        output_dir: Directory to save the plot.
    """
    print("--- Generating human vs. agent insight comparison plot ---")

    # --- 1. Process Human Insight Data ---
    if 'insight' not in human_df.columns:
        print("WARNING: 'insight' column not found in human data file. Skipping insight plot.")
        return
    human_df['is_mf'] = human_df['insight']
    human_mf_proportion = human_df.groupby('trial_num')['is_mf'].mean()

    # --- 2. Process Agent Insight Data ---
    if 'chosen_system' not in agent_df.columns:
        print("WARNING: 'chosen_system' column not found in agent data. Skipping insight plot.")
        return
    agent_df['is_mf'] = (agent_df['chosen_system'] == 'MF').astype(int)
    agent_mf_proportion = agent_df.groupby('trial_num')['is_mf'].mean()

    # --- 3. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot Human Data
    ax.plot(human_mf_proportion.index, human_mf_proportion.rolling(window=10, center=True, min_periods=1).mean(),
            color='red', linewidth=2.5, label='Human Insight (Smoothed)')
    ax.fill_between(human_mf_proportion.index, human_mf_proportion.values,
                    color='red', alpha=0.1, label='Human Insight (Per Trial)')

    # Plot Agent Data
    ax.plot(agent_mf_proportion.index, agent_mf_proportion.rolling(window=10, center=True, min_periods=1).mean(),
            color='darkblue', linewidth=2.5, label='Agent Heuristic Choice (Smoothed)')
    ax.fill_between(agent_mf_proportion.index, agent_mf_proportion.values,
                    color='lightblue', alpha=0.3, label='Agent Choice (Per Trial)')

    # --- Add Reference Lines ---
    ax.axhline(y=0.50, color='gray', linestyle=':', linewidth=1.5, label='50% Chance Level')
    ax.axvline(x=30, color='black', linestyle='--', linewidth=1.5, label='End of Reinforcement Phase')

    # --- Formatting ---
    ax.set_title("Human vs. Agent Propensity to Use Heuristic/Insightful Strategy", fontsize=18, fontweight='bold')
    ax.set_xlabel("Trial Number", fontsize=14)
    ax.set_ylabel("Proportion of Heuristic (MF) Choices", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 91)
    ax.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'human_vs_agent_insight_plot.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Successfully generated and saved the insight comparison plot to '{save_path}'")


def print_results_summary(human_df: pd.DataFrame, agent_df: pd.DataFrame):
    """
    Calculates and prints a text-based summary comparing human and agent performance.

    Args:
        human_df: DataFrame with the human experimental data.
        agent_df: DataFrame with the simulated agent data.
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "AGENT VS. HUMAN PERFORMANCE SUMMARY")
    print("=" * 80)

    # --- 1. Overall Average Accuracy ---
    human_avg_acc = human_df['accuracy'].mean()
    agent_avg_acc = agent_df['accuracy'].mean()
    print("\n--- 1. Overall Average Accuracy ---")
    print(f"Human: {human_avg_acc:.3f}")
    print(f"Agent: {agent_avg_acc:.3f}")

    # --- 2. Insight / Heuristic Choice Comparison ---
    # For humans, the 'insight' column is used (1=insight/heuristic, 0=analysis)
    human_df['is_mf'] = human_df['insight']
    human_insight_ratio = human_df['is_mf'].mean()

    # For agents, the 'chosen_system' column is used
    agent_df['is_mf'] = (agent_df['chosen_system'] == 'MF').astype(int)
    agent_insight_ratio = agent_df['is_mf'].mean()
    print("\n--- 2. Heuristic / Insightful Choice Ratio ---")
    print(f"Human: {human_insight_ratio:.2%}")
    print(f"Agent: {agent_insight_ratio:.2%}")

    # --- 3. Accuracy Distribution Statistics ---
    # This gives a detailed statistical breakdown of performance across participants
    human_participant_acc = human_df.groupby('prolific_id')['accuracy'].mean()
    agent_participant_acc = agent_df.groupby('prolific_id')['accuracy'].mean()

    print("\n--- 3. Accuracy Distribution Statistics (per participant) ---")
    print("\nHumans:")
    print(human_participant_acc.describe().round(3))
    print("\nAgents:")
    print(agent_participant_acc.describe().round(3))
    print("\n" + "=" * 80)