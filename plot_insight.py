#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 17:44 on 28/8/25
# Title: plot_insight.py (Single Log File Version)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse
import os


def plot_insight_comparison(log_file_path: str, human_data_path: str, output_dir: str = 'plots'):
    print(f"--- Parsing agent decisions from '{log_file_path}' ---")

    agent_log_data = []
    participant_regex = re.compile(r"--- Starting fit for participant .* \(([\w\d]+)\) ---")
    choice_regex = re.compile(r"\[DEBUG\] Trial (\d+):.*ChoseSystem='(MF|MB)'")
    current_participant_id = None

    with open(log_file_path, 'r') as f:
        for line in f:
            # Lines are interleaved, so we must check for participant ID on every line
            p_match = participant_regex.search(line)
            if p_match:
                current_participant_id = p_match.group(1)

            c_match = choice_regex.search(line)
            if c_match and current_participant_id:
                agent_log_data.append({
                    'participant_id': current_participant_id,
                    'trial_num': int(c_match.group(1)) + 1,
                    'chosen_system': c_match.group(2)
                })

    if not agent_log_data:
        print("No agent choice data found. Aborting.")
        return

    agent_df = pd.DataFrame(agent_log_data)
    agent_df['is_mf'] = (agent_df['chosen_system'] == 'MF').astype(int)
    agent_mf_proportion = agent_df.groupby('trial_num')['is_mf'].mean()
    print(f"Parsed {len(agent_df)} agent choices across {agent_df['participant_id'].nunique()} agents.")

    print(f"--- Loading human insight data from '{human_data_path}' ---")
    human_df = pd.read_csv(human_data_path)
    human_df['is_mf'] = human_df['insight']
    human_mf_proportion = human_df.groupby('trial_num')['is_mf'].mean()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot Human Data and Agent Data as before...
    ax.plot(human_mf_proportion.index, human_mf_proportion.rolling(window=10, center=True, min_periods=1).mean(),
            color='red', linewidth=2.5, label='Human Insight (Smoothed)')
    ax.fill_between(human_mf_proportion.index, human_mf_proportion.values, color='red', alpha=0.1,
                    label='Human Insight (Per Trial)')
    ax.plot(agent_mf_proportion.index, agent_mf_proportion.rolling(window=10, center=True, min_periods=1).mean(),
            color='darkblue', linewidth=2.5, label='Agent Heuristic Choice (Smoothed)')
    ax.fill_between(agent_mf_proportion.index, agent_mf_proportion.values, color='lightblue', alpha=0.3,
                    label='Agent Choice (Per Trial)')
    ax.axhline(y=0.50, color='gray', linestyle=':', linewidth=1.5, label='50% Chance Level')
    ax.axvline(x=30, color='black', linestyle='--', linewidth=1.5, label='End of Reinforcement Phase')

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

    print(f"\nSuccessfully generated and saved the comparison plot to '{save_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot agent and human insight from a single debug log and data file.")
    parser.add_argument('--log_file', type=str, required=True, help="Path to the consolidated debug log file.")
    parser.add_argument('--human_data', type=str, required=True, help="Path to the human data CSV file.")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"FATAL ERROR: Log file not found at '{args.log_file}'.")
    elif not os.path.exists(args.human_data):
        print(f"FATAL ERROR: Human data file not found at '{args.human_data}'.")
    else:
        plot_insight_comparison(args.log_file, args.human_data)