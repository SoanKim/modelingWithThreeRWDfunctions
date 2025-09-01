#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 21:45 on 29/8/25
# Title: run_deep_analysis.py
# Explanation: Loads human data, agent simulation data, and inferred agent
#              parameters to generate a comprehensive set of final analysis plots.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def get_participant_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the 20-feature "Cognitive Bias" profile for each participant.
    """
    if df.empty: return pd.DataFrame()
    attributes = ['color', 'fill', 'shape', 'back']
    pids = []
    bias_profiles = []
    required_cols = [f'{ct}_{attr}' for ct in ['modSum', 'choice'] for attr in attributes]
    if not all(col in df.columns for col in required_cols):
        print("Warning: Missing detailed choice columns in DataFrame. Cannot generate profiles.")
        return None

    for pid, group in df.groupby('prolific_id'):
        pids.append(pid)
        profile = {}
        for attr in attributes:
            modsum_col, choice_col = f'modSum_{attr}', f'choice_{attr}'
            profile[f'{attr}_Correct_Same'] = ((group[modsum_col] == 1) & (group[choice_col] == 1)).sum()
            profile[f'{attr}_Correct_Diff'] = ((group[modsum_col] == 3) & (group[choice_col] == 3)).sum()
            profile[f'{attr}_Same_for_Diff_Err'] = ((group[modsum_col] == 3) & (group[choice_col] == 1)).sum()
            profile[f'{attr}_Diff_for_Same_Err'] = ((group[modsum_col] == 1) & (group[choice_col] == 3)).sum()
            profile[f'{attr}_Invalid_Pattern_Err'] = (group[choice_col] == 2).sum()
        bias_profiles.append(profile)
    profiles_df = pd.DataFrame(bias_profiles, index=pids)
    profiles_df.index.name = 'prolific_id'
    return profiles_df


def find_optimal_k_bic(data: np.ndarray, max_k: int = 5) -> int:
    """Finds the optimal number of clusters using BIC."""
    bics = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=123)
        gmm.fit(data)
        bics.append(gmm.bic(data))
    return np.argmin(bics) + 1


def run_full_analysis(human_df, agent_df, params_df, output_dir='plots'):
    """
    Runs the complete suite of final analyses and generates all plots.
    """
    print("\n" + "#" * 60)
    print("###  Running Full Deep Analysis Suite  ###")
    print("#" * 60)

    # --- 1. PCA and Cluster Analysis ---
    print("\n--- Performing PCA and Cluster Analysis ---")
    human_profiles = get_participant_profiles(human_df)
    agent_profiles = get_participant_profiles(agent_df)

    if human_profiles is not None and agent_profiles is not None:
        # --- Human PCA ---
        human_proportions = human_profiles.div(human_profiles.sum(axis=1), axis=0).fillna(0)
        human_scaled = StandardScaler().fit_transform(human_proportions)
        pca_human = PCA(n_components=1)
        human_pc1 = pca_human.fit_transform(human_scaled)
        optimal_k_human = find_optimal_k_bic(human_pc1)
        print(f"Optimal k for Humans: {optimal_k_human}")

        # --- Agent PCA ---
        agent_proportions = agent_profiles.div(agent_profiles.sum(axis=1), axis=0).fillna(0)
        agent_scaled = StandardScaler().fit_transform(agent_proportions)
        pca_agent = PCA(n_components=1)
        agent_pc1 = pca_agent.fit_transform(agent_scaled)
        optimal_k_agent = find_optimal_k_bic(agent_pc1)
        print(f"Optimal k for Agents: {optimal_k_agent}")

        # --- Generate Agent Cluster Pairgrid ---
        if optimal_k_agent > 1:
            gmm = GaussianMixture(n_components=optimal_k_agent, random_state=123)
            agent_clusters = gmm.fit_predict(agent_pc1)
            params_df['cluster'] = agent_clusters

            param_cols = [col for col in params_df.columns if col not in ['prolific_id', 'log_likelihood', 'cluster']]

            g = sns.pairplot(params_df, vars=param_cols, hue='cluster', palette='viridis')
            g.fig.suptitle('Agent Parameter Space by Behavioral Cluster', y=1.02, fontweight='bold')
            plot_path = os.path.join(output_dir, 'agent_cluster_pairgrid.png')
            g.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Saved agent cluster pairgrid to '{plot_path}'")

    # --- 2. Metacognitive Profile Analysis ---
    print("\n--- Generating Agent Metacognitive Profile ---")
    if 'confidence' in agent_df.columns:
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=agent_df, x='confidence', y='accuracy', errorbar='se', marker='o')
        plt.title('Agent Metacognitive Profile', fontweight='bold')
        plt.xlabel('Agent Confidence')
        plt.ylabel('Proportion Correct')
        plt.grid(True, linestyle='--')
        plot_path = os.path.join(output_dir, 'agent_metacognitive_profile.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved agent metacognitive profile to '{plot_path}'")

    # --- 3. Time-Series Analysis ---
    print("\n--- Generating Time-Series Comparison Plot ---")
    if 'confidence' in human_df.columns and 'confidence' in agent_df.columns:
        # Normalize human confidence (assuming 1-4 scale)
        human_df['confidence'] = (human_df['confidence'] - 1) / 3

        human_acc = human_df.groupby('trial_num')['accuracy'].mean()
        agent_acc = agent_df.groupby('trial_num')['accuracy'].mean()
        human_conf = human_df.groupby('trial_num')['confidence'].mean()
        agent_conf = agent_df.groupby('trial_num')['confidence'].mean()

        fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        # Accuracy Plot
        ax[0].plot(human_acc.index, human_acc.rolling(10, center=True).mean(), label='Human Accuracy', color='red')
        ax[0].plot(agent_acc.index, agent_acc.rolling(10, center=True).mean(), label='Agent Accuracy', color='blue')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()
        ax[0].grid(True, linestyle='--')
        # Confidence Plot
        ax[1].plot(human_conf.index, human_conf.rolling(10, center=True).mean(), label='Human Confidence',
                   color='darkred')
        ax[1].plot(agent_conf.index, agent_conf.rolling(10, center=True).mean(), label='Agent Confidence',
                   color='darkblue')
        ax[1].set_ylabel('Confidence')
        ax[1].set_xlabel('Trial Number')
        ax[1].legend()
        ax[1].grid(True, linestyle='--')

        fig.suptitle('Time-Series of Accuracy and Confidence', fontweight='bold', fontsize=16)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plot_path = os.path.join(output_dir, 'timeseries_accuracy_confidence_comparison.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved time-series comparison plot to '{plot_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a deep analysis of human vs. agent data.")
    parser.add_argument('--human_data', type=str, required=True, help="Path to the raw human data CSV.")
    parser.add_argument('--agent_data', type=str, required=True, help="Path to the agent simulation results CSV.")
    parser.add_argument('--params_data', type=str, required=True, help="Path to the inferred agent parameters CSV.")
    args = parser.parse_args()

    try:
        human_df = pd.read_csv(args.human_data)
        agent_df = pd.read_csv(args.agent_data)
        params_df = pd.read_csv(args.params_data)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: A required data file was not found. {e}")
        exit()

    run_full_analysis(human_df, agent_df, params_df)

    print("\n\n--- Deep analysis complete. ---")