#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 17:36 on 23/8/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
import os

# --- Configuration ---
PARAMS_FILE = 'csv/control_250716_inferred_params_4D.csv'
HUMAN_DATA_FILE = 'csv/control_250716.csv'
PLOT_DIR = 'plots'

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


def run_tau_group_analysis():
    """
    Performs a full analysis comparing human participants split by their inferred tau value.
    """
    # --- 1. Load Data ---
    try:
        params_df = pd.read_csv(PARAMS_FILE)
        human_df = pd.read_csv(HUMAN_DATA_FILE)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: A required data file was not found. {e}")
        return

    print(f"Loaded {len(params_df)} inferred parameter sets.")
    print(f"Loaded {human_df.shape[0]} trials from {human_df['prolific_id'].nunique()} participants.")

    # --- 2. Create Groups based on Median Split of Tau ---
    median_tau = params_df['inferred_tau'].median()
    params_df['tau_group'] = np.where(params_df['inferred_tau'] <= median_tau, 'Low Tau (Deterministic)',
                                      'High Tau (Stochastic)')

    print("\n" + "=" * 50)
    print("Group Assignment based on Median Tau Split")
    print(f"Median Tau Value: {median_tau:.4f}")
    print(params_df['tau_group'].value_counts())
    print("=" * 50 + "\n")

    # --- 3. Compare Overall Performance (T-Test) ---
    overall_accuracy = human_df.groupby('prolific_id')['accuracy'].mean().reset_index()
    analysis_df = pd.merge(overall_accuracy, params_df[['prolific_id', 'tau_group']], on='prolific_id')

    group_low = analysis_df[analysis_df['tau_group'] == 'Low Tau (Deterministic)']['accuracy']
    group_high = analysis_df[analysis_df['tau_group'] == 'High Tau (Stochastic)']['accuracy']

    t_stat, p_value = ttest_ind(group_low, group_high)

    print("--- 1. Overall Performance Analysis ---")
    print(f"Mean Accuracy for Low Tau Group: {group_low.mean():.3f}")
    print(f"Mean Accuracy for High Tau Group: {group_high.mean():.3f}")
    print(f"Independent T-Test: t = {t_stat:.2f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("--> Result is STATISTICALLY SIGNIFICANT. The two groups have different overall accuracy.")
    else:
        print("--> No significant difference in overall accuracy between the groups.")
    print("-" * 35 + "\n")

    # --- 4. Compare Learning Trajectories (Plot) ---
    print("--- 2. Learning Trajectory Analysis ---")
    full_analysis_df = pd.merge(human_df, params_df[['prolific_id', 'tau_group']], on='prolific_id')
    full_analysis_df['block'] = (full_analysis_df['trial_num'] - 1) // 30 + 1

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=full_analysis_df[full_analysis_df['block'].isin([1, 2, 3])],
        x='block',
        y='accuracy',
        hue='tau_group',
        hue_order=['Low Tau (Deterministic)', 'High Tau (Stochastic)'],
        palette={'Low Tau (Deterministic)': 'blue', 'High Tau (Stochastic)': 'orange'},
        marker='o',
        markersize=10,
        linewidth=4,
        errorbar=('ci', 95)
    )
    plt.title('Accuracy Trajectory by Inferred Tau Group', fontweight='bold', fontsize=16)
    plt.xlabel('Block')
    plt.ylabel('Average Accuracy')
    plt.xticks([1, 2, 3], ['Block 1 (Train)', 'Block 2 (Test)', 'Block 3 (Test)'])
    plt.ylim(0, 1.05)
    plt.legend(title='Inferred Strategy Group')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plot_path = os.path.join(PLOT_DIR, 'accuracy_trajectory_by_tau_group.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved learning trajectory plot to '{plot_path}'")
    print("-" * 35 + "\n")

    # --- 5. Compare Survey Responses (Chi-Squared Test) ---
    print("--- 3. Survey Response Analysis ---")
    survey_cols = [f'feedback_{i}' for i in range(1, 9)]

    if all(col in human_df.columns for col in survey_cols):
        unique_responses = human_df.groupby('prolific_id')[survey_cols].first().reset_index()
        survey_analysis_df = pd.merge(unique_responses, params_df[['prolific_id', 'tau_group']], on='prolific_id')

        for col in survey_cols:
            if survey_analysis_df[col].dropna().empty:
                continue

            print(f"\n--- Analysis for Survey Question: {col} ---")
            contingency_table = pd.crosstab(survey_analysis_df[col], survey_analysis_df['tau_group'])
            print("Response Counts by Group:")
            print(contingency_table)

            try:
                chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                print(f"Chi-squared Test: p-value = {p_val:.4f}")
                if p_val < 0.05:
                    print("--> SIGNIFICANT association between survey response and tau group.")
            except ValueError:
                print("Could not perform Chi-squared test (likely insufficient data).")
    else:
        # Handle the case where survey columns are missing
        print("Survey columns ('feedback_1' through 'feedback_8') not found in the dataset.")
        print("Skipping survey response analysis.")

    print("-" * 35 + "\n")


if __name__ == "__main__":
    run_tau_group_analysis()
    run_tau_group_analysis()