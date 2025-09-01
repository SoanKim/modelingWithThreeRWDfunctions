#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 11:15 on 28/7/25
# Title: humanAnalysis.py
# Explanation: Analyzes trial-by-trial data to infer learning parameters and optimal strategy clusters for each participant.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.linear_model import LogisticRegression
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo


warnings.filterwarnings("ignore", category=ConvergenceWarning)

csv_dir = 'csv'
plot_dir = 'plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


def _prepare_and_validate_data(human_df: pd.DataFrame):
    """Prepares and validates the dataframe for analysis."""
    attributes = ['color', 'fill', 'shape', 'back']
    modsum_cols = [f'modSum_{attr}' for attr in attributes]
    choice_cols = [f'choice_{attr}' for attr in attributes]
    if not all(col in human_df.columns for col in modsum_cols + choice_cols):
        return None, None
    return human_df.copy(), attributes


def get_participant_profiles(human_df: pd.DataFrame):
    """Calculates the 20-feature behavioral profile for each participant."""
    df, attributes = _prepare_and_validate_data(human_df)
    if df is None: return None

    pids = []
    bias_profiles = []
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

    profiles_df = pd.DataFrame(bias_profiles, index=pids, dtype=int)
    profiles_df.index.name = 'prolific_id'
    return profiles_df


def get_strategy_clusters(profiles_df: pd.DataFrame, n_clusters: int):
    """
    Performs clustering on participant profiles and returns cluster assignments with sorted labels.
    Returns both the cluster assignments DataFrame and a sorted list of group names for plotting.
    This version is robust to non-numeric columns like 'prolific_id'.
    """
    # Ensure prolific_id is the index to exclude it from numeric calculations
    if 'prolific_id' in profiles_df.columns:
        profiles_df = profiles_df.set_index('prolific_id')

    numeric_profiles_df = profiles_df.select_dtypes(include=np.number)
    row_sums = numeric_profiles_df.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    profiles_proportions = numeric_profiles_df.div(row_sums, axis=0)

    scaler = StandardScaler()
    profiles_scaled = scaler.fit_transform(profiles_proportions.fillna(0))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    profiles_df['Cluster_ID'] = kmeans.fit_predict(profiles_scaled)

    # Use the 'total_knowledge' or 'total_correct' column for sorting cluster labels by performance
    sort_col = 'total_knowledge' if 'total_knowledge' in profiles_df.columns else 'total_correct'
    if sort_col not in profiles_df.columns:
        # Fallback if sorting column is missing, create it
        correct_cols = [col for col in profiles_df.columns if 'Correct' in col]
        profiles_df[sort_col] = profiles_df[correct_cols].sum(axis=1)

    cluster_summary = profiles_df.groupby('Cluster_ID')[sort_col].mean().sort_values(ascending=False)

    all_labels = ['High', 'Medium', 'Low', 'Very Low', 'Lowest']
    if n_clusters == 2:
        labels_to_use = [all_labels[0], all_labels[2]]  # High, Low
    else:
        labels_to_use = all_labels[:n_clusters]

    if len(labels_to_use) < n_clusters:
        labels_to_use.extend([f'Group {i + 1}' for i in range(len(labels_to_use), n_clusters)])

    label_mapping = {cluster_id: labels_to_use[i] for i, cluster_id in enumerate(cluster_summary.index)}
    profiles_df['Performance Group'] = profiles_df['Cluster_ID'].map(label_mapping)

    sorted_group_names = [label_mapping[idx] for idx in cluster_summary.index]

    # Return the prolific_id back as a column
    return profiles_df.reset_index()[['prolific_id', 'Performance Group']], sorted_group_names

def find_optimal_k_programmatically(profiles_scaled: np.ndarray, max_k: int = 5):
    """
    Finds the optimal k using GMM with BIC on the first Principal Component.
    This is the most robust method identified.
    """
    print("\n--- Programmatically Finding Optimal k using PCA and BIC ---")
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(profiles_scaled)

    k_range = range(1, max_k + 1)
    bic_scores = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        gmm.fit(pc1_scores)
        bic_scores.append(gmm.bic(pc1_scores))

    optimal_k = k_range[np.argmin(bic_scores)]

    print("|   k   |     BIC Score    |")
    print("-" * 28)
    for i, k in enumerate(k_range):
        print(f"|  {k:<3}  |   {bic_scores[i]:<14.2f} | {'<- Optimal' if k == optimal_k else ''}")
    print("-" * 28)
    print(f"Based on the BIC analysis, the optimal number of clusters is: k = {optimal_k}")

    return optimal_k


def find_optimal_clusters_visual(profiles_scaled: np.ndarray, max_k: int = 8):
    """Plots Elbow and Silhouette methods for visual inspection of optimal k."""
    print("\n--- Generating Elbow and Silhouette plots for visual inspection ---")
    k_range = range(2, max_k + 1)
    inertias = []
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(profiles_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(profiles_scaled, kmeans.labels_))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax1.plot(k_range, inertias, marker='o')
    ax1.set_title('Elbow Method for Optimal k', fontweight='bold')
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax2.plot(k_range, silhouette_scores, marker='o')
    ax2.set_title('Silhouette Score for Optimal k', fontweight='bold')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_xticks(k_range)
    save_path = os.path.join(plot_dir, 'optimal_k_visual_analysis.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visual k analysis plot to '{save_path}'")


def analyze_error_strategies(profiles_df: pd.DataFrame, n_clusters: int, title_prefix: str = "", save_suffix: str = ""):
    """Generates heatmaps of cognitive strategies for the given number of clusters."""
    print(f"\n--- Analyzing cognitive strategies for k={n_clusters} ({title_prefix.strip(' :')}) ---")
    clusters_df, sorted_labels = get_strategy_clusters(profiles_df.copy(), n_clusters)
    merged_profiles = profiles_df.join(clusters_df)
    cluster_summary = merged_profiles.groupby('Performance Group').mean().reindex(sorted_labels)

    fig, axes = plt.subplots(1, n_clusters, figsize=(n_clusters * 6, 7), sharey=True)
    if n_clusters == 1: axes = [axes]

    outcome_labels = ['Correct Same', 'Correct Diff', 'Same-for-Diff Err', 'Diff-for-Same Err', 'Invalid Pattern Err']
    attribute_labels = [col.split('_')[0].capitalize() for col in profiles_df.columns if '_Correct_Same' in col]

    for i, label in enumerate(sorted_labels):
        ax = axes[i]
        heatmap_data = cluster_summary.loc[label, profiles_df.columns].to_numpy().reshape(len(attribute_labels), 5)
        sns.heatmap(heatmap_data, ax=ax, annot=True, fmt=".1f", cmap="viridis",
                    xticklabels=outcome_labels, yticklabels=attribute_labels)
        ax.set_title(label, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.suptitle(f'{title_prefix}Cognitive Bias Analysis by Performance Group (k={n_clusters})', fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save_path = os.path.join(plot_dir, f'cognitive_biases_k{n_clusters}{save_suffix}.png')
    plt.savefig(save_path)
    print(f"Saved cognitive bias plot to '{save_path}'")


def plot_clusters_on_pca_axes(human_df: pd.DataFrame, profiles_df: pd.DataFrame, n_clusters: int):
    """
    Visualizes the clusters in the 2D space defined by the first two principal components.
    It orients PC1 to correlate positively with accuracy from the provided human_df.
    """
    print(f"\n--- Visualizing {n_clusters} clusters on PC1 vs. PC2 axes ---")
    clusters_df, sorted_labels = get_strategy_clusters(profiles_df.copy(), n_clusters)

    profiles_proportions = profiles_df.div(profiles_df.sum(axis=1), axis=0)
    scaler = StandardScaler()
    profiles_scaled = scaler.fit_transform(profiles_proportions.fillna(0))
    pca = PCA(n_components=2)
    profiles_pca = pca.fit_transform(profiles_scaled)
    pca_df = pd.DataFrame(data=profiles_pca, columns=['PC1', 'PC2'], index=profiles_df.index).join(clusters_df)

    # Flip PC1 sign to align with performance for consistent plotting
    overall_accuracy = human_df.groupby('prolific_id')['accuracy'].mean()

    pca_df = pca_df.join(overall_accuracy)
    if 'accuracy' in pca_df.columns and pca_df[['PC1', 'accuracy']].corr().iloc[0, 1] < 0:
        print("Flipping sign of PC1 to align with performance.")
        pca_df['PC1'] = -pca_df['PC1']

    plt.figure(figsize=(12, 9))
    variance_explained = pca.explained_variance_ratio_
    pc1_variance, pc2_variance = variance_explained[0] * 100, variance_explained[1] * 100

    if n_clusters == 2:
        label_map = {sorted_labels[0]: 'Expert Group', sorted_labels[1]: 'Developing Group'}
        pca_df['Performance Group'] = pca_df['Performance Group'].map(label_map)
        sorted_labels = [label_map.get(l, l) for l in sorted_labels]

    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Performance Group', hue_order=sorted_labels,
                    palette='viridis', s=100, alpha=0.8)

    plt.title(f'Participant Clusters Visualized on Principal Components (k={n_clusters})', fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({pc1_variance:.1f}% Variance)', fontweight='bold')
    plt.ylabel(f'Principal Component 2 ({pc2_variance:.1f}% Variance)', fontweight='bold')
    plt.legend(title='Strategy Group')
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    save_path = os.path.join(plot_dir, f'pca_clusters_k{n_clusters}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved PCA cluster visualization to '{save_path}'")


def plot_overall_accuracy_by_cluster(human_df: pd.DataFrame, profiles_df: pd.DataFrame, n_clusters: int):
    """Generates box/strip plot for overall accuracy by cluster."""
    print(f"\n--- Plotting overall accuracy for k={n_clusters} clusters ---")
    clusters_df, sorted_labels = get_strategy_clusters(profiles_df.copy(), n_clusters)

    overall_accuracy = human_df.groupby('prolific_id')['accuracy'].mean().reset_index()
    merged_df = pd.merge(overall_accuracy, clusters_df, on='prolific_id')

    plt.figure(figsize=(10, 8))

    sns.boxplot(data=merged_df, x='Performance Group', y='accuracy', order=sorted_labels,
                hue='Performance Group', palette=['lightgrey'] * n_clusters,
                boxprops=dict(alpha=.9), legend=False)

    sns.stripplot(data=merged_df, x='Performance Group', y='accuracy', order=sorted_labels,
                  hue='Performance Group', palette='viridis', jitter=True,
                  alpha=0.8, s=8, legend=False)

    plt.title(f'Overall Accuracy by Strategy Cluster (k={n_clusters})', fontweight='bold', fontsize=16)
    plt.xlabel('Performance Group')
    plt.ylabel('Average Accuracy (across all trials)')
    plt.ylim(0, 1.1)
    save_path = os.path.join(plot_dir, f'overall_accuracy_by_cluster_k{n_clusters}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved overall accuracy plot to '{save_path}'")


def line_plot_strategic_groups(human_df: pd.DataFrame, profiles_df: pd.DataFrame, n_clusters: int):
    """Generates a line plot of accuracy trajectory by cluster."""
    print(f"\n--- Plotting accuracy trajectories for k={n_clusters} clusters ---")

    clusters_df, sorted_labels = get_strategy_clusters(profiles_df.copy(), n_clusters)

    df = human_df.copy()
    df['block'] = (df['trial_num'] - 1) // 30 + 1

    df = df[df['block'].isin([1, 2, 3])]

    block_accuracy = df.groupby(['prolific_id', 'block'])['accuracy'].mean().reset_index()
    merged_df = pd.merge(block_accuracy, clusters_df, on='prolific_id')

    plt.figure(figsize=(12, 8))

    # Filter for participants who have complete data for the three relevant blocks
    id_counts = merged_df['prolific_id'].value_counts()
    complete_ids = id_counts[id_counts == 3].index
    filtered_df_for_lines = merged_df[merged_df['prolific_id'].isin(complete_ids)]

    # Convert 'block' to an ordered Categorical data type to ensure correct plotting
    block_labels = {
        1: 'Block 1 (Train)',
        2: 'Block 2 (Test)',
        3: 'Block 3 (Test)'
    }
    block_order = [block_labels[i] for i in [1, 2, 3]]

    merged_df['block'] = pd.Categorical(merged_df['block'].map(block_labels), categories=block_order, ordered=True)
    if not filtered_df_for_lines.empty:
        filtered_df_for_lines['block'] = pd.Categorical(filtered_df_for_lines['block'].map(block_labels),
                                                        categories=block_order, ordered=True)

    # Plot faint lines for each participant
    if not filtered_df_for_lines.empty:
        sns.lineplot(
            data=filtered_df_for_lines,
            x='block',
            y='accuracy',
            hue='Performance Group',
            hue_order=sorted_labels,
            units='prolific_id',
            estimator=None,
            alpha=0.1,
            palette='viridis',
            legend=False
        )

    # Overlay a bold line for the mean of each group
    sns.lineplot(
        data=merged_df,
        x='block',
        y='accuracy',
        hue='Performance Group',
        hue_order=sorted_labels,
        palette='viridis',
        marker='o',
        markersize=10,
        linewidth=4,
        errorbar=('ci', 95)
    )

    plt.title(f'Accuracy Trajectory by Performance Group (k={n_clusters})', fontweight='bold', fontsize=16)
    plt.xlabel('Block')
    plt.ylabel('Average Accuracy')
    plt.ylim(0, 1.05)
    plt.legend(title='Performance Group')
    save_path = os.path.join(plot_dir, f'accuracy_trajectories_k{n_clusters}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved accuracy trajectory plot to '{save_path}'")


def plot_pca_joint_distribution(human_df: pd.DataFrame, profiles_df: pd.DataFrame, n_clusters: int):
    """
    Generates a joint plot showing the relationship between PC1 and PC2,
    with marginal distributions for each component. Gracefully handles k=1.
    """
    print(f"\n--- Plotting joint distribution of PC1 and PC2 for k={n_clusters} clusters ---")

    # Recalculate PCA to get PC1 and PC2 scores
    profiles_proportions = profiles_df.div(profiles_df.sum(axis=1), axis=0)
    scaler = StandardScaler()
    profiles_scaled = scaler.fit_transform(profiles_proportions.fillna(0))
    pca = PCA(n_components=2)
    profiles_pca = pca.fit_transform(profiles_scaled)

    pca_df = pd.DataFrame(data=profiles_pca, columns=['PC1', 'PC2'], index=profiles_df.index)

    # Align PC1 Score with performance for consistent interpretation
    overall_accuracy = human_df.groupby('prolific_id')['accuracy'].mean()
    pca_df = pca_df.join(overall_accuracy)
    if 'accuracy' in pca_df.columns and pca_df[['PC1', 'accuracy']].corr().iloc[0, 1] < 0:
        pca_df['PC1'] = -pca_df['PC1']

    if n_clusters == 1:
        # For k=1, we don't need cluster labels or hue.
        g = sns.jointplot(
            data=pca_df,
            x='PC1',
            y='PC2',
            kind='scatter',  # Use scatter for single group
            height=8,
            color='skyblue'
        )
    else:
        # For k>1, use the original logic with hue for coloring.
        clusters_df, sorted_labels = get_strategy_clusters(profiles_df.copy(), n_clusters)
        pca_df = pca_df.join(clusters_df)

        # Use descriptive labels for the common k=2 case
        if n_clusters == 2:
            label_map = {sorted_labels[0]: 'Expert Group', sorted_labels[1]: 'Developing Group'}
            pca_df['Performance Group'] = pca_df['Performance Group'].map(label_map)
            sorted_labels = [label_map[l] for l in sorted_labels]
        else:
            pca_df.rename(columns={'Performance Group': 'Cluster'}, inplace=True)
            sorted_labels = clusters_df['Performance Group'].unique()

        g = sns.jointplot(
            data=pca_df,
            x='PC1',
            y='PC2',
            hue='Cluster' if n_clusters > 2 else 'Performance Group',
            hue_order=sorted_labels,
            palette='viridis',
            kind='scatter',
            height=8
        )

    variance_explained = pca.explained_variance_ratio_
    pc1_variance = variance_explained[0] * 100
    pc2_variance = variance_explained[1] * 100

    g.set_axis_labels(
        f'Principal Component 1 ({pc1_variance:.1f}% Variance)',
        f'Principal Component 2 ({pc2_variance:.1f}% Variance)',
        fontweight='bold'
    )

    g.fig.suptitle(f'Joint Distribution of Principal Components (k={n_clusters})', y=1.02, fontweight='bold')

    save_path = os.path.join(plot_dir, f'pca_joint_distribution_k{n_clusters}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved PCA joint distribution plot to '{save_path}'")

def profile_clusters_with_survey_data(human_df: pd.DataFrame, profiles_df: pd.DataFrame, n_clusters: int):
    """
    Analyzes multiple-choice survey responses for each performance cluster.

    For each of the 8 survey questions, this function creates a contingency table
    and performs a Chi-squared test to see if the distribution of answers
    is significantly different across the performance groups.
    """
    print("\n" + "=" * 50)
    print("--- Profiling Performance Clusters with Survey Data ---")
    print("=" * 50)

    # Step 1: Get the performance group for each participant
    clusters_df, sorted_labels = get_strategy_clusters(profiles_df.copy(), n_clusters)

    # Step 2: Extract unique survey responses for each participant
    survey_cols = [f'feedback_{i}' for i in range(1, 9)]  # 8 questions
    if not all(col in human_df.columns for col in survey_cols):
        print("Error: One or more 'feedback_n' columns (1-8) not found.")
        return

    unique_responses = human_df.groupby('prolific_id')[survey_cols].first().reset_index()

    # Step 3: Merge the cluster labels with the unique survey responses
    profile_df = pd.merge(clusters_df, unique_responses, on='prolific_id')

    # Step 4: Loop through each survey question and perform Chi-squared analysis
    for col in survey_cols:
        print(f"\n--- Analysis for Survey Question: {col} ---")

        # Drop participants who did not answer this question
        question_df = profile_df[['Performance Group', col]].dropna()

        # Create a contingency table (cross-tabulation) of responses vs. performance group
        contingency_table = pd.crosstab(question_df[col], question_df['Performance Group'])

        # Ensure the columns are in a logical, performance-sorted order
        if all(label in contingency_table.columns for label in sorted_labels):
            contingency_table = contingency_table[sorted_labels]

        print("\nResponse Counts by Performance Group:")
        print(contingency_table)

        # Perform the Chi-squared test
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            print(f"\nChi-squared Test Results:")
            print(f"  Statistic: {chi2:.2f}")
            print(f"  p-value: {p_value:.4f}")

            # Interpret the result
            if p_value < 0.05:
                print(
                    "  --> Result is STATISTICALLY SIGNIFICANT. Response distribution is associated with performance group.")
            else:
                print("  --> No significant association found between response distribution and performance group.")
        except ValueError:
            print("\nCould not perform Chi-squared test (likely due to insufficient data in a category).")


def profile_groups_by_median_split(human_df: pd.DataFrame, profiles_df: pd.DataFrame):
    """
    Creates two performance groups using a median split on the PC1 Strategy Score
    and runs a Chi-squared test on survey data for these groups.
    """
    print("\n" + "=" * 60)
    print("--- Profiling Groups via Median Split with Survey Data ---")
    print("=" * 60)

    # Step 1: Calculate PC1 Strategy Score for all participants
    profiles_proportions = profiles_df.div(profiles_df.sum(axis=1), axis=0)
    scaler = StandardScaler()
    profiles_scaled = scaler.fit_transform(profiles_proportions.fillna(0))
    pca = PCA(n_components=1)

    # Create a new DataFrame for this analysis
    split_df = profiles_df.copy()
    split_df['PC1_Score'] = pca.fit_transform(profiles_scaled)

    # Align PC1 with performance
    overall_accuracy = human_df.groupby('prolific_id')['accuracy'].mean()
    split_df = split_df.join(overall_accuracy)
    if 'accuracy' in split_df.columns and split_df[['PC1_Score', 'accuracy']].corr().iloc[0, 1] < 0:
        split_df['PC1_Score'] = -split_df['PC1_Score']

    # Step 2: Perform the median split
    median_score = split_df['PC1_Score'].median()
    split_df['Performance Group'] = np.where(split_df['PC1_Score'] >= median_score, 'High', 'Low')
    print(f"Performed median split on PC1 Score at a value of: {median_score:.2f}")
    print("Group sizes from median split:")
    print(split_df['Performance Group'].value_counts())

    # Step 3: Extract unique survey responses
    survey_cols = [f'feedback_{i}' for i in range(1, 9)]
    if not all(col in human_df.columns for col in survey_cols):
        print("Error: One or more 'feedback_n' columns (1-8) not found.")
        return
    unique_responses = human_df.groupby('prolific_id')[survey_cols].first().reset_index()

    # Step 4: Merge the new groups with the survey responses
    profile_df = pd.merge(split_df[['Performance Group']], unique_responses, on='prolific_id')

    # Step 5: Loop through each survey question and perform Chi-squared analysis
    sorted_labels = ['High', 'Low']
    for col in survey_cols:
        print(f"\n--- Analysis for Survey Question: {col} ---")
        question_df = profile_df[['Performance Group', col]].dropna()
        contingency_table = pd.crosstab(question_df[col], question_df['Performance Group'])
        if all(label in contingency_table.columns for label in sorted_labels):
            contingency_table = contingency_table[sorted_labels]

        print("\nResponse Counts by Performance Group (Median Split):")
        print(contingency_table)

        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            print(f"\nChi-squared Test Results:")
            print(f"  Statistic: {chi2:.2f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print("  --> Result is STATISTICALLY SIGNIFICANT.")
            else:
                print("  --> No significant association found.")
        except ValueError:
            print("\nCould not perform Chi-squared test.")


def analyze_clusters_from_training_block(human_df: pd.DataFrame):
    """
    Determines the optimal number of learner types based on Block 1 data, then analyzes
    their subsequent performance and survey responses.
    """
    print("\n" + "#" * 60)
    print("###  Analysis Based on Training Performance (Block 1)  ###")
    print("#" * 60)

    # --- Step 1: Isolate Block 1 data and create behavioral profiles ---
    print("\n--- Step 1: Generating behavioral profiles from Block 1 data ---")
    df_block1 = human_df[human_df['block'] == 1]
    profiles_block1 = get_participant_profiles(df_block1)
    profiles_block1.dropna(inplace=True)
    print(f"Found {len(profiles_block1)} participants with sufficient data in Block 1.")

    # --- Step 2: Scale the Block 1 profiles for PCA ---
    profiles_proportions_b1 = profiles_block1.div(profiles_block1.sum(axis=1), axis=0)
    scaler_b1 = StandardScaler()
    profiles_scaled_b1 = scaler_b1.fit_transform(profiles_proportions_b1.fillna(0))

    # --- Step 3: Programmatically find the optimal k for Block 1 data ---
    print("\n--- Step 3: Determining optimal number of clusters for Block 1 profiles ---")
    optimal_k_block1 = find_optimal_k_programmatically(profiles_scaled_b1)

    # --- Step 4: Perform Clustering on Block 1 profiles using the optimal k ---
    print(f"\n--- Step 4: Clustering participants into {optimal_k_block1} groups based on Block 1 profiles ---")
    clusters_block1, sorted_labels_block1 = get_strategy_clusters(profiles_block1.copy(), n_clusters=optimal_k_block1)
    clusters_block1.rename(columns={'Performance Group': 'Block 1 Group'}, inplace=True)

    print(f"Cluster sizes based on Block 1 performance (k={optimal_k_block1}):")
    print(clusters_block1['Block 1 Group'].value_counts())

    # --- Step 5: Analyze performance of these groups in Test Blocks (2 & 3) ---
    # The T-test is only valid for comparing two groups.
    if optimal_k_block1 == 2:
        print("\n--- Step 5: Testing if Block 1 groups perform differently in Blocks 2 & 3 ---")
        df_test = human_df[human_df['block'].isin([2, 3])]
        test_accuracy = df_test.groupby('prolific_id')['accuracy'].mean().reset_index()
        test_accuracy.rename(columns={'accuracy': 'test_accuracy'}, inplace=True)

        merged_test_df = pd.merge(clusters_block1, test_accuracy, on='prolific_id')

        group1_label, group2_label = sorted_labels_block1[0], sorted_labels_block1[1]
        group1_scores = merged_test_df[merged_test_df['Block 1 Group'] == group1_label]['test_accuracy']
        group2_scores = merged_test_df[merged_test_df['Block 1 Group'] == group2_label]['test_accuracy']

        print(f"  Mean Test Accuracy for '{group1_label}' group (from Block 1): {group1_scores.mean():.3f}")
        print(f"  Mean Test Accuracy for '{group2_label}' group (from Block 1): {group2_scores.mean():.3f}")

        t_stat, p_value = ttest_ind(group1_scores, group2_scores, nan_policy='omit')
        print(f"\nT-test Results:")
        print(f"  T-statistic: {t_stat:.2f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("  --> Result is STATISTICALLY SIGNIFICANT. The groups identified in Block 1 perform differently in the test blocks.")
        else:
            print("  --> No significant difference in test block performance between the groups identified in Block 1.")
    else:
        print(f"\n--- Step 5: Skipping T-test for test block performance (requires k=2, but found k={optimal_k_block1}) ---")


    # --- Step 6: Analyze Survey Responses for the Block 1 groups ---
    print("\n--- Step 6: Testing for correlation with survey responses ---")
    survey_cols = [f'feedback_{i}' for i in range(1, 9)]
    unique_responses = human_df.groupby('prolific_id')[survey_cols].first().reset_index()
    profile_df = pd.merge(clusters_block1, unique_responses, on='prolific_id')

    for col in survey_cols:
        print(f"\n--- Analysis for Survey Question: {col} ---")
        question_df = profile_df[['Block 1 Group', col]].dropna()
        contingency_table = pd.crosstab(question_df[col], question_df['Block 1 Group'])
        if all(label in contingency_table.columns for label in sorted_labels_block1):
            contingency_table = contingency_table[sorted_labels_block1]

        print("\nResponse Counts by Block 1 Group:")
        print(contingency_table)

        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            print(f"\nChi-squared Test Results:")
            print(f"  Statistic: {chi2:.2f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print("  --> Result is STATISTICALLY SIGNIFICANT. Response is associated with Block 1 group.")
            else:
                print("  --> No significant association found.")
        except ValueError:
            print("\nCould not perform Chi-squared test (likely due to insufficient data).")


def run_full_analysis_for_blocks(human_df: pd.DataFrame, blocks: list, description: str):
    """
    A self-contained function that runs a full analysis pipeline on a specific subset of blocks.
    It finds the optimal k for the subset and then runs the full survey correlation analysis.
    """
    print("\n" + "#" * 60)
    print(f"###  Full Analysis for: {description}  ###")
    print("#" * 60)

    # Step 1: Isolate data for the specified blocks
    df_subset = human_df[human_df['block'].isin(blocks)]
    if df_subset.empty:
        print(f"No data found for the specified blocks: {blocks}. Skipping analysis.")
        return

    # Step 2: Create and scale behavioral profiles for the subset
    profiles_subset = get_participant_profiles(df_subset)
    profiles_subset.dropna(inplace=True)

    if profiles_subset.empty:
        print(f"Could not generate sufficient profiles for {description}. Skipping analysis.")
        return

    print(f"\nAnalyzing {len(profiles_subset)} participants with data in {description}.")
    profiles_proportions_subset = profiles_subset.div(profiles_subset.sum(axis=1), axis=0)
    scaler = StandardScaler()
    profiles_scaled_subset = scaler.fit_transform(profiles_proportions_subset.fillna(0))

    # Step 3: Programmatically find the optimal k for the subset
    optimal_k_subset = find_optimal_k_programmatically(profiles_scaled_subset)

    # Step 4: Run the full survey analysis for these clusters
    # The t-test and other logic inside this function is skipped if k != 2
    profile_clusters_with_survey_data(human_df, profiles_subset.copy(), n_clusters=optimal_k_subset)


def plot_significant_survey_result(human_df: pd.DataFrame, block_to_analyze: int, question_to_plot: str):
    """
    Generates a normalized bar chart to visualize a significant Chi-squared result.
    """
    print(f"\n--- Generating plot for significant survey result (Block {block_to_analyze}, {question_to_plot}) ---")

    # --- Isolate data and find optimal clusters for the specified block ---
    df_block = human_df[human_df['block'] == block_to_analyze]
    profiles_block = get_participant_profiles(df_block)
    profiles_block.dropna(inplace=True)

    profiles_proportions = profiles_block.div(profiles_block.sum(axis=1), axis=0)
    scaler = StandardScaler()
    profiles_scaled = scaler.fit_transform(profiles_proportions.fillna(0))

    optimal_k = find_optimal_k_programmatically(profiles_scaled, max_k=5)
    clusters_df, sorted_labels = get_strategy_clusters(profiles_block.copy(), n_clusters=optimal_k)

    # --- Prepare data for plotting ---
    survey_cols = [f'feedback_{i}' for i in range(1, 9)]
    unique_responses = human_df.groupby('prolific_id')[survey_cols].first().reset_index()
    profile_df = pd.merge(clusters_df, unique_responses, on='prolific_id')

    # Create a normalized crosstab (proportions)
    crosstab_norm = pd.crosstab(profile_df[question_to_plot], profile_df['Performance Group'], normalize='columns')
    crosstab_norm = crosstab_norm.mul(100).reindex(sorted_labels, axis=1)  # Convert to percentage and sort

    ax = crosstab_norm.T.plot(kind='bar', stacked=True, figsize=(12, 8), cmap='viridis', alpha=0.8)

    plt.title(f'Self-Reported Strategy ({question_to_plot}) by Performance Group in Block {block_to_analyze}',
              fontweight='bold', fontsize=16)
    plt.ylabel('Percentage of Participants (%)', fontweight='bold')
    plt.xlabel('Performance Group (derived from Block 2 behavior)', fontweight='bold')
    plt.xticks(rotation=0)
    plt.legend(title='Reported Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add percentage labels to the bars
    for container in ax.containers:
        ax.bar_label(container, labels=[f'{v:.1f}%' for v in container.datavalues], label_type='center', color='white',
                     fontweight='bold')

    save_path = os.path.join(plot_dir, 'significant_survey_result.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved significant survey result plot to '{save_path}'")


def generate_block_by_block_heatmaps(human_df: pd.DataFrame):
    """
    Runs the error strategy heatmap analysis for each block individually.
    """
    print("\n" + "#" * 60)
    print("###  Block-by-Block Analysis of Strategy Heatmaps  ###")
    print("#" * 60)

    for block_num in [1, 2, 3]:
        description = f"Block {block_num}"
        df_block = human_df[human_df['block'] == block_num]

        profiles_block = get_participant_profiles(df_block)
        profiles_block.dropna(inplace=True)

        if profiles_block.empty:
            print(f"\nSkipping {description}: not enough data.")
            continue

        profiles_proportions = profiles_block.div(profiles_block.sum(axis=1), axis=0)
        scaler = StandardScaler()
        profiles_scaled = scaler.fit_transform(profiles_proportions.fillna(0))

        optimal_k = find_optimal_k_programmatically(profiles_scaled)

        analyze_error_strategies(
            profiles_block.copy(),
            n_clusters=optimal_k,
            title_prefix=f"{description}: ",
            save_suffix=f"_block{block_num}"
        )

def plot_agent_clusters_on_pca_axes(params_df: pd.DataFrame, n_clusters: int):
    """
    A dedicated function to visualize agent parameter clusters on PC1 vs. PC2.
    This version does not require human_df for axis orientation.
    """
    print(f"\n--- Visualizing {n_clusters} agent clusters on PC1 vs. PC2 axes ---")

    # The params_df is used as the basis for clustering
    # The get_strategy_clusters function now expects 'prolific_id' as a column
    # but params_df has it as an index. We reset it for the call.
    clusters_df, sorted_labels = get_strategy_clusters(params_df.reset_index().copy(), n_clusters=n_clusters)

    # Scale the parameters and perform PCA
    scaler = StandardScaler()
    params_scaled = scaler.fit_transform(params_df)
    pca = PCA(n_components=2)
    profiles_pca = pca.fit_transform(params_scaled)

    pca_df = pd.DataFrame(data=profiles_pca, columns=['PC1', 'PC2'], index=params_df.index)
    pca_df = pca_df.merge(clusters_df.set_index('prolific_id'), left_index=True, right_index=True)

    plt.figure(figsize=(12, 9))
    variance_explained = pca.explained_variance_ratio_
    pc1_variance, pc2_variance = variance_explained[0] * 100, variance_explained[1] * 100

    if n_clusters == 2:
        label_map = {sorted_labels[0]: 'Expert Group', sorted_labels[1]: 'Developing Group'}
        pca_df['Performance Group'] = pca_df['Performance Group'].map(label_map)
        sorted_labels = [label_map.get(l, l) for l in sorted_labels]

    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Performance Group', hue_order=sorted_labels,
                    palette='viridis', s=100, alpha=0.8)

    plt.title(f'Agent Model Parameter Clusters Visualized on Principal Components (k={n_clusters})', fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({pc1_variance:.1f}% Variance)', fontweight='bold')
    plt.ylabel(f'Principal Component 2 ({pc1_variance:.1f}% Variance)', fontweight='bold')
    plt.legend(title='Agent Strategy Group')
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)

    save_path = os.path.join(plot_dir, f'pca_clusters_agents_k{n_clusters}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved Agent PCA cluster visualization to '{save_path}'")


def infer_parameters_heuristically(human_df: pd.DataFrame, save_path: str):
    """
    Infers learning rate and selection temperature (tau) for each participant
    using the heuristic methods from the user's reference code.
    """
    print("\n" + "#" * 60)
    print("###  Inferring Human Parameters Using Heuristic Methods  ###")
    print("#" * 60)

    df, attributes = _prepare_and_validate_data(human_df)
    if df is None: return None

    # --- Infer General Learning Rate ---
    participant_params = []
    for pid, group in df.groupby('prolific_id'):
        df_participant = group.copy()

        # Calculate learning rate using the logistic regression method
        df_participant['is_same_trial'] = (df_participant['modSum_color'] == 1)
        df_participant['past_perf_same'] = df_participant.loc[df_participant['is_same_trial'], 'accuracy'].shift(
            1).rolling(window=5, min_periods=1).mean()
        df_participant['past_perf_diff'] = df_participant.loc[~df_participant['is_same_trial'], 'accuracy'].shift(
            1).rolling(window=5, min_periods=1).mean()
        df_participant.ffill(inplace=True)
        df_participant.fillna(0.5, inplace=True)
        X = df_participant[['past_perf_same', 'past_perf_diff']]
        y = df_participant['accuracy']
        learning_rate = np.mean(LogisticRegression(solver='liblinear').fit(X, y).coef_[0]) if y.nunique() > 1 else 0.0

        # --- Infer Temperature (Tau) ---
        # This uses the direct formula from your reference code
        temp_min, temp_max, gamma = 0.01, 1.5, 2.0
        mean_accuracy = df_participant['accuracy'].mean()
        temperature = temp_min + (temp_max - temp_min) * ((1 - np.clip(mean_accuracy, 0.0, 1.0)) ** gamma)

        participant_params.append({
            'prolific_id': pid,
            'learning_rate': learning_rate,
            'selection_temperature': temperature
        })

    params_df = pd.DataFrame(participant_params)
    params_df.to_csv(save_path, index=False)
    print(f"Successfully inferred and saved parameters for {len(params_df)} participants to '{save_path}'")
    print("Sample of inferred parameters:")
    print(params_df.head())
    return params_df


def analyze_information_driven_search(human_df: pd.DataFrame, processed_data: dict):
    """
    Calculates an Information-Driven Search (IDS) score for each participant to classify their strategy.
    """
    print("\n" + "#" * 60)
    print("###  Classifying Strategies by Information-Driven Search (IDS) Score  ###")
    print("#" * 60)

    # --- Step 1: Calculate Information Value (IV) for each attribute on each trial ---
    relation_matrices = processed_data["relation_matrices"]
    information_values = np.zeros_like(relation_matrices[:, 0, :], dtype=float)  # Shape: (n_trials, n_attributes)

    for i in range(relation_matrices.shape[0]):  # Loop through trials
        for j in range(relation_matrices.shape[2]):  # Loop through attributes
            # Find the size of the smallest non-empty candidate bin for this attribute
            counts = np.bincount(relation_matrices[i, :, j], minlength=4)
            non_empty_counts = counts[1:][counts[1:] > 0]
            min_bin_size = np.min(non_empty_counts) if len(non_empty_counts) > 0 else 10
            # Information Value is how many candidates are eliminated
            information_values[i, j] = 10 - min_bin_size

    # --- Step 2: Calculate IDS Score for each participant ---
    attributes = ['color', 'fill', 'shape', 'back']
    acc_cols = [f'cho{attr[0].upper()}_acc' for attr in attributes]

    participant_scores = []
    for pid, group in human_df.groupby('prolific_id'):
        trial_scores = []
        for trial_idx in range(90):
            try:
                trial_data = group[group['trial_num'] == trial_idx + 1].iloc[0]
            except IndexError:
                continue  # Skip if trial data is missing for this participant

            iv_for_trial = information_values[trial_idx, :]
            acc_for_trial = trial_data[acc_cols].values

            # Calculate score based on whether correct perceptions were on informative attributes
            score_correct = np.sum(iv_for_trial[acc_for_trial == 1])
            score_incorrect = np.sum(iv_for_trial[acc_for_trial == 0])
            trial_score = score_correct - score_incorrect
            trial_scores.append(trial_score)

        if trial_scores:
            participant_scores.append({'prolific_id': pid, 'ids_score': np.mean(trial_scores)})

    ids_df = pd.DataFrame(participant_scores)

    # --- Step 3: Cluster on IDS Score and Visualize ---
    features_for_clustering = ids_df[['ids_score']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_for_clustering)

    print("\n--- Determining optimal number of IDS clusters ---")
    optimal_k = find_optimal_k_programmatically(features_scaled, max_k=5)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    ids_df['Strategy Group'] = kmeans.fit_predict(features_scaled)

    # Create meaningful labels by sorting clusters by IDS score
    cluster_summary = ids_df.groupby('Strategy Group')['ids_score'].mean().sort_values(ascending=False)
    label_map = {cid: f'IDS Group {i + 1}' for i, cid in enumerate(cluster_summary.index)}
    ids_df['Strategy Group'] = ids_df['Strategy Group'].map(label_map)
    sorted_labels = [label_map[idx] for idx in cluster_summary.index]

    print(f"\nFound {optimal_k} strategy groups based on IDS Score:")
    print(ids_df.groupby('Strategy Group')['ids_score'].mean().round(2))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(data=ids_df, x='ids_score', hue='Strategy Group', hue_order=sorted_labels,
                 multiple="layer", alpha=0.5, kde=True, ax=ax1, palette='viridis')
    ax1.set_title('Distribution of Information-Driven Search (IDS) Scores', fontweight='bold')

    # Merge with overall accuracy for a scatter plot
    overall_accuracy = human_df.groupby('prolific_id')['accuracy'].mean().reset_index()
    merged_plot_df = pd.merge(ids_df, overall_accuracy, on='prolific_id')

    sns.scatterplot(data=merged_plot_df, x='ids_score', y='accuracy', hue='Strategy Group',
                    hue_order=sorted_labels, s=80, alpha=0.8, ax=ax2, palette='viridis')
    ax2.set_title('IDS Score vs. Overall Accuracy', fontweight='bold')
    ax2.set_xlabel('IDS Score')
    ax2.set_ylabel('Overall Average Accuracy')

    save_path = os.path.join(plot_dir, 'ids_score_analysis.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nSaved IDS analysis plot to '{save_path}'")


def run_human_behavioral_analysis(human_df: pd.DataFrame):
    """
    Performs the definitive, correct clustering analysis on the human behavioral data.
    """
    print("\n" + "#" * 60)
    print("###  Final Human Behavioral Cluster Analysis  ###")
    print("#" * 60)

    # Step 1: Create the 20-feature behavioral profiles
    human_profiles_df = get_participant_profiles(human_df)
    cols_to_drop = [col for col in human_profiles_df.columns if 'Invalid_Pattern_Err' in col]
    profiles_for_pca = human_profiles_df.drop(columns=cols_to_drop)

    # Step 2: Standardize the profiles
    human_proportions = profiles_for_pca.div(profiles_for_pca.sum(axis=1), axis=0)
    scaler = StandardScaler()
    human_scaled = scaler.fit_transform(human_proportions.fillna(0))

    # Step 3: Apply PCA to get the PC1 scores
    print("\n--- Applying PCA to behavioral profiles ---")
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(human_scaled)
    print("PCA complete. PC1 scores generated.")

    # Step 4: Run BIC analysis on the 1D PC1 scores
    optimal_k_human = find_optimal_k_programmatically(pc1_scores)

    # Step 5: Get cluster assignments and report
    correct_cols = [col for col in human_profiles_df.columns if 'Correct' in col]
    human_profiles_df['total_correct'] = human_profiles_df[correct_cols].sum(axis=1)

    human_clusters, _ = get_strategy_clusters(human_profiles_df, n_clusters=optimal_k_human)
    print(f"\nOptimal number of HUMAN behavioral clusters is: {optimal_k_human}")
    print("Human Cluster Sizes:\n", human_clusters['Performance Group'].value_counts())


def plot_final_heatmap_comparison(human_df: pd.DataFrame, agent_df: pd.DataFrame):
    """
    Generates a final heatmap figure comparing the behavioral strategies of human and
    agent clusters. This version dynamically determines the optimal k for each
    population and adjusts the plot grid accordingly.
    """
    print("\n--- Generating Final Heatmap Comparison ---")

    # --- 1. Process Human Data to find optimal k and clusters ---
    human_profiles = get_participant_profiles(human_df)
    human_proportions = human_profiles.div(human_profiles.sum(axis=1), axis=0)
    scaler_human = StandardScaler()
    human_scaled = scaler_human.fit_transform(human_proportions.fillna(0))
    pca_human_1d = PCA(n_components=1)
    pc1_scores_human = pca_human_1d.fit_transform(human_scaled)
    optimal_k_human = find_optimal_k_programmatically(pc1_scores_human)
    human_clusters, human_labels = get_strategy_clusters(human_profiles.copy(), n_clusters=optimal_k_human)
    human_merged = human_profiles.join(human_clusters.set_index('prolific_id'))
    human_summary = human_merged.groupby('Performance Group').mean().reindex(human_labels)

    # --- 2. Process Agent Data to find optimal k and clusters ---
    agent_profiles = get_participant_profiles(agent_df)
    agent_proportions = agent_profiles.div(agent_profiles.sum(axis=1), axis=0)
    scaler_agent = StandardScaler()
    agent_scaled = scaler_agent.fit_transform(agent_proportions.fillna(0))
    pca_agent_1d = PCA(n_components=1)
    pc1_scores_agent = pca_agent_1d.fit_transform(agent_scaled)
    optimal_k_agent = find_optimal_k_programmatically(pc1_scores_agent)
    agent_clusters, agent_labels = get_strategy_clusters(agent_profiles.copy(), n_clusters=optimal_k_agent)
    agent_merged = agent_profiles.join(agent_clusters.set_index('prolific_id'))
    agent_summary = agent_merged.groupby('Performance Group').mean().reindex(agent_labels)

    # --- 3. Create the Plot Dynamically ---
    # The number of columns will be the max of the two k's
    num_cols = max(optimal_k_human, optimal_k_agent)
    fig, axes = plt.subplots(2, num_cols, figsize=(num_cols * 6, 14), sharex=True, sharey=True)
    fig.suptitle('Comparison of Human vs. Agent Behavioral Strategies', fontsize=20, fontweight='bold')

    outcome_labels = ['Correct Same', 'Correct Diff', 'Same-for-Diff Err', 'Diff-for-Same Err', 'Invalid Pattern Err']
    attribute_labels = [col.split('_')[0].capitalize() for col in human_profiles.columns if '_Correct_Same' in col]

    # Plot Human Heatmaps
    for i, label in enumerate(human_labels):
        ax = axes[0, i] if num_cols > 1 else axes[0]
        heatmap_data = human_summary.loc[label].to_numpy().reshape(len(attribute_labels), 5)
        sns.heatmap(heatmap_data, ax=ax, annot=True, fmt=".1f", cmap="viridis",
                    xticklabels=outcome_labels, yticklabels=attribute_labels)
        ax.set_title(f"Human '{label}' Group", fontweight='bold', fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Plot Agent Heatmaps
    for i, label in enumerate(agent_labels):
        ax = axes[1, i] if num_cols > 1 else axes[1]
        heatmap_data = agent_summary.loc[label].to_numpy().reshape(len(attribute_labels), 5)
        sns.heatmap(heatmap_data, ax=ax, annot=True, fmt=".1f", cmap="viridis",
                    xticklabels=outcome_labels, yticklabels=attribute_labels)
        ax.set_title(f"Agent '{label}' Group", fontweight='bold', fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Hide unused subplots if k's are different
    if optimal_k_human < num_cols:
        for i in range(optimal_k_human, num_cols): axes[0, i].set_visible(False)
    if optimal_k_agent < num_cols:
        for i in range(optimal_k_agent, num_cols): axes[1, i].set_visible(False)

    save_path = os.path.join(plot_dir, 'final_heatmap_comparison.png')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(save_path)
    print(f"Saved final heatmap comparison plot to '{save_path}'")

def plot_human_parameter_clusters(human_params_df: pd.DataFrame):
    """
    Performs clustering and generates a PCA scatter plot for human inferred parameters.
    """
    print("\n--- Visualizing Human Parameter Clusters on PC1 vs. PC2 axes ---")

    # --- 1. Cluster HUMANS based on their inferred cognitive parameters ---
    features_to_cluster = human_params_df[['inferred_tau', 'inferred_influence']]
    scaler = StandardScaler()
    params_scaled = scaler.fit_transform(features_to_cluster)
    optimal_k = find_optimal_k_programmatically(params_scaled)

    temp_profiles_df = human_params_df.copy()
    temp_profiles_df['total_knowledge'] = temp_profiles_df['inferred_influence']

    clusters_df, sorted_labels = get_strategy_clusters(temp_profiles_df, n_clusters=optimal_k)
    print(f"Found {optimal_k} human cognitive profiles.")
    print("Human Parameter Cluster Sizes:\n", clusters_df['Performance Group'].value_counts())

    # --- 2. Perform PCA and Prepare for Plotting ---
    pca = PCA(n_components=2)
    params_pca = pca.fit_transform(params_scaled)
    pca_df = pd.DataFrame(data=params_pca, columns=['PC1', 'PC2'])
    pca_df['prolific_id'] = human_params_df['prolific_id'].values

    plot_df = pd.merge(pca_df, clusters_df, on='prolific_id')

    # --- 3. Plotting ---
    plt.figure(figsize=(12, 9))
    variance_explained = pca.explained_variance_ratio_
    pc1_variance, pc2_variance = variance_explained[0] * 100, variance_explained[1] * 100

    sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Performance Group', hue_order=sorted_labels,
                    palette='viridis', s=100, alpha=0.8)

    plt.title(f'Human Inferred Parameter Clusters Visualized on Principal Components (k={optimal_k})',
              fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({pc1_variance:.1f}%)', fontweight='bold')
    plt.ylabel(f'Principal Component 2 ({pc2_variance:.1f}%)', fontweight='bold')
    plt.legend(title='Human Strategy Group')
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)

    save_path = os.path.join(plot_dir, f'pca_clusters_humans_k{optimal_k}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved Human Parameter cluster visualization to '{save_path}'")


def plot_final_behavioral_pca_comparison(human_df: pd.DataFrame, agent_df: pd.DataFrame):
    """
    Creates a side-by-side figure comparing the PCA-based behavioral clusters of
    human participants and simulated agents. This version dynamically determines the
    optimal k for each population and plots accordingly.
    """
    print("\n--- Generating final comparative PCA plot for Human vs. Agent BEHAVIOR ---")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
    fig.suptitle('Comparison of Human vs. Agent Behavioral Clusters', fontsize=20, fontweight='bold')

    # --- 1. Process and Plot Human Data ---
    human_profiles = get_participant_profiles(human_df)
    human_proportions = human_profiles.div(human_profiles.sum(axis=1), axis=0)
    scaler_human = StandardScaler()
    human_scaled = scaler_human.fit_transform(human_proportions.fillna(0))

    # Find optimal k for humans based on PC1 of their behavior
    pca_human_1d = PCA(n_components=1)
    pc1_scores_human = pca_human_1d.fit_transform(human_scaled)
    optimal_k_human = find_optimal_k_programmatically(pc1_scores_human)

    # Get clusters for the optimal k
    human_clusters, human_labels = get_strategy_clusters(human_profiles.copy(), n_clusters=optimal_k_human)

    # Get 2D PCA for plotting
    pca_human_2d = PCA(n_components=2)
    human_pca_coords = pca_human_2d.fit_transform(human_scaled)
    human_plot_df = pd.DataFrame(human_pca_coords, columns=['PC1', 'PC2'], index=human_profiles.index)
    human_plot_df = pd.merge(human_plot_df.reset_index(), human_clusters, on='prolific_id')

    # Plot Human Data
    if optimal_k_human == 1:
        sns.scatterplot(data=human_plot_df, x='PC1', y='PC2', color='skyblue', s=100, alpha=0.8, ax=ax1)
    else:
        sns.scatterplot(data=human_plot_df, x='PC1', y='PC2', hue='Performance Group', hue_order=human_labels,
                        palette='viridis', s=100, alpha=0.8, ax=ax1)
        ax1.legend(title='Human Performance Group')

    pc1_var_h = pca_human_2d.explained_variance_ratio_[0] * 100
    pc2_var_h = pca_human_2d.explained_variance_ratio_[1] * 100
    ax1.set_title(f'Human Behavioral Clusters (k={optimal_k_human})', fontweight='bold')
    ax1.set_xlabel(f'Principal Component 1 ({pc1_var_h:.1f}%)')
    ax1.set_ylabel(f'Principal Component 2 ({pc2_var_h:.1f}%)')
    ax1.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    ax1.axhline(0, color='grey', linestyle='--', linewidth=0.5)

    # --- 2. Process and Plot Agent Data ---
    agent_profiles = get_participant_profiles(agent_df)
    agent_proportions = agent_profiles.div(agent_profiles.sum(axis=1), axis=0)
    scaler_agent = StandardScaler()
    agent_scaled = scaler_agent.fit_transform(agent_proportions.fillna(0))

    # Find optimal k for agents based on PC1 of their behavior
    pca_agent_1d = PCA(n_components=1)
    pc1_scores_agent = pca_agent_1d.fit_transform(agent_scaled)
    optimal_k_agent = find_optimal_k_programmatically(pc1_scores_agent)

    # Get clusters for the optimal k
    agent_clusters, agent_labels = get_strategy_clusters(agent_profiles.copy(), n_clusters=optimal_k_agent)

    # Get 2D PCA for plotting
    pca_agent_2d = PCA(n_components=2)
    agent_pca_coords = pca_agent_2d.fit_transform(agent_scaled)
    agent_plot_df = pd.DataFrame(agent_pca_coords, columns=['PC1', 'PC2'], index=agent_profiles.index)
    agent_plot_df = pd.merge(agent_plot_df.reset_index(), agent_clusters, on='prolific_id')

    # Plot Agent Data
    if optimal_k_agent == 1:
        sns.scatterplot(data=agent_plot_df, x='PC1', y='PC2', color='lightgreen', s=100, alpha=0.8, ax=ax2)
    else:
        sns.scatterplot(data=agent_plot_df, x='PC1', y='PC2', hue='Performance Group', hue_order=agent_labels,
                        palette='viridis', s=100, alpha=0.8, ax=ax2)
        ax2.legend(title='Agent Performance Group')

    pc1_var_a = pca_agent_2d.explained_variance_ratio_[0] * 100
    pc2_var_a = pca_agent_2d.explained_variance_ratio_[1] * 100
    ax2.set_title(f'Agent Behavioral Clusters (k={optimal_k_agent})', fontweight='bold')
    ax2.set_xlabel(f'Principal Component 1 ({pc1_var_a:.1f}%)')
    ax2.set_ylabel(f'Principal Component 2 ({pc2_var_a:.1f}%)')
    ax2.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    ax2.axhline(0, color='grey', linestyle='--', linewidth=0.5)

    # --- Set shared axes for direct comparison ---
    global_ymin = min(human_plot_df['PC2'].min(), agent_plot_df['PC2'].min())
    global_ymax = max(human_plot_df['PC2'].max(), agent_plot_df['PC2'].max())
    y_padding = (global_ymax - global_ymin) * 0.05
    ax1.set_ylim(global_ymin - y_padding, global_ymax + y_padding)
    ax2.set_ylim(global_ymin - y_padding, global_ymax + y_padding)

    global_xmin = min(human_plot_df['PC1'].min(), agent_plot_df['PC1'].min())
    global_xmax = max(human_plot_df['PC1'].max(), agent_plot_df['PC1'].max())
    x_padding = (global_xmax - global_xmin) * 0.05
    ax1.set_xlim(global_xmin - x_padding, global_xmax + x_padding)
    ax2.set_xlim(global_xmin - x_padding, global_xmax + x_padding)

    save_path = os.path.join(plot_dir, 'final_behavioral_pca_comparison.png')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(save_path)
    print(f"Saved final comparative behavioral PCA plot to '{save_path}'")

def justify_pca_usage(profiles_df: pd.DataFrame):
    """
    Performs and interprets statistical tests (Bartlett's, KMO) to
    justify the use of PCA on the behavioral profiles.
    """
    print("\n" + "#" * 60)
    print("###  Statistical Justification for Using PCA  ###")
    print("#" * 60)

    print("\nNote: The behavioral features exhibit multicollinearity because the five outcomes")
    print("for each attribute are not fully independent. This is expected. The tests below")
    print("are performed on a subset of features to ensure statistical validity.")

    # Drop one redundant column per attribute to resolve multicollinearity
    cols_to_drop = [col for col in profiles_df.columns if 'Invalid_Pattern_Err' in col]
    profiles_for_test = profiles_df.drop(columns=cols_to_drop)

    # Bartlett's Test of Sphericity
    chi_square_value, p_value = calculate_bartlett_sphericity(profiles_for_test)
    print("\n--- Bartlett's Test of Sphericity ---")
    print(f"  Chi-Squared Statistic: {chi_square_value:.2f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(
            "  --> Result is significant (p < 0.05). The variables are sufficiently inter-correlated; PCA is appropriate.")
    else:
        print("  --> Result is not significant. The variables may be too uncorrelated for PCA.")

    # Kaiser-Meyer-Olkin (KMO) Test
    kmo_per_variable, kmo_total = calculate_kmo(profiles_for_test)
    print("\n--- Kaiser-Meyer-Olkin (KMO) Test ---")
    print(f"  Overall KMO Statistic: {kmo_total:.2f}")
    if kmo_total >= 0.6:
        print("  --> Result is acceptable (KMO > 0.6). The data is suitable for PCA.")
    else:
        print("  --> Result is not acceptable (KMO < 0.6). The data may not be suitable for PCA.")


def main():
    """Main analysis script to find optimal clusters and generate all plots."""
    DATA_FILE_PATH = os.path.join(csv_dir, 'control_250716.csv')
    print(f"Loading human performance data from '{DATA_FILE_PATH}'...")
    try:
        global human_df
        human_df = pd.read_csv(DATA_FILE_PATH)
        human_df['block'] = (human_df['trial_num'] - 1) // 30 + 1
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file not found at '{DATA_FILE_PATH}'.")
        return

    print(f"Found data for {len(human_df['prolific_id'].unique())} unique participants.")

    print("\n" + "#" * 60)
    print("###  Primary Analysis: Overall Performance (All Blocks)  ###")
    print("#" * 60)

    profiles_df = get_participant_profiles(human_df)
    if profiles_df is None: return
    profiles_proportions = profiles_df.div(profiles_df.sum(axis=1), axis=0)
    scaler = StandardScaler()
    profiles_scaled = scaler.fit_transform(profiles_proportions.fillna(0))

    optimal_k_overall = find_optimal_k_programmatically(profiles_scaled)

    print(f"\n--- Generating All Plots for the Primary Analysis (k={optimal_k_overall}) ---")
    find_optimal_clusters_visual(profiles_scaled)
    analyze_error_strategies(profiles_df.copy(), n_clusters=optimal_k_overall)  # The main heatmap
    plot_clusters_on_pca_axes(human_df, profiles_df.copy(), n_clusters=optimal_k_overall)
    plot_pca_joint_distribution(human_df, profiles_df.copy(), n_clusters=optimal_k_overall)

    generate_block_by_block_heatmaps(human_df)

    # --- FINAL CORRELATIONAL ANALYSIS: Block 1 Learners vs. Survey Data ---
    # This is still a valuable analysis to run.
    run_full_analysis_for_blocks(human_df, blocks=[1], description="Training Block (Block 1) Only")

    print("\n\nAnalysis complete.")


if __name__ == "__main__":
    main()