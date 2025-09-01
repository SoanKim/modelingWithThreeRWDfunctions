#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:59 on 28/8/25
# Title: main.py (Metacontroller Version)
# Explanation: Simulates Metacontroller agents with inferred human parameters,
#              logs their behavior, and runs comparative analyses.

from __future__ import annotations
import pandas as pd
import numpy as np
import os
import argparse

from metacontroller import Metacontroller
from environment import Environment
from preprocess import prepare_data
from config import FIXED_PARAMS, PARAM_GRID
from humanAnalysis import plot_final_behavioral_pca_comparison
from utils import plot_accuracy_distributions_standardized, plot_insight_comparison, print_results_summary


def run_agent_simulations(human_df: pd.DataFrame, human_data_file_path: str,
                          final_params_df: pd.DataFrame, reward_strategy: str):
    """
    Simulates Metacontroller agents using inferred parameters and returns a detailed,
    trial-by-trial DataFrame of their behavior for direct comparison with human data.
    """
    print("\n" + "#" * 60)
    print("### Running METACONTROLLER Simulations w/ Inferred Parameters ###")
    print("#" * 60)

    all_runs_results = []

    unique_prolific_ids = human_df['prolific_id'].unique()
    num_simulations = len(unique_prolific_ids)

    for sim_num in range(num_simulations):
        human_id = unique_prolific_ids[sim_num]

        try:
            human_params = final_params_df[final_params_df['prolific_id'] == human_id].iloc[0]
        except (KeyError, IndexError):
            print(f"Warning: Parameters for participant {human_id} not found. Skipping.")
            continue

        fitted_params = human_params.to_dict()
        model_params = {key: fitted_params[key] for key in PARAM_GRID.keys() if key in fitted_params}
        agent_params = {**model_params, **FIXED_PARAMS}

        agent = Metacontroller(
            **agent_params,
            reward_strategy=reward_strategy
        )

        processed_data = prepare_data(human_data_file_path, participant_index=sim_num)
        if not processed_data: continue

        for trial_idx in range(90):
            env = Environment(
                card_availability=processed_data["card_availability"][trial_idx],
                relation_matrix=processed_data["relation_matrices"][trial_idx],
                true_set_index=processed_data["true_answers"][trial_idx],
                attributes=FIXED_PARAMS['attributes']
            )

            final_candidates, search_path, agent_final_choice = agent.solve_one_trial(env, trial_idx)
            true_reward = env.evaluate_final_choice(agent_final_choice)

            log_entry = {
                "prolific_id": f"agent_{sim_num}",
                "trial_num": trial_idx + 1,
                "accuracy": true_reward,
                "chosen_system": agent.last_meta_action,
                "confidence": agent.last_confidence
            }

            attribute_names_lower = ['color', 'fill', 'shape', 'back']
            agent_choice_vector = env.get_triplet_relations(agent_final_choice)
            correct_choice_vector = env.get_triplet_relations(env.true_set_index)

            for i, attr_name in enumerate(attribute_names_lower):
                if agent_choice_vector is not None:
                    log_entry[f'choice_{attr_name}'] = agent_choice_vector[i]
                else:
                    log_entry[f'choice_{attr_name}'] = np.nan
                if correct_choice_vector is not None:
                    log_entry[f'modSum_{attr_name}'] = correct_choice_vector[i]
                else:
                    log_entry[f'modSum_{attr_name}'] = np.nan
            all_runs_results.append(log_entry)

            agent.learn_from_trial(env, agent_final_choice, search_path, trial_idx)

        if (sim_num + 1) % 10 == 0:
            print(f"Completed agent simulation run {sim_num + 1}/{num_simulations}")

    return pd.DataFrame(all_runs_results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Metacontroller simulations and analysis.")
    parser.add_argument('--human_data', type=str, required=True,
                        help="Path to the human data CSV file.")
    parser.add_argument('--params_data', type=str, required=True,
                        help="Path to the INFERRED agent parameters CSV file.")
    parser.add_argument('--reward_strategy', type=str, required=True,
                        choices=['candidate_reduction', 'value_of_information', 'belief_gated_reduction'],
                        help="The MCTS reward strategy to use for the simulation.")
    args = parser.parse_args()

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    print("--- Loading Data and Inferred Metacontroller Parameters ---")
    try:
        human_df = pd.read_csv(args.human_data)
        params_df = pd.read_csv(args.params_data)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: A required data file was not found. {e}")
        exit()

    print(f"Found data for {len(human_df['prolific_id'].unique())} unique participants.")

    agent_df = run_agent_simulations(human_df, args.human_data, params_df, args.reward_strategy)

    if not agent_df.empty:
        print("\n--- Generating final analysis plots and summary ---")

        # This generates accuracy_distributions_standardized.png
        plot_accuracy_distributions_standardized(human_df, agent_df)

        # This generates human_vs_agent_insight_plot.png
        plot_insight_comparison(human_df, agent_df)

        # This generates final_behavioral_pca_comparison.png
        plot_final_behavioral_pca_comparison(human_df, agent_df)

        # This prints the text summary to the console
        print_results_summary(human_df, agent_df)

        base_filename = os.path.splitext(os.path.basename(args.human_data))[0]
        agent_data_save_path = os.path.join('csv', f"{base_filename}_agent_simulation_data.csv")
        agent_df.to_csv(agent_data_save_path, index=False)
        print(f"\n--- Agent simulation data saved to '{agent_data_save_path}' ---")

    print("\n\nFull Metacontroller simulation and analysis pipeline complete.")