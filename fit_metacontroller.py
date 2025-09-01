#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:41 on 28/8/25
# Title: fit_metacontroller.py
# Explanation: Infers parameters for the dynamic Metacontroller model.

import numpy as np
import pandas as pd
import itertools
import argparse
import os
import time
import multiprocessing
import sys

from metacontroller import Metacontroller
from environment import Environment
from preprocess import prepare_data
from config import PARAM_GRID, FIXED_PARAMS


def fit_one_participant(participant_info: tuple):
    """
    Finds the best parameters for a single participant using a simple grid search.
    """
    participant_index, participant_id, input_file, human_df, log_path, reward_strategy = participant_info

    original_stdout = sys.stdout
    with open(log_path, 'a') as log_file:
        sys.stdout = log_file
        print(
            f"\n--- Starting fit for participant {participant_index + 1}/{len(human_df['prolific_id'].unique())} ({participant_id}) with strategy: {reward_strategy} ---")

        processed_data = prepare_data(input_file, participant_index=participant_index)
        card_name_combos = sorted([''.join(c) for c in itertools.combinations('12345', 3)])

        def response_to_index(resp):
            try:
                resp_str = str(resp).replace('.0', '')
                return card_name_combos.index(resp_str)
            except (ValueError, TypeError):
                return -1

        participant_data = human_df[human_df['prolific_id'] == participant_id].copy()
        participant_data['human_choice_idx'] = participant_data['concatenated'].apply(response_to_index)
        human_choices = participant_data.set_index('trial_num')['human_choice_idx'].to_dict()

        keys, values = zip(*PARAM_GRID.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        best_log_likelihood = -np.inf
        best_params = None

        for params in param_combinations:
            agent_params = {**params, **FIXED_PARAMS}
            agent = Metacontroller(
                **agent_params,
                reward_strategy=reward_strategy
            )
            total_log_likelihood = 0
            for trial_idx in range(90):
                env = Environment(
                    card_availability=processed_data["card_availability"][trial_idx],
                    relation_matrix=processed_data["relation_matrices"][trial_idx],
                    true_set_index=processed_data["true_answers"][trial_idx],
                    attributes=["C", "F", "S", "B"]
                )
                final_candidates, search_path, _ = agent.solve_one_trial(env, trial_idx)
                human_choice = human_choices.get(trial_idx + 1, -1)
                if human_choice == -1:
                    continue
                log_likelihood = agent.calculate_choice_log_likelihood(env, final_candidates, human_choice)
                if log_likelihood is not None:
                    total_log_likelihood += log_likelihood
                agent.learn_from_trial(env, human_choice, search_path, trial_idx)
            if total_log_likelihood > best_log_likelihood:
                best_log_likelihood = total_log_likelihood
                best_params = params

        if best_params is None:
            best_params = param_combinations[0]

        best_params['prolific_id'] = participant_id
        best_params['log_likelihood'] = best_log_likelihood
        print(f"--- Finished fit for participant {participant_id}. Best LL: {best_params['log_likelihood']:.2f} ---")

        sys.stdout = original_stdout
    return best_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit Metacontroller parameters.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input human data CSV file.")
    parser.add_argument('--reward_strategy', type=str, required=True,
                        choices=['candidate_reduction', 'value_of_information', 'belief_gated_reduction'],
                        help="The MCTS reward strategy to use for fitting.")
    parser.add_argument('--output_dir', type=str, default='csv', help="Directory to save the output parameter file.")
    parser.add_argument('--cores', type=int, default=-1, help="Number of CPU cores to use. -1 means use all available.")
    args = parser.parse_args()

    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    log_path = os.path.join(args.output_dir, f"{base_filename}_debug_log.txt")
    if os.path.exists(log_path):
        os.remove(log_path)
    print(
        f"--- Starting PARALLEL Metacontroller Fitting. Strategy: '{args.reward_strategy}'. Debug output will be saved to '{log_path}' ---")

    start_time = time.time()
    human_df = pd.read_csv(args.input)
    unique_ids = human_df['prolific_id'].unique()

    participant_args = [(i, pid, args.input, human_df, log_path, args.reward_strategy) for i, pid in
                        enumerate(unique_ids)]

    num_cores = multiprocessing.cpu_count() if args.cores == -1 else args.cores
    print(f"Setting up multiprocessing pool with {num_cores} cores.")

    all_final_params = []
    with multiprocessing.Pool(processes=num_cores) as pool:
        total_participants = len(unique_ids)
        results_iterator = pool.imap_unordered(fit_one_participant, participant_args)

        for i, result in enumerate(results_iterator, 1):
            all_final_params.append(result)
            progress_percent = (i / total_participants) * 100
            sys.stdout.write(f"\rProgress: {i}/{total_participants} participants fitted ({progress_percent:.1f}%)")
            sys.stdout.flush()
    print()

    final_params_df = pd.DataFrame(all_final_params)
    final_save_path = os.path.join(args.output_dir, f"{base_filename}_inferred_params_metacontroller.csv")
    final_params_df.to_csv(final_save_path, index=False)

    total_time = time.time() - start_time
    print(f"\n--- Full METACONTROLLER parameter fitting complete in {total_time / 60:.2f} minutes. ---")
    print(f"Final results for {len(unique_ids)} participants saved to {final_save_path}")