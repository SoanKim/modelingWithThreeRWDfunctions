#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 10:03 on 25/7/25
# Title: preprocess.py
# Explanation: Loads the csv file of the human data and preprocess it for the simulation

import pandas as pd
import numpy as np
import itertools
from typing import Dict
import collections


def prepare_data(file_path: str, participant_index: int) -> Dict:
    #print("Loading and processing data...")
    try:
        df = pd.read_csv(file_path, index_col=0)
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file not found at '{file_path}'")
        return {}
    
    # Pick the first subject's data to provide the stimuli to the agent
    prolific_ids = df['prolific_id'].unique()
    if not len(prolific_ids):
        raise ValueError("No subjects found in the data file.")

    if participant_index >= len(prolific_ids):
        print(f"FATAL ERROR: participant_index {participant_index} is out of bounds. "
              f"There are only {len(prolific_ids)} participants.")
        return {}

    subj_df = df[df['prolific_id'] == prolific_ids[participant_index]].reset_index(drop=True)

    # Bookkeeping of the experimental dataset
    num_trials = len(subj_df)
    attributes = {
        "C": ["red", "yellow", "green"],
        "F": ["empty", "half", "full"],
        "S": ["circle", "triangle", "square"],
        "B": ["black", "grey", "white"]
    }
    attribute_names = list(attributes.keys())
    card_cols = ['one', 'two', 'three', 'four', 'five']
    
    # Digitizing the nominal cards
    digit_cards = np.zeros((num_trials, 5, 4), dtype=int)
    for i in range(num_trials):
        for card_idx, card_col_name in enumerate(card_cols):
            card_str_features = subj_df.loc[i, card_col_name].rstrip(".png").split("_")
            for att_idx, att_name in enumerate(attribute_names):
                feature_str = card_str_features[att_idx]
                digit_cards[i, card_idx, att_idx] = attributes[att_name].index(feature_str)
                
    # Creating a relation matrix with the size of 10 (5 choose 3) x 4 attributes
    combinations_of_3 = list(itertools.combinations(range(5), r=3))
    relation_matrices = np.zeros((num_trials, 10, 4), dtype=int)
    for i in range(num_trials):
        for att_idx in range(len(attribute_names)):
            for combo_idx, combo in enumerate(combinations_of_3):
                three_features = digit_cards[i, list(combo), att_idx]
                relation_matrices[i, combo_idx, att_idx] = len(np.unique(three_features))
                
    # Creating a projective triplets' matrix
    projection_matrices = np.zeros((num_trials, 4, 4))
    card_availability = []

    for i in range(num_trials):
        trial_rel_matrix = relation_matrices[i, :, :]
        atts_for_this_trial = []
        for att_idx in range(len(attribute_names)):
            rels_for_att = trial_rel_matrix[:, att_idx]
            bins = {1: [], 2: [], 3: []}
            for choice_idx, rel in enumerate(rels_for_att):
                bins[rel].append(choice_idx)
            atts_for_this_trial.append([bins[1], bins[2], bins[3]])
            projection_matrices[i, att_idx, 1] = len(bins[1])
            projection_matrices[i, att_idx, 2] = len(bins[2])
            projection_matrices[i, att_idx, 3] = len(bins[3])
        projection_matrices[i, :, 0] = np.sum(projection_matrices[i, :, 1:], axis=1)
        card_availability.append(atts_for_this_trial)

    # Correct answers from the data files
    card_name_combos = sorted([''.join(c) for c in itertools.combinations('12345', 3)])
    true_answers = np.array([
        card_name_combos.index(str(int(ans))) for ans in subj_df['correct_response']
    ])

    subCombi_matrices = np.zeros((num_trials, 4, 3), dtype=int)
    for i in range(num_trials):
        for att_idx in range(len(attribute_names)):
            cnts = collections.Counter(relation_matrices[i, :, att_idx])
            subCombi_matrices[i, att_idx, 0] = cnts.get(1, 0)
            subCombi_matrices[i, att_idx, 1] = cnts.get(2, 0)
            subCombi_matrices[i, att_idx, 2] = cnts.get(3, 0)

    return {
        "relation_matrices": relation_matrices,
        "true_answers": true_answers,
        "attributes": attribute_names,
        "num_trials": num_trials,
        "digit_cards": digit_cards,
        "projection_matrices": projection_matrices,
        "card_availability": card_availability,
        "subCombi_matrices": subCombi_matrices,
        "full_human_df": df
    }


if __name__ == "__main__":
    pass