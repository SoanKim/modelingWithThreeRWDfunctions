#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:41 on 28/8/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from __future__ import annotations
import numpy as np
import pandas as pd
from environment import Environment
from typing import List, Set, Optional


class Agent:
    def __init__(self, attributes: List[str], lr_same: float, lr_diff: float, selection_temperature: float):
        self.attributes = attributes
        self.lr_same = lr_same
        self.lr_diff = lr_diff
        self.selection_temperature = selection_temperature

        self.v_relation = pd.DataFrame(np.zeros((len(attributes), 3)), index=attributes, columns=[1, 2, 3])
        self.v_attribute = pd.Series([0.25] * len(attributes), index=attributes)

    def solve_one_trial(self, env: Environment) -> tuple[int, list]:
        """Solves a trial using only a simple heuristic search."""
        final_candidates, search_path_details = self.simulate_search_process(env)

        if len(final_candidates) == 1:
            final_choice = list(final_candidates)[0]
        else:
            final_scores = {idx: self.calculate_learned_h_score(env.get_triplet_relations(idx)) for idx in
                            final_candidates if env.get_triplet_relations(idx) is not None}
            if not final_scores:
                # If no valid candidates have scores, make a random choice from all possibilities
                return np.random.choice(list(range(10))), search_path_details

            scores_array = np.array(list(final_scores.values()))
            if self.selection_temperature > 0:
                exp_scores = np.exp(scores_array / self.selection_temperature)
                probabilities = exp_scores / np.sum(exp_scores)
                if np.isnan(probabilities).any(): probabilities = None
            else:
                probabilities = None

            final_choice = np.random.choice(list(final_scores.keys()), p=probabilities)

        return final_candidates, search_path_details, final_choice

    def simulate_search_process(self, env: Environment):
        """A simple, greedy, model-free search process."""
        valid_candidates = set(range(10))
        analyzed_atts = frozenset()
        search_path = []

        for _ in range(len(self.attributes)):
            if len(valid_candidates) <= 1:
                break

            available_attributes = list(frozenset(self.attributes) - analyzed_atts)
            if not available_attributes:
                break

            # --- HEURISTIC CHOICE: Simply pick the best-rated attribute ---
            relevant_v_attribute = self.v_attribute[available_attributes]
            chosen_attribute = relevant_v_attribute.idxmax()

            # This _analyze_and_partition function is now just a helper
            bins, chosen_bin_prop, reward = self._analyze_and_partition(env, chosen_attribute, valid_candidates)
            search_path.append({'attribute_chosen': chosen_attribute, 'reward': reward})
            analyzed_atts = analyzed_atts.union({chosen_attribute})

            if chosen_bin_prop:
                valid_candidates = set(bins[chosen_bin_prop])
            else:
                break

        return valid_candidates, search_path

    def learn_from_trial_outcome(self, final_choice_vector: List[int], true_reward: float):
        predicted_h = self.calculate_learned_h_score(final_choice_vector)
        err = true_reward - predicted_h
        for i, p in enumerate(final_choice_vector):
            relation = int(p)
            learning_rate = (self.lr_same + self.lr_diff) / 2
            if relation == 1:
                learning_rate = self.lr_same
            elif relation == 3:
                learning_rate = self.lr_diff

            update_value = learning_rate * err
            self.v_relation.loc[self.attributes[i], relation] += update_value

    def _analyze_and_partition(self, game: Environment, attribute: str, current_candidates: Set[int]) -> tuple[
        dict, int | None, float]:
        relations = game.get_relations_for_attribute(attribute)
        bins = {1: [], 2: [], 3: []}
        for cand_idx in current_candidates:
            bins[int(relations[cand_idx])].append(cand_idx)

        bin_scores = {}
        for prop_type, candidates_in_bin in bins.items():
            if not candidates_in_bin:
                continue
            tactical_score = 1.0 / len(candidates_in_bin)
            strategic_value = self.v_relation.loc[attribute, prop_type]
            combined_score = tactical_score + (1.0 * strategic_value)  # Using fixed weight of 1
            bin_scores[prop_type] = combined_score

        if not bin_scores:
            return bins, None, 0.0

        active_tau = self.selection_temperature
        if active_tau == 0:
            chosen_bin_prop = max(bin_scores, key=bin_scores.get)
        else:
            props = list(bin_scores.keys())
            scores = np.array(list(bin_scores.values()))
            exp_scores = np.exp(scores / active_tau)
            probabilities = exp_scores / np.sum(exp_scores)
            if np.isnan(probabilities).any():
                probabilities = None
            chosen_bin_prop = np.random.choice(props, p=probabilities)

        chosen_bin_candidates = bins.get(chosen_bin_prop, [])
        internal_reward = 1.0 / len(chosen_bin_candidates) if chosen_bin_candidates else 0.0
        return bins, chosen_bin_prop, internal_reward

    def learn_from_search_strategy(self, attribute: str, internal_reward: float):
        learning_rate = (self.lr_same + self.lr_diff) / 2
        err = internal_reward - self.v_attribute[attribute]
        self.v_attribute[attribute] += learning_rate * err

    def calculate_learned_h_score(self, relation_vector: List[int]) -> float:
        score = 0
        if relation_vector is not None:
            for i, p in enumerate(relation_vector):
                score += self.v_relation.loc[self.attributes[i], int(p)]
        return score

    def calculate_choice_log_likelihood(self, env: Environment, final_candidates: Set[int], human_choice: int,
                                        tau: float) -> float:
        floor_probability = 1e-9
        if human_choice not in final_candidates:
            return np.log(floor_probability)

        final_scores = {idx: self.calculate_learned_h_score(env.get_triplet_relations(idx)) for idx in final_candidates
                        if env.get_triplet_relations(idx) is not None}
        if not final_scores:
            return np.log(floor_probability)

        if tau == 0:
            best_choice = max(final_scores, key=final_scores.get)
            return np.log(1.0 - floor_probability) if human_choice == best_choice else np.log(floor_probability)

        scores = np.array(list(final_scores.values()), dtype=float)
        scores -= np.max(scores)
        exp_scores = np.exp(scores / tau)
        sum_exp_scores = np.sum(exp_scores)

        if sum_exp_scores == 0:
            return np.log(floor_probability)

        human_choice_score = final_scores.get(human_choice)
        if human_choice_score is None:
            return np.log(floor_probability)

        human_choice_exp_score = np.exp((human_choice_score - np.max(scores)) / tau)
        probability = human_choice_exp_score / sum_exp_scores
        return np.log(max(probability, floor_probability))