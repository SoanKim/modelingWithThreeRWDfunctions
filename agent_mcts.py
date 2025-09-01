#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:41 on 28/8/25
# MODIFIED: To use the generic MCTS tool and implement all three reward strategies.

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Set, FrozenSet, Any, Tuple, Dict, Callable

from mcts import MCTS, default_reward_calculator
from environment import Environment


class Agent:
    def __init__(self, attributes: List[str],
                 lr_same: float,
                 lr_diff: float,
                 selection_temperature: float,
                 strategic_influence: float,
                 mcts_time_budget_ms: int,
                 ucb_exploration_constant: float,
                 reward_strategy: str = 'candidate_reduction'):

        self.attributes = attributes
        self.lr_same = lr_same
        self.lr_diff = lr_diff
        self.selection_temperature = selection_temperature
        self.strategic_influence = strategic_influence
        self.mcts_time_budget_ms = mcts_time_budget_ms
        self.ucb_exploration_constant = ucb_exploration_constant
        self.reward_strategy = reward_strategy

        self.mcts_search = MCTS(attributes=self.attributes,
                                ucb_exploration_constant=self.ucb_exploration_constant)

        self.v_relation = pd.DataFrame(np.zeros((len(attributes), 3)),
                                       index=attributes, columns=[1, 2, 3])
        self.v_attribute = pd.Series([0.0] * len(attributes), index=self.attributes)
        self.last_confidence = 0.0

    def calculate_learned_h_score(self, relation_vector: list) -> float:
        if relation_vector is None: return 0.0
        h = 0.0
        for i, relation_type in enumerate(relation_vector):
            attribute_name = self.attributes[i]
            h += self.v_relation.loc[attribute_name, int(relation_type)]
        return h

    def _calculate_voi_reward(self, initial_candidates: Set[int], final_candidates: Set[int], **kwargs) -> float:
        env = kwargs.get('env')
        if not env: return 0.0
        h_initial = 0.0
        if initial_candidates:
            h_initial = max((self.calculate_learned_h_score(env.get_triplet_relations(c)) for c in initial_candidates),
                            default=0.0)
        h_final = 0.0
        if final_candidates:
            h_final = max((self.calculate_learned_h_score(env.get_triplet_relations(c)) for c in final_candidates),
                          default=0.0)
        raw_reward = h_final - h_initial
        return np.tanh(raw_reward)

    # --- NEW: The Agent's "Belief * Uncertainty Reduction" reward function ---
    def _calculate_belief_reward(self, initial_candidates: Set[int], final_candidates: Set[int], **kwargs) -> float:
        env = kwargs.get('env')
        path = kwargs.get('path')
        if not env or not path or not final_candidates:
            return 0.0

        if not initial_candidates:
            uncertainty_reduction = 0.0
        else:
            reduction = len(initial_candidates) - len(final_candidates)
            uncertainty_reduction = float(reduction) / len(initial_candidates)

        last_action_attribute = list(path)[-1]
        relations = env.get_relations_for_attribute(last_action_attribute)

        total_belief_score = 0.0
        for cand_idx in final_candidates:
            relation_type = int(relations[cand_idx])
            total_belief_score += self.v_relation.loc[last_action_attribute, relation_type]

        avg_belief = total_belief_score / len(final_candidates) if final_candidates else 0.0
        final_reward = avg_belief * (1 + uncertainty_reduction)
        return np.tanh(final_reward)

    # --- MODIFIED: Includes all three strategy options ---
    def _get_reward_function(self) -> Callable:
        if self.reward_strategy == 'value_of_information':
            return self._calculate_voi_reward
        elif self.reward_strategy == 'belief_gated_reduction':
            return self._calculate_belief_reward
        else:
            return default_reward_calculator

    # --- All other functions are complete ---
    def learn_from_trial_outcome(self, final_choice_vector: List[int], true_reward: float):
        predicted_h = self.calculate_learned_h_score(final_choice_vector)
        err = true_reward - predicted_h
        for i, p in enumerate(final_choice_vector):
            relation = int(p)
            learning_rate = self.lr_same if relation == 1 else self.lr_diff if relation == 3 else (
                                                                                                              self.lr_same + self.lr_diff) / 2
            self.v_relation.loc[self.attributes[i], relation] += learning_rate * err

    def learn_from_search_strategy(self, attribute: str, internal_reward: float):
        learning_rate = (self.lr_same + self.lr_diff) / 2
        err = internal_reward - self.v_attribute[attribute]
        self.v_attribute[attribute] += learning_rate * err

    def _analyze_and_partition(self, game: Environment, attribute: str, current_candidates: Set[int]) -> tuple[
        dict, int | None, float]:
        relations = game.get_relations_for_attribute(attribute)
        bins = {1: [], 2: [], 3: []}
        for cand_idx in current_candidates:
            bins[int(relations[cand_idx])].append(cand_idx)
        bin_scores = {}
        for prop_type, candidates_in_bin in bins.items():
            if not candidates_in_bin: continue
            tactical_score = 1.0 / len(candidates_in_bin)
            strategic_value = self.v_relation.loc[attribute, prop_type]
            combined_score = tactical_score + (self.strategic_influence * strategic_value)
            bin_scores[prop_type] = combined_score
        if not bin_scores:
            self.last_confidence = 0.0
            return bins, None, 0.0
        chosen_bin_prop = max(bin_scores, key=bin_scores.get)
        chosen_bin_candidates = bins.get(chosen_bin_prop, [])
        internal_reward = 1.0 / len(chosen_bin_candidates) if chosen_bin_candidates else 0.0
        return bins, chosen_bin_prop, internal_reward

    def calculate_choice_log_likelihood(self, env: Environment, final_candidates: Set[int], human_choice: int,
                                        tau: float) -> float:
        floor_probability = 1e-9
        if human_choice not in final_candidates: return np.log(floor_probability)
        final_scores = {idx: self.calculate_learned_h_score(env.get_triplet_relations(idx)) for idx in final_candidates
                        if env.get_triplet_relations(idx) is not None}
        if not final_scores: return np.log(1.0 / len(final_candidates)) if final_candidates else np.log(
            floor_probability)
        if tau == 0:
            best_choice = max(final_scores, key=final_scores.get)
            return np.log(1.0 - floor_probability) if human_choice == best_choice else np.log(floor_probability)
        scores = np.array(list(final_scores.values()), dtype=float)
        scores -= np.max(scores)
        exp_scores = np.exp(scores / tau)
        sum_exp_scores = np.sum(exp_scores)
        if sum_exp_scores == 0: return np.log(floor_probability)
        human_choice_score = final_scores.get(human_choice)
        if human_choice_score is None: return np.log(floor_probability)
        human_choice_exp_score = np.exp((human_choice_score - np.max(scores)) / tau)
        probability = human_choice_exp_score / sum_exp_scores
        return np.log(max(probability, floor_probability))

    def solve_one_trial(self, env: Environment) -> Tuple[Set[int], List[Dict[str, Any]], int]:
        analyzed_atts: FrozenSet[str] = frozenset()
        valid_candidates = set(range(10))
        search_path_details = []
        reward_function_to_use = self._get_reward_function()
        for i in range(len(self.attributes)):
            if len(valid_candidates) <= 1: break
            chosen_attribute, mcts_metrics = self.mcts_search.run_search(
                time_budget_ms=self.mcts_time_budget_ms,
                current_analyzed_atts=analyzed_atts,
                current_valid_candidates=valid_candidates,
                env=env,
                reward_function=reward_function_to_use
            )
            if chosen_attribute is None: break
            partition_result = self._analyze_and_partition(env, chosen_attribute, valid_candidates)
            if partition_result is None: break
            bins, chosen_bin_prop, reward = partition_result
            if reward > 0:
                search_steps_info = {'attribute_chosen': chosen_attribute, 'reward': reward}
                search_steps_info.update(mcts_metrics)
                search_path_details.append(search_steps_info)
            analyzed_atts = analyzed_atts.union({chosen_attribute})
            if chosen_bin_prop:
                valid_candidates = set(bins[chosen_bin_prop])
            else:
                break
        final_candidates = valid_candidates if valid_candidates else set(range(10))
        if len(final_candidates) == 1:
            final_choice = list(final_candidates)[0]
        else:
            final_scores = {idx: self.calculate_learned_h_score(env.get_triplet_relations(idx)) for idx in
                            final_candidates}
            if not final_scores:
                final_choice = np.random.choice(list(final_candidates))
            else:
                if self.selection_temperature == 0:
                    final_choice = max(final_scores, key=final_scores.get)
                else:
                    scores = np.array(list(final_scores.values()), dtype=float)
                    scores -= np.max(scores)
                    exp_scores = np.exp(scores / self.selection_temperature)
                    probabilities = exp_scores / np.sum(exp_scores)
                    if np.isnan(probabilities).any(): probabilities = None
                    final_choice = np.random.choice(list(final_scores.keys()), p=probabilities)
        return final_candidates, search_path_details, final_choice