#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:54 on 28/8/25
# Title: metacontroller.py
# MODIFIED: To accept and apply memory decay (forgetting).

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Set, Tuple, Dict, Any

from agent_heuristic import Agent as HeuristicAgent
from agent_mcts import Agent as MctsAgent
from environment import Environment


class Metacontroller:
    """
    A high-level agent that manages and chooses between a fast, heuristic (MF) agent
    and a slow, deliberative (MB) agent to solve a task.
    """

    def __init__(self, attributes: List[str], cost_mcts: float, alpha_meta: float,
                 lr_same: float, lr_diff: float, selection_temperature: float,
                 mcts_time_budget_ms: int, ucb_exploration_constant: float,
                 strategic_influence: float,
                 decay_rate: float,
                 reward_strategy: str = 'candidate_reduction'):
        """
        Initializes the Metacontroller and its two sub-agents.
        """
        self.attributes = attributes

        self.alpha_meta = alpha_meta
        self.gamma_meta = 0.99
        self.epsilon_meta = 0.3  # FIXED exploration rate
        self.cost_heuristic = 0.01
        self.cost_mcts = cost_mcts
        self.decay_rate = decay_rate

        # --- Initialize the two "worker" agents ---
        self.mf_agent = HeuristicAgent(attributes=attributes, lr_same=lr_same, lr_diff=lr_diff,
                                       selection_temperature=selection_temperature)

        self.mb_agent = MctsAgent(attributes=attributes, lr_same=lr_same, lr_diff=lr_diff,
                                  selection_temperature=selection_temperature,
                                  strategic_influence=strategic_influence,
                                  mcts_time_budget_ms=mcts_time_budget_ms,
                                  ucb_exploration_constant=ucb_exploration_constant,
                                  reward_strategy=reward_strategy)

        # --- Shared knowledge base ---
        self.v_relation = pd.DataFrame(np.zeros((len(attributes), 3)), index=attributes, columns=[1, 2, 3])
        self.v_attribute = pd.Series([0.0] * len(attributes), index=attributes)
        self.mf_agent.v_relation = self.v_relation
        self.mb_agent.v_relation = self.v_relation
        self.mf_agent.v_attribute = self.v_attribute
        self.mb_agent.v_attribute = self.v_attribute

        # --- Metacontroller's Q-table ---
        self.q_meta = {}
        self.last_meta_state = None
        self.last_meta_action = None
        self.last_used_agent = None
        self.last_confidence = 0.0

        # --- CRITICAL: Make both agents share the SAME knowledge base ---
        # The Metacontroller owns the v_tables, and the agents just have references to them.
        self.v_relation = pd.DataFrame(np.zeros((len(attributes), 3)), index=attributes, columns=[1, 2, 3])
        self.v_attribute = pd.Series([0.0] * len(attributes), index=attributes)  # Start at 0
        self.mf_agent.v_relation = self.v_relation
        self.mb_agent.v_relation = self.v_relation
        self.mf_agent.v_attribute = self.v_attribute
        self.mb_agent.v_attribute = self.v_attribute

        # --- Initialize the Metacontroller's Q-table ---
        # States are discretized tuples: (trial_bin, confidence_bin, variance_bin)
        # Actions are 'MF' or 'MB'
        self.q_meta = {}
        self.last_meta_state = None
        self.last_meta_action = None
        self.last_used_agent = None

    def _get_meta_state(self, env: Environment, trial_idx: int) -> tuple:
        """
        Calculates and discretizes the current state for the meta-decision.
        """
        # 1. Trial number feature (early, mid, late)
        if trial_idx < 30:
            trial_bin = 0  # Reinforcement phase
        elif trial_idx < 60:
            trial_bin = 1  # Early test phase
        else:
            trial_bin = 2  # Late test phase

        # 2. MF Policy Confidence feature (value diff in v_attribute)
        if self.v_attribute.sum() == 0:
            mf_confidence_bin = 0  # Low confidence if no learning yet
            confidence = 0.0
        else:
            sorted_vals = self.v_attribute.sort_values(ascending=False)
            confidence = sorted_vals.iloc[0] - sorted_vals.iloc[1]
            mf_confidence_bin = 1 if confidence > 0.1 else 0

        self.last_confidence = confidence

        # 3. Task Difficulty feature (informativeness of attributes)
        # This checks for "magic bullet" or "useless attribute" problems.
        initial_counts = [len(env.card_availability[i][0]) + len(env.card_availability[i][2]) for i in
                          range(len(self.attributes))]
        variance = np.var(initial_counts)
        # Simple discretization: is the problem balanced or asymmetric?
        variance_bin = 1 if variance > 5 else 0

        return (trial_bin, mf_confidence_bin, variance_bin)

    def _choose_meta_action(self, meta_state: tuple) -> str:
        """
        Chooses between 'MF' and 'MB' using an epsilon-greedy policy.
        """
        # Ensure the state exists in the Q-table
        if meta_state not in self.q_meta:
            self.q_meta[meta_state] = {'MF': 0.0, 'MB': 0.0}

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon_meta:
            return np.random.choice(['MF', 'MB'])
        else:
            return max(self.q_meta[meta_state], key=self.q_meta[meta_state].get)

    def solve_one_trial(self, env: Environment, trial_idx: int) -> Tuple[Set[int], List[Dict[str, Any]], int]:
        """
        The main entry point for a single trial. It performs the meta-decision
        and delegates the problem-solving to the chosen sub-agent.
        """
        meta_state = self._get_meta_state(env, trial_idx)
        meta_action = self._choose_meta_action(meta_state)

        # This line creates the log message that the plotting script needs.
        #print(f"[DEBUG] Trial {trial_idx}: MetaState={meta_state}, ChoseSystem='{meta_action}'")

        self.last_meta_state = meta_state
        self.last_meta_action = meta_action

        # Delegate to the chosen "worker" agent
        if meta_action == 'MF':
            self.last_used_agent = self.mf_agent
            final_candidates, search_path, final_choice = self.mf_agent.solve_one_trial(env)
        else:  # 'MB'
            self.last_used_agent = self.mb_agent
            final_candidates, search_path, final_choice = self.mb_agent.solve_one_trial(env)

        return final_candidates, search_path, final_choice

    def learn_from_trial(self, env: Environment, human_choice: int, search_path: list, trial_idx: int):
        """
        Updates all models: applies decay, lets the sub-agent learn, and then
        updates the metacontroller's own Q-table.
        """
        # All learned values in the shared knowledge base decay slightly towards zero.
        # This happens *before* the new learning for the current trial occurs.
        if self.decay_rate > 0:
            self.v_relation *= (1 - self.decay_rate)
            self.v_attribute *= (1 - self.decay_rate)

        # --- 2. Sub-agent Learning ---
        if trial_idx < 30:
            final_choice_vector = env.get_triplet_relations(human_choice)
            if final_choice_vector is not None:
                true_reward = env.evaluate_final_choice(human_choice)
                self.last_used_agent.learn_from_trial_outcome(final_choice_vector, true_reward)

        for step in search_path:
            self.last_used_agent.learn_from_search_strategy(step['attribute_chosen'], step['reward'])

        # --- 3. Metacontroller Learning (Q-Learning Update) ---
        if self.last_meta_state is None:
            return

        r_env = env.evaluate_final_choice(human_choice)
        cost = self.cost_mcts if self.last_meta_action == 'MB' else self.cost_heuristic
        r_meta = r_env - cost

        S = self.last_meta_state
        A = self.last_meta_action

        S_prime = self._get_meta_state(env, trial_idx + 1)
        if S_prime not in self.q_meta:
            self.q_meta[S_prime] = {'MF': 0.0, 'MB': 0.0}
        max_q_prime = max(self.q_meta[S_prime].values())

        current_q = self.q_meta[S][A]
        td_target = r_meta + self.gamma_meta * max_q_prime
        td_error = td_target - current_q
        new_q = current_q + self.alpha_meta * td_error
        self.q_meta[S][A] = new_q

    def calculate_choice_log_likelihood(self, env: Environment, final_candidates: Set[int], human_choice: int) -> float:
        """
        Calculates the log-likelihood of the human's choice, delegating to the
        agent that was used for the trial.
        """
        if self.last_used_agent is None:
            return np.log(1e-9)  # Should not happen after the first trial

        # The sub-agent needs access to its own selection_temperature, which was passed at init
        tau = self.last_used_agent.selection_temperature
        return self.last_used_agent.calculate_choice_log_likelihood(env, final_candidates, human_choice, tau)
