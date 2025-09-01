#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 10:31 on 25/7/25
# Title: mcts.py
# MODIFIED: To be a generic search tool that accepts a reward function.

from __future__ import annotations
import time
import random
import math
from typing import List, Set, FrozenSet, Tuple, Callable, Dict, Any
from node import Node
from environment import Environment


def default_reward_calculator(initial_candidates: Set[int], final_candidates: Set[int], **kwargs) -> float:
    """
    Calculates the normalized reward based on the proportion of candidates eliminated.
    This is the simple, tactical 'Candidate Reduction' strategy.
    """
    if not initial_candidates:
        return 0.0

    reduction = len(initial_candidates) - len(final_candidates)
    return float(reduction) / len(initial_candidates)


class MCTS:
    def __init__(self, attributes: List[str], ucb_exploration_constant: float = math.sqrt(2)):
        self.attributes = attributes
        self.ucb_exploration_constant = ucb_exploration_constant

    # The _get_tree_stats, _selection, _expansion, and _backpropagate methods are unchanged.
    def _get_tree_stats(self, node: Node) -> Tuple[int, int]:
        if not node.children:
            return 1, 1
        depth = 0
        num_nodes = 1
        for child in node.children.values():
            child_depth, child_nodes = self._get_tree_stats(child)
            depth = max(depth, child_depth)
            num_nodes += child_nodes
        return depth + 1, num_nodes

    def _selection(self, node: Node) -> Node:
        current_node = node
        while current_node.children:
            unvisited_children = [child for child in current_node.children.values() if child.N == 0]
            if unvisited_children:
                return random.choice(unvisited_children)
            current_node = max(current_node.children.values(), key=lambda n: n.ucb1(self.ucb_exploration_constant))
        return current_node

    def _expansion(self, node: Node) -> Node:
        analyzed_atts = node.state
        available_atts = [att for att in self.attributes if att not in analyzed_atts]
        if available_atts:
            chosen_att_to_expand = random.choice(available_atts)
            next_state = node.state.union({chosen_att_to_expand})
            new_child = Node(state=next_state, parent=node, action=chosen_att_to_expand)
            node.children[chosen_att_to_expand] = new_child
            return new_child
        return node

    def _backpropagate(self, node: Node, reward: float):
        temp_node = node
        while temp_node is not None:
            temp_node.N += 1
            temp_node.Q += reward
            temp_node = temp_node.parent

    def _rollout(self, node: Node, initial_candidates: Set[int], env: Environment, reward_function: Callable) -> float:
        """
        Performs a simulation and then calls the provided function to score the outcome.
        """
        # The simulation part: find the set of final candidates for this path
        if not node.state:
            return 0.0

        final_candidates = set(initial_candidates)
        for att_to_analyze in node.state:
            if len(final_candidates) <= 1:
                break

            relations = env.get_relations_for_attribute(att_to_analyze)
            bins = {1: [], 2: [], 3: []}
            for cand_idx in final_candidates:
                bins[int(relations[cand_idx])].append(cand_idx)

            # Use a simple, greedy policy for the rollout simulation
            best_prop = min((p for p, c in bins.items() if c), key=lambda p: len(bins[p]), default=None)
            if best_prop:
                final_candidates = set(bins[best_prop])

        # The reward calculation is now delegated to the passed-in function
        reward = reward_function(
            initial_candidates=initial_candidates,
            final_candidates=final_candidates,
            path=node.state,
            env=env
        )
        return float(reward)

    def run_search(self, time_budget_ms: int, current_analyzed_atts: FrozenSet[str],
                   current_valid_candidates: Set[int], env: Environment,
                   reward_function: Callable = default_reward_calculator, **kwargs) -> tuple[str | None, dict]:

        mcts_root = Node(state=current_analyzed_atts)
        start_search_time = time.perf_counter()
        time_budget_sec = time_budget_ms / 1000.0

        while (time.perf_counter() - start_search_time) < time_budget_sec:
            node = self._selection(mcts_root)

            if node.N > 0 and len(node.state) < len(self.attributes):
                node = self._expansion(node)

            rollout_reward = self._rollout(node, current_valid_candidates, env, reward_function)

            self._backpropagate(node, rollout_reward)

        if not mcts_root.children:
            return None, {}

        best_child = max(mcts_root.children.values(), key=lambda n: n.N)

        return best_child.action, {}