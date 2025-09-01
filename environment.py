#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 10:25 on 25/7/25
# Title: environment.py
# Explanation: A single trial of the problem (90 in total)

import numpy as np
from typing import List, Optional


class Environment:
    def __init__(self, card_availability: List[List[List[int]]], relation_matrix: np.ndarray, true_set_index: int,
                 attributes: List[str]):
        self.card_availability = card_availability
        self.relation_matrix = relation_matrix
        self.true_set_index = true_set_index
        self.attributes = attributes

    # An example of get_card_availability:
    # [[7], [0, 1, 2, 3, 4, 5, 6, 8, 9], []]
    # [[], [0, 1, 2, 3, 6, 9], [4, 5, 7, 8]]
    # [[0, 2, 4, 7], [1, 3, 5, 6, 8, 9], []]
    # [[1], [0, 2, 3, 5, 6, 8], [4, 7, 9]]
    # <class 'list'>
    def get_card_availability(self, attribute_to_analyze: str, relation: int) -> list:
        att_idx = self.attributes.index(attribute_to_analyze)
        return self.card_availability[att_idx][relation - 1]

    # Returns the array of 10 relations for the attribute.
    def get_relations_for_attribute(self, attribute_to_analyze: str) -> np.ndarray:
        att_idx = self.attributes.index(attribute_to_analyze)
        return self.relation_matrix[:, att_idx]

    # Returns the full relation vector [relC, relF, relS, relB] for a given triplet index.
    def get_triplet_relations(self, choice_index: int) -> Optional[np.ndarray]:
        if 0 <= choice_index < len(self.relation_matrix):
            return self.relation_matrix[choice_index, :]
        return None

    # Compares the final choice with the ground truth.
    def evaluate_final_choice(self, choice_index: int) -> int:
        if choice_index == -1: return 0
        return 1 if choice_index == self.true_set_index else 0
