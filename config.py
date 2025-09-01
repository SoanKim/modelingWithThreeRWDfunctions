#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:14 on 29/8/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

import numpy as np

# ==============================================================================
# 1. FIXED MODEL PARAMETERS
# These are the architectural choices for the model that are NOT tuned in the
# grid search. Both the fitting script and the main simulation script will
# import these to ensure consistency.
# ==============================================================================
FIXED_PARAMS = {
    'attributes': ["C", "F", "S", "B"],
    'mcts_time_budget_ms': 1,
    'ucb_exploration_constant': np.sqrt(2),
    'strategic_influence': 0.5,  # Required by MctsAgent
}

# ==============================================================================
# 2. PARAMETER GRID FOR FITTING
# This dictionary defines the search space for the `fit_metacontroller.py`
# script. To experiment with different parameters, you only need to change them
# here.
# ==============================================================================
PARAM_GRID = {
    'cost_mcts': [0.01, 0.03, 0.05],
    'alpha_meta': [0.05, 0.15],
    'lr_same': [0.05, 0.2],
    'lr_diff': [0.05, 0.2],
    'selection_temperature': [0.1, 0.3, 0.6, 0.9],
    'decay_rate': [0.0, 0.01, 0.05]
}