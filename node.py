#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 10:12 on 25/7/25
# Title: node.py
# Explanation: Creates MCTS nodes for a tree search

from __future__ import annotations
from typing import Dict, Optional, FrozenSet
import math


class Node:
    def __init__(self, state: FrozenSet[str], parent: Optional[Node] = None, action: Optional[str] = None):
        self.state: FrozenSet[str] = state
        self.parent: Optional[Node] = parent
        self.action: Optional[str] = action
        self.children: Dict[str, Node] = {}
        self.N: int = 0
        self.Q: float = 0.0

    # The average reward (Q/N) for this node
    def get_value(self) -> float:
        if self.N == 0:
            return 0.0
        return self.Q / self.N

    def ucb1(self, C: float = math.sqrt(2)) -> float:
        if self.N == 0:
            return float('inf')  # Exploring unvisited nodes first
        if self.parent is None or self.parent.N == 0:
            return self.get_value()  # Should not happen during normal selection from non-root
        return self.get_value() + C * math.sqrt(math.log(self.parent.N) / self.N)

    # A string representation for debugging
    def __repr__(self) -> str:
        state_str = ", ".join(sorted(list(self.state))) if self.state else "Root"
        return (f"Node(state={{{state_str}}}, "
                f"value={self.get_value():.2f}, Q={self.Q:.2f}, N={self.N})")