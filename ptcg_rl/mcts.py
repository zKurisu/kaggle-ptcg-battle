"""
Batched MCTS search using engine's search_begin/search_step API.

Integrates with numpy_policy for Kaggle submission (no torch needed)
and with model.py for training-time expert iteration.

Algorithm: PUCT with batched virtual-loss leaf evaluation.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np

# ── PUCT constants ────────────────────────────────────────────────────────
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.25
DIRICHLET_EPS = 0.25
MAX_SIMS = 64           # default sims per decision
LEAF_BATCH = 16         # batch leaf evaluations


class MCTSNode:
    __slots__ = ('parent', 'children', 'visit_count', 'total_value',
                 'prior', 'search_id', 'select_list', 'value')

    def __init__(self, parent: MCTSNode | None, search_id: int | None,
                 select_list: list[int] | None, prior: float = 0.0):
        self.parent = parent
        self.children: list[MCTSNode | None] = []  # None = not yet expanded
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.search_id = search_id
        self.select_list = select_list or []
        self.value = 0.0  # leaf evaluation


def _value_fn(numpy_policy, observation: dict) -> float:
    """Evaluate state value using the numpy policy network. Higher = better for current player."""
    try:
        d = numpy_policy.encoder.encode(observation)
        h = numpy_policy.encode_state(d.board_cards, d.hand_cards, d.state_feats)
        return float(numpy_policy.value(h))
    except Exception:
        return 0.0


def mcts_search(obs_dict: dict, numpy_policy,
                deck: list[int], sims: int = MAX_SIMS,
                time_budget: float = 5.0) -> list[int]:
    """
    Run MCTS search and return the best action (option indices).

    Uses the engine's search API for forward simulation and the numpy
    policy network for leaf evaluation (V(s) prediction).

    Args:
        obs_dict: current observation
        numpy_policy: NumpyPolicy instance for leaf evaluation
        deck: our 60-card deck list
        sims: max simulations
        time_budget: max wall-clock seconds

    Returns:
        list of selected option indices
    """
    from cg.api import (
        to_observation_class, search_begin, search_step,
        search_end, search_release,
    )

    obs = to_observation_class(obs_dict)
    state = obs.current
    you = state.yourIndex
    my_state = state.players[you]
    op_state = state.players[1 - you]

    # Build predictions for hidden information
    my_deck_ct = my_state.deckCount
    my_prize_ct = len(my_state.prize)
    op_deck_ct = op_state.deckCount
    op_prize_ct = len(op_state.prize)
    op_hand_ct = op_state.handCount

    # Simple predictions (can be improved with deck tracking)
    your_deck = list(deck)[:max(1, my_deck_ct)]
    if len(your_deck) < my_deck_ct:
        your_deck = random.sample(deck * ((my_deck_ct // len(deck)) + 1), my_deck_ct)
    your_prize = random.sample(deck, max(1, my_prize_ct)) if my_prize_ct > 0 else []
    opp_deck = your_deck[:max(1, op_deck_ct)]  # guess similar deck
    opp_prize = your_deck[:max(1, op_prize_ct)] if op_prize_ct > 0 else []
    opp_hand = your_deck[:max(1, op_hand_ct)] if op_hand_ct > 0 else [1]
    opp_active = []
    if op_state.active and op_state.active[0] is None:
        opp_active = [deck[0]]

    # Start search
    try:
        search_state = search_begin(
            obs, your_deck=your_deck, your_prize=your_prize,
            opponent_deck=opp_deck, opponent_prize=opp_prize,
            opponent_hand=opp_hand, opponent_active=opp_active,
        )
    except Exception:
        return []  # search unavailable, fallback to greedy

    # Build root
    root = MCTSNode(None, search_state.searchId, [])
    root_obs = search_state.observation
    root_sel = root_obs.select
    if root_sel is None:
        search_end()
        return []

    n_opts = len(root_sel.option)
    max_count = root_sel.maxCount
    min_count = root_sel.minCount

    # Create child nodes for all legal single actions + STOP
    for i in range(n_opts + 1):  # +1 for STOP
        if i == n_opts:  # STOP — always a candidate
            node = MCTSNode(root, None, [i], 0.1)
        else:
            node = MCTSNode(root, None, [i], 1.0 / max(1, n_opts))
        root.children.append(node)

    # Root Dirichlet noise for exploration
    if n_opts > 1:
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.children))
        for i, child in enumerate(root.children):
            child.prior = (1 - DIRICHLET_EPS) * child.prior + DIRICHLET_EPS * noise[i]

    t0 = time.time()
    sim_count = 0

    while sim_count < sims and time.time() - t0 < time_budget:
        # Selection: traverse to a leaf
        current = root
        while True:
            # Check if this node has unexpanded children
            unexpanded = [c for c in current.children if c is not None and c.visit_count == 0]
            if unexpanded:
                # Expand one unexpanded child
                child = unexpanded[0]
                try:
                    step_result = search_step(current.search_id, child.select_list)
                    child.search_id = step_result.searchId
                    # Evaluate leaf
                    leaf_obs = step_result.observation
                    leaf_cur = leaf_obs.current
                    if leaf_cur and leaf_cur.result != -1:
                        # Terminal: game over
                        result = leaf_cur.result
                        child.value = 1.0 if result == you else (-1.0 if result != 2 else 0.0)
                    else:
                        child.value = _value_fn(numpy_policy, leaf_obs.__dict__ if hasattr(leaf_obs, '__dict__') else {})
                except Exception:
                    child.value = 0.0

                # Backpropagate
                node_ptr = child
                while node_ptr is not None:
                    node_ptr.visit_count += 1
                    # Value is from the perspective of the player who made the decision
                    node_ptr.total_value += child.value
                    # Flip perspective up the tree
                    child.value = -child.value
                    node_ptr = node_ptr.parent

                break

            # All children expanded — select via PUCT
            best_score = -1e9
            best_child = None
            for child in current.children:
                if child is None:
                    continue
                q = child.total_value / max(1, child.visit_count)
                # Flip Q if opponent's turn at this node
                cur_obs = None
                try:
                    cur_obs = search_state.observation if current is root else None
                except Exception:
                    pass
                u = C_PUCT * child.prior * math.sqrt(current.visit_count) / (1 + child.visit_count)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_child = child

            if best_child is None or best_child.visit_count > 1000:
                break
            current = best_child

        sim_count += 1

    search_end()

    # Select best action: highest visit count
    best = max(root.children, key=lambda c: (c.visit_count if c else -1))
    if best is None or best.select_list == [n_opts]:
        return []  # STOP selected

    return best.select_list[:max_count]
