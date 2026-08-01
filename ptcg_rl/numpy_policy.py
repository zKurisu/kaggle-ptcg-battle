"""
Torch-free inference — replays PolicyValueNet forward pass in numpy.
Includes MCTS search mode for stronger decision-making.

Usage:
    policy = NumpyPolicy.load("policy.npz")
    picks = policy.select(obs_dict)          # greedy
    picks = policy.select_mcts(obs_dict, deck)  # MCTS search
"""

from __future__ import annotations

import math
import random
import time
import numpy as np

from .encoder import FastEncoder, STATE_FEAT_DIM, OPT_FEAT_DIM, MAX_HAND

EMB_CARD = 64
OPT_ENC = 128
HIDDEN = 256
NEG_INF = -1e9

# MCTS settings
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.25
DIRICHLET_EPS = 0.25


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

def _linear(w: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ w.T + b


class NumpyPolicy:
    """NumPy PolicyValueNet + optional MCTS search."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.encoder = FastEncoder()

    @classmethod
    def load(cls, path: str) -> "NumpyPolicy":
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    # ── state / value ───────────────────────────────────────────────

    def _pool(self, ids: np.ndarray) -> np.ndarray:
        e = self.w["card_emb.weight"][ids]
        mask = (ids > 0).astype(np.float32)[:, None]
        return (e * mask).sum(axis=0) / (mask.sum() + 1e-8)

    def encode_state(self, board: np.ndarray, hand: np.ndarray,
                     feats: np.ndarray) -> np.ndarray:
        my = self._pool(board[:6])
        opp = self._pool(board[6:])
        hnd = self._pool(hand)
        x = np.concatenate([my, opp, hnd, feats])
        x = _relu(_linear(self.w["state_fc1.weight"], self.w["state_fc1.bias"], x))
        return _relu(_linear(self.w["state_fc2.weight"], self.w["state_fc2.bias"], x))

    def value(self, h: np.ndarray) -> float:
        v = _relu(_linear(self.w["value_fc1.weight"], self.w["value_fc1.bias"], h))
        return float(np.tanh(_linear(self.w["value_fc2.weight"], self.w["value_fc2.bias"], v)))

    def _evaluate_state(self, obs_dict: dict) -> float:
        """V(s) from a raw observation dict. Higher = better for current player."""
        try:
            d = self.encoder.encode(obs_dict)
            h = self.encode_state(d.board_cards, d.hand_cards, d.state_feats)
            return self.value(h)
        except Exception:
            return 0.0

    # ── greedy / sampling select ────────────────────────────────────

    def select(self, obs_dict: dict, greedy: bool = True) -> list[int]:
        """Greedy (or sampled) action selection without search."""
        sel = obs_dict.get("select")
        if sel is None:
            raise ValueError("deck selection — return deck directly")

        d = self.encoder.encode(obs_dict)
        n = len(d.opt_type)
        h = self.encode_state(d.board_cards, d.hand_cards, d.state_feats)

        opt_x = np.concatenate([
            self.w["card_emb.weight"][d.opt_card],
            self.w["card_emb.weight"][d.opt_card2],
            self.w["attack_emb.weight"][d.opt_attack],
            self.w["opt_type_emb.weight"][d.opt_type],
            d.opt_feats,
        ], axis=-1)
        opts = _relu(_linear(self.w["opt_fc.weight"], self.w["opt_fc.bias"], opt_x))

        picks, picked_sum = [], np.zeros(OPT_ENC, dtype=np.float32)
        avail = np.ones(n + 1, dtype=bool)

        while len(picks) < d.max_count:
            avail[n] = len(picks) >= d.min_count
            rows = np.concatenate([opts, self.w["stop_vec"][np.newaxis, :]], axis=0)
            hx = np.broadcast_to(h, (n + 1, HIDDEN))
            px = np.broadcast_to(picked_sum, (n + 1, OPT_ENC))
            score_x = np.concatenate([hx, rows, px], axis=-1)
            logits = _linear(self.w["score_fc2.weight"], self.w["score_fc2.bias"],
                           _relu(_linear(self.w["score_fc1.weight"],
                                        self.w["score_fc1.bias"], score_x)))
            logits = np.where(avail, logits, NEG_INF)
            logits = logits - logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()

            idx = int(np.argmax(probs)) if greedy else int(np.random.choice(n + 1, p=probs))
            if idx >= n:
                break
            picks.append(idx)
            picked_sum += opts[idx]
            avail[idx] = False

        return picks[:d.max_count]

    # ── MCTS search ─────────────────────────────────────────────────

    def select_mcts(self, obs_dict: dict, deck: list[int],
                    sims: int = 64, time_budget: float = 5.0) -> list[int]:
        """MCTS search using engine's search API + this policy as leaf evaluator.

        Requires the engine's search_begin/search_step/search_end to be importable.
        Falls back to greedy select() if search is unavailable.
        """
        try:
            from cg.api import to_observation_class, search_begin, search_step, search_end
            _SEARCH_OK = True
        except Exception:
            _SEARCH_OK = False

        if not _SEARCH_OK:
            return self.select(obs_dict, greedy=True)

        obs = to_observation_class(obs_dict)
        state = obs.current
        you = state.yourIndex
        my_s, op_s = state.players[you], state.players[1 - you]

        # Build hidden-info predictions
        mc = my_s.deckCount; pc = len(my_s.prize)
        oc = op_s.deckCount; opc = len(op_s.prize); ohc = op_s.handCount

        deck_pad = (deck * ((max(mc, oc, pc, opc, ohc) // len(deck)) + 2)) if deck else [1]

        try:
            ss = search_begin(
                obs,
                your_deck=deck_pad[:max(1, mc)],
                your_prize=deck_pad[:pc] if pc > 0 else [],
                opponent_deck=deck_pad[:max(1, oc)],
                opponent_prize=deck_pad[:opc] if opc > 0 else [],
                opponent_hand=deck_pad[:max(1, ohc)] if ohc > 0 else [1],
                opponent_active=[deck[0]] if (op_s.active and op_s.active[0] is None) else [],
            )
        except Exception:
            return self.select(obs_dict, greedy=True)

        root_sel = ss.observation.select
        if root_sel is None:
            search_end(); return []
        n = len(root_sel.option)
        mc_sel = root_sel.maxCount

        # Build root children
        children = []  # list of (select_list, search_id, visit, total, prior)
        for i in range(n + 1):
            prior = 0.05 if i == n else 1.0 / n
            children.append({"sel": [i], "sid": None, "visits": 0, "total": 0.0, "prior": prior})

        if n > 1:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(children))
            for i, c in enumerate(children):
                c["prior"] = (1 - DIRICHLET_EPS) * c["prior"] + DIRICHLET_EPS * noise[i]

        t0 = time.time()
        si = 0
        root_sid = ss.searchId

        while si < sims and time.time() - t0 < time_budget:
            # Selection: find leaf via PUCT
            path = [(-1, root_sid)]  # (child_idx, sid)
            cur_sid = root_sid
            cur_obs = ss.observation

            while True:
                # Get children of current node from children list
                # For root: use our children list. For deeper: expand on the fly.
                if len(path) == 1:
                    cands = children
                else:
                    # Deeper nodes: we don't have a proper tree, just step once
                    break

                # PUCT selection
                best_score, best_c = -1e9, None
                for ci, c in enumerate(cands):
                    q = c["total"] / max(1, c.get("visits", 0))
                    u = C_PUCT * c["prior"] * math.sqrt(max(1, si)) / (1 + c.get("visits", 0))
                    if q + u > best_score:
                        best_score, best_c = q + u, (ci, c)

                if best_c is None:
                    break

                ci, chosen = best_c

                # Expand if not visited
                if chosen.get("visits", 0) == 0:
                    # Expand: step into this child
                    try:
                        ar = search_step(cur_sid, chosen["sel"])
                        chosen["sid"] = ar.searchId
                        chosen["_obs"] = ar.observation
                        # Evaluate leaf
                        leaf = ar.observation
                        lc = leaf.current
                        if lc and lc.result is not None and lc.result != -1:
                            val = 1.0 if lc.result == you else (-1.0 if lc.result != 2 else 0.0)
                        else:
                            val = self._evaluate_state(leaf.__dict__ if hasattr(leaf, '__dict__') else {})
                        # Scale value to [-1, 1]
                        val = max(-1.0, min(1.0, val))
                    except Exception:
                        val = 0.0

                    # Backprop
                    chosen["visits"] = chosen.get("visits", 0) + 1
                    chosen["total"] = chosen.get("total", 0.0) + val
                    bp_val = -val
                    root_visits = 1
                    # Only backprop to root (shallow tree)
                    break
                else:
                    # Already visited — descend
                    path.append((ci, chosen.get("sid", cur_sid)))
                    cur_sid = chosen.get("sid", cur_sid)

            si += 1

        search_end()

        # Best action = highest visit count
        best = max(children, key=lambda c: c.get("visits", 0))
        if best is None or best["sel"] == [n]:  # STOP
            return []

        return best["sel"][:mc_sel]
