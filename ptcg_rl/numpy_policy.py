"""
Torch-free greedy inference — replays PolicyValueNet forward pass in numpy.

Used by the Kaggle submission agent so the bundle doesn't need torch (~2GB).
Policy weights stored as a single .npz file (~2-8MB depending on model size).

Must stay in sync with model.py's architecture.
"""

from __future__ import annotations

import numpy as np

from .encoder import FastEncoder, CARD_DIM, STATE_FEAT_DIM, OPT_FEAT_DIM
from .encoder import N_CARDS, N_ATTACKS, N_OPT_TYPES, BOARD_SLOTS, MAX_HAND

EMB_CARD = 64
EMB_ATTACK = 32
EMB_OPT_TYPE = 16
OPT_ENC = 128
HIDDEN = 256

NEG_INF = -1e9


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _linear(w: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ w.T + b


class NumpyPolicy:
    """NumPy reimplementation of PolicyValueNet.forward()."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.encoder = FastEncoder()

    @classmethod
    def load(cls, path: str) -> "NumpyPolicy":
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def _pool(self, ids: np.ndarray) -> np.ndarray:
        """Mean-pool card embeddings. (K,) → (EMB_CARD,)."""
        e = self.w["card_emb.weight"][ids]
        mask = (ids > 0).astype(np.float32)[:, None]
        denom = mask.sum() + 1e-8
        return (e * mask).sum(axis=0) / denom

    def encode_state(self, board: np.ndarray, hand: np.ndarray,
                     feats: np.ndarray) -> np.ndarray:
        """(12,), (MAX_HAND,), (FEAT,) → (HIDDEN,)."""
        my = self._pool(board[:6])
        opp = self._pool(board[6:])
        hnd = self._pool(hand)
        x = np.concatenate([my, opp, hnd, feats])
        x = _relu(_linear(self.w["state_fc1.weight"], self.w["state_fc1.bias"], x))
        return _relu(_linear(self.w["state_fc2.weight"], self.w["state_fc2.bias"], x))

    def value(self, h: np.ndarray) -> float:
        v = _relu(_linear(self.w["value_fc1.weight"], self.w["value_fc1.bias"], h))
        return float(np.tanh(_linear(self.w["value_fc2.weight"], self.w["value_fc2.bias"], v)))

    def select(self, obs_dict: dict, greedy: bool = True) -> list[int]:
        """Full decision: obs dict → action indices."""
        sel = obs_dict.get("select")
        if sel is None:
            raise ValueError("deck selection — return deck directly")

        d = self.encoder.encode(obs_dict)
        n = len(d.opt_type)

        # State
        h = self.encode_state(d.board_cards, d.hand_cards, d.state_feats)

        # Options
        opt_card = self.w["card_emb.weight"][d.opt_type]
        opt_card_emb = self.w["card_emb.weight"][d.opt_card]
        opt_card2_emb = self.w["card_emb.weight"][d.opt_card2]
        opt_attack_emb = self.w["attack_emb.weight"][d.opt_attack]
        opt_type_emb = self.w["opt_type_emb.weight"][d.opt_type]

        opt_x = np.concatenate([opt_card_emb, opt_card2_emb, opt_attack_emb,
                                opt_type_emb, d.opt_feats], axis=-1)
        opts = _relu(_linear(self.w["opt_fc.weight"], self.w["opt_fc.bias"], opt_x))

        # Sequential selection
        picks = []
        picked_sum = np.zeros(OPT_ENC, dtype=np.float32)
        available = np.ones(n + 1, dtype=bool)

        while len(picks) < d.max_count:
            available[n] = len(picks) >= d.min_count
            stop = self.w["stop_vec"]
            rows = np.concatenate([opts, stop[np.newaxis, :]], axis=0)
            hx = np.broadcast_to(h, (n + 1, HIDDEN))
            px = np.broadcast_to(picked_sum, (n + 1, OPT_ENC))
            score_x = np.concatenate([hx, rows, px], axis=-1)
            logits = _linear(self.w["score_fc2.weight"], self.w["score_fc2.bias"],
                           _relu(_linear(self.w["score_fc1.weight"],
                                        self.w["score_fc1.bias"], score_x)))
            logits = np.where(available, logits, NEG_INF)
            logits = logits - np.max(logits)  # stabilize softmax
            probs = np.exp(logits) / np.exp(logits).sum()

            if greedy:
                idx = int(np.argmax(probs))
            else:
                idx = int(np.random.choice(n + 1, p=probs))

            if idx == n or idx >= n:  # STOP
                break
            picks.append(int(idx))
            picked_sum += opts[int(idx)]
            available[int(idx)] = False

        return picks[:d.max_count]
