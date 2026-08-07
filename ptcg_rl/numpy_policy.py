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

from .encoder import FastEncoder, MAX_HAND

NEG_INF = -1e9

# MCTS settings
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.25
DIRICHLET_EPS = 0.25


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

def _linear(w: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ w.T + b


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-12)


class NumpyPolicy:
    """NumPy PolicyValueNet + optional MCTS search."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.encoder = FastEncoder()
        self._arch = "cross_attn" if "state_token_fc.weight" in self.w else "pointer"
        # Auto-detect dimensions from weights
        self._oe = self.w['stop_vec'].shape[0]
        self._hd = (
            self.w["state_out_fc.bias"].shape[0]
            if self._arch == "cross_attn"
            else self.w["state_fc2.bias"].shape[0]
        )
        self._ec = self.w['card_emb.weight'].shape[1]
        self._has_option_context = "context_emb.weight" in self.w
        self._hierarchical_plan = "plan_condition_fc.weight" in self.w
        self._plan_dim = int(self.w["plan_fc2.bias"].shape[0]) if "plan_fc2.bias" in self.w else 0
        self._plan_cond_dim = (
            int(self.w["plan_condition_fc.bias"].shape[0])
            if self._hierarchical_plan
            else 0
        )
        self._history_k = int(self.w["history_pos_emb.weight"].shape[0]) if "history_pos_emb.weight" in self.w else 0
        self._history: list[dict[str, float | int]] = []
        if self._arch == "cross_attn":
            self._slot_state = True
            self._state_feat_dim = self.w["feat_token_fc.weight"].shape[1]
            self._state_layer_count = 0
            while f"state_layers.{self._state_layer_count}.q.weight" in self.w:
                self._state_layer_count += 1
        else:
            state_in = self.w["state_fc1.weight"].shape[1]
            slot_feat_dim = state_in - 5 * self._ec
            legacy_feat_dim = state_in - 3 * self._ec
            self._slot_state = 8 <= slot_feat_dim <= 256
            self._state_feat_dim = slot_feat_dim if self._slot_state else legacy_feat_dim
        opt_extra = 0
        if self._has_option_context:
            opt_extra = (
                self.w["context_emb.weight"].shape[1]
                + self.w["select_type_emb.weight"].shape[1]
                + self.w["area_emb.weight"].shape[1]
                + self.w["index_emb.weight"].shape[1]
                + self.w["inplay_area_emb.weight"].shape[1]
                + self.w["inplay_index_emb.weight"].shape[1]
            )
        self._opt_feat_dim = self.w["opt_fc.weight"].shape[1] - (
            2 * self._ec
            + self.w["attack_emb.weight"].shape[1]
            + self.w["opt_type_emb.weight"].shape[1]
            + opt_extra
        )

    def reset_history(self) -> None:
        self._history.clear()

    def _history_arrays(self) -> dict[str, np.ndarray] | None:
        if self._history_k <= 0:
            return None
        out = {
            "type": np.zeros(self._history_k, dtype=np.int64),
            "card": np.zeros(self._history_k, dtype=np.int64),
            "card2": np.zeros(self._history_k, dtype=np.int64),
            "attack": np.zeros(self._history_k, dtype=np.int64),
            "context": np.zeros(self._history_k, dtype=np.int64),
            "select_type": np.zeros(self._history_k, dtype=np.int64),
            "count": np.zeros(self._history_k, dtype=np.float32),
            "mask": np.zeros(self._history_k, dtype=np.float32),
        }
        events = self._history[-self._history_k:]
        start = self._history_k - len(events)
        for i, event in enumerate(events, start):
            for key in ("type", "card", "card2", "attack", "context", "select_type"):
                out[key][i] = int(event.get(key, 0))
            out["count"][i] = float(event.get("count", 0.0))
            out["mask"][i] = 1.0
        return out

    def _remember_decision(self, d, picks: list[int]) -> None:
        if self._history_k <= 0:
            return
        if picks:
            idx = int(picks[0])
            if 0 <= idx < len(d.opt_type):
                opt_feats = self._fit_feat_dim(d.opt_feats, self._opt_feat_dim)
                ctx = int(np.rint(opt_feats[idx, 3] * 64.0)) if opt_feats.shape[1] > 3 else 0
                sel_type = int(np.rint(opt_feats[idx, 4] * 16.0)) if opt_feats.shape[1] > 4 else 0
                event = {
                    "type": int(d.opt_type[idx]) + 1,
                    "card": int(d.opt_card[idx]),
                    "card2": int(d.opt_card2[idx]),
                    "attack": int(d.opt_attack[idx]),
                    "context": max(0, min(ctx, 64)) + 1,
                    "select_type": max(0, min(sel_type, 16)) + 1,
                    "count": min(len(picks), max(int(d.max_count), 1)) / float(max(int(d.max_count), 1)),
                }
            else:
                event = None
        else:
            ctx = int(np.rint(d.state_feats[17] * 64.0)) if len(d.state_feats) > 17 else 0
            event = {
                "type": 15,
                "card": 0,
                "card2": 0,
                "attack": 0,
                "context": max(0, min(ctx, 64)) + 1,
                "select_type": 1,
                "count": 0.0,
            }
        if event is None:
            return
        self._history.append(event)
        if len(self._history) > self._history_k:
            del self._history[:-self._history_k]

    @classmethod
    def load(cls, path: str) -> "NumpyPolicy":
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    # ── state / value ───────────────────────────────────────────────

    def _pool(self, ids: np.ndarray) -> np.ndarray:
        e = self.w["card_emb.weight"][ids]
        mask = (ids > 0).astype(np.float32)[:, None]
        return (e * mask).sum(axis=0) / (mask.sum() + 1e-8)

    def _encode_history(self, history: dict[str, np.ndarray] | None) -> np.ndarray:
        if self._history_k <= 0:
            return np.zeros(0, dtype=np.float32)
        hidden_dim = self.w["history_gru.weight_hh_l0"].shape[1]
        if not history:
            return np.zeros(hidden_dim, dtype=np.float32)
        mask = np.asarray(history.get("mask", np.zeros(self._history_k)), dtype=np.float32)[-self._history_k:]
        if mask.shape[0] < self._history_k:
            mask = np.pad(mask, (self._history_k - mask.shape[0], 0))
        def arr(name: str, dtype=np.int64):
            x = np.asarray(history.get(name, np.zeros(self._history_k)), dtype=dtype)[-self._history_k:]
            if x.shape[0] < self._history_k:
                x = np.pad(x, (self._history_k - x.shape[0], 0))
            return x
        typ = arr("type").clip(0, self.w["history_type_emb.weight"].shape[0] - 1)
        card = arr("card").clip(0, self.w["card_emb.weight"].shape[0] - 1)
        card2 = arr("card2").clip(0, self.w["card_emb.weight"].shape[0] - 1)
        attack = arr("attack").clip(0, self.w["attack_emb.weight"].shape[0] - 1)
        ctx = arr("context").clip(0, self.w["history_context_emb.weight"].shape[0] - 1)
        sel_type = arr("select_type").clip(0, self.w["history_select_type_emb.weight"].shape[0] - 1)
        count = arr("count", dtype=np.float32).astype(np.float32)
        pos = np.arange(self._history_k, dtype=np.int64)
        x = np.concatenate([
            self.w["card_emb.weight"][card],
            self.w["card_emb.weight"][card2],
            self.w["attack_emb.weight"][attack],
            self.w["history_type_emb.weight"][typ],
            self.w["history_context_emb.weight"][ctx],
            self.w["history_select_type_emb.weight"][sel_type],
            self.w["history_pos_emb.weight"][pos],
            count[:, None],
            mask[:, None],
        ], axis=-1)
        token = _relu(_linear(self.w["history_token_fc.weight"], self.w["history_token_fc.bias"], x))
        token *= mask[:, None]
        w_ih = self.w["history_gru.weight_ih_l0"]
        w_hh = self.w["history_gru.weight_hh_l0"]
        b_ih = self.w["history_gru.bias_ih_l0"]
        b_hh = self.w["history_gru.bias_hh_l0"]
        h = np.zeros(hidden_dim, dtype=np.float32)
        for t in range(self._history_k):
            if mask[t] <= 0:
                continue
            gi = token[t] @ w_ih.T + b_ih
            gh = h @ w_hh.T + b_hh
            i_r, i_z, i_n = np.split(gi, 3)
            h_r, h_z, h_n = np.split(gh, 3)
            r = 1.0 / (1.0 + np.exp(-np.clip(i_r + h_r, -30.0, 30.0)))
            z = 1.0 / (1.0 + np.exp(-np.clip(i_z + h_z, -30.0, 30.0)))
            n = np.tanh(i_n + r * h_n)
            h = (1.0 - z) * n + z * h
        return h.astype(np.float32, copy=False)

    def _merge_history(self, h: np.ndarray, history: dict[str, np.ndarray] | None) -> np.ndarray:
        if self._history_k <= 0:
            return h
        hist = self._encode_history(history)
        return _relu(_linear(
            self.w["history_out_fc.weight"],
            self.w["history_out_fc.bias"],
            np.concatenate([h, hist]),
        ))

    def encode_state(self, board: np.ndarray, hand: np.ndarray,
                     feats: np.ndarray, history: dict[str, np.ndarray] | None = None) -> np.ndarray:
        if self._arch == "cross_attn":
            h, tokens, mask = self._encode_state_cross(board, hand, feats, history)
            self._cached_state_tokens = tokens
            self._cached_state_mask = mask
            return h
        feats = self._fit_feat_dim(feats, self._state_feat_dim)
        if self._slot_state:
            emb = self.w["card_emb.weight"]
            my_active = emb[board[0]]
            my_bench = self._pool(board[1:6])
            opp_active = emb[board[6]]
            opp_bench = self._pool(board[7:])
            hnd = self._pool(hand)
            x = np.concatenate([my_active, my_bench, opp_active, opp_bench, hnd, feats])
            x = _relu(_linear(self.w["state_fc1.weight"], self.w["state_fc1.bias"], x))
            return self._merge_history(
                _relu(_linear(self.w["state_fc2.weight"], self.w["state_fc2.bias"], x)),
                history,
            )
        my = self._pool(board[:6])
        opp = self._pool(board[6:])
        hnd = self._pool(hand)
        x = np.concatenate([my, opp, hnd, feats])
        x = _relu(_linear(self.w["state_fc1.weight"], self.w["state_fc1.bias"], x))
        return self._merge_history(
            _relu(_linear(self.w["state_fc2.weight"], self.w["state_fc2.bias"], x)),
            history,
        )

    def _encode_state_cross(
        self,
        board: np.ndarray,
        hand: np.ndarray,
        feats: np.ndarray,
        history: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats = self._fit_feat_dim(feats, self._state_feat_dim)
        ids = np.concatenate([board.astype(np.int64), hand.astype(np.int64)])
        board_area = np.asarray([1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4], dtype=np.int64)
        board_index = np.asarray([0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4], dtype=np.int64)
        hand_area = np.full(hand.shape[0], 5, dtype=np.int64)
        hand_index = np.arange(hand.shape[0], dtype=np.int64).clip(0, 64)
        area = np.concatenate([board_area, hand_area])
        index = np.concatenate([board_index, hand_index])
        token_in = np.concatenate([
            self.w["card_emb.weight"][ids],
            self.w["state_area_emb.weight"][area],
            self.w["state_index_emb.weight"][index],
        ], axis=-1)
        card_tokens = _relu(_linear(
            self.w["state_token_fc.weight"],
            self.w["state_token_fc.bias"],
            token_in,
        ))
        feat_token = _relu(_linear(
            self.w["feat_token_fc.weight"],
            self.w["feat_token_fc.bias"],
            feats,
        ))[np.newaxis, :]
        tokens = np.concatenate([feat_token, card_tokens], axis=0)
        mask = np.concatenate([np.ones(1, dtype=bool), ids > 0])

        for i in range(self._state_layer_count):
            prefix = f"state_layers.{i}."
            q = _linear(self.w[prefix + "q.weight"], self.w[prefix + "q.bias"], tokens)
            k = _linear(self.w[prefix + "k.weight"], self.w[prefix + "k.bias"], tokens)
            v = _linear(self.w[prefix + "v.weight"], self.w[prefix + "v.bias"], tokens)
            scores = q @ k.T / math.sqrt(max(q.shape[-1], 1))
            scores[:, ~mask] = NEG_INF
            attn = _softmax(scores, axis=-1)
            y = _linear(self.w[prefix + "o.weight"], self.w[prefix + "o.bias"], attn @ v)
            tokens = _relu(tokens + y)
            y = _linear(
                self.w[prefix + "ff2.weight"],
                self.w[prefix + "ff2.bias"],
                _relu(_linear(self.w[prefix + "ff1.weight"], self.w[prefix + "ff1.bias"], tokens)),
            )
            tokens = _relu(tokens + y)
            tokens[~mask] = 0.0

        pool_logits = _linear(
            self.w["state_pool_fc.weight"],
            self.w["state_pool_fc.bias"],
            tokens,
        ).reshape(-1)
        pool_logits[~mask] = NEG_INF
        pool = _softmax(pool_logits, axis=-1)
        pooled = (tokens * pool[:, None]).sum(axis=0)
        h = _relu(_linear(self.w["state_out_fc.weight"], self.w["state_out_fc.bias"], pooled))
        return self._merge_history(h, history), tokens, mask

    def value(self, h: np.ndarray) -> float:
        v = _relu(_linear(self.w["value_fc1.weight"], self.w["value_fc1.bias"], h))
        return float(np.tanh(_linear(self.w["value_fc2.weight"], self.w["value_fc2.bias"], v)))

    def _plan_context(self, h: np.ndarray) -> np.ndarray:
        if not self._hierarchical_plan:
            return np.zeros(0, dtype=np.float32)
        x = _relu(_linear(self.w["plan_fc1.weight"], self.w["plan_fc1.bias"], h))
        logits = _linear(self.w["plan_fc2.weight"], self.w["plan_fc2.bias"], x)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return _relu(_linear(
            self.w["plan_condition_fc.weight"],
            self.w["plan_condition_fc.bias"],
            probs,
        )).astype(np.float32, copy=False)

    def _score_rows(self, h: np.ndarray, rows: np.ndarray, picked_sum: np.ndarray) -> np.ndarray:
        hx = np.broadcast_to(h, (rows.shape[0], self._hd))
        px = np.broadcast_to(picked_sum, (rows.shape[0], self._oe))
        score_x = np.concatenate([hx, rows, px], axis=-1)
        logits = _linear(
            self.w["score_fc2.weight"],
            self.w["score_fc2.bias"],
            _relu(_linear(self.w["score_fc1.weight"], self.w["score_fc1.bias"], score_x)),
        ).reshape(-1)
        if self._hierarchical_plan:
            pc = self._plan_context(h)
            plan_x = np.concatenate([
                rows,
                px,
                np.broadcast_to(pc, (rows.shape[0], self._plan_cond_dim)),
            ], axis=-1)
            logits = logits + _linear(
                self.w["plan_score_fc2.weight"],
                self.w["plan_score_fc2.bias"],
                _relu(_linear(self.w["plan_score_fc1.weight"], self.w["plan_score_fc1.bias"], plan_x)),
            ).reshape(-1)
        return logits

    @staticmethod
    def _fit_feat_dim(x: np.ndarray, dim: int) -> np.ndarray:
        if x.shape[-1] == dim:
            return x.astype(np.float32, copy=False)
        if x.shape[-1] > dim:
            return x[..., :dim].astype(np.float32, copy=False)
        pad = [(0, 0)] * x.ndim
        pad[-1] = (0, dim - x.shape[-1])
        return np.pad(x.astype(np.float32, copy=False), pad)

    def _evaluate_state(self, obs_dict: dict) -> float:
        """V(s) from a raw observation dict. Higher = better for current player."""
        try:
            d = self.encoder.encode(obs_dict)
            h = self.encode_state(d.board_cards, d.hand_cards, d.state_feats, self._history_arrays())
            return self.value(h)
        except Exception:
            return 0.0

    # ── greedy / sampling select ────────────────────────────────────

    def select(self, obs_dict: dict, greedy: bool = True, temperature: float = 1.0,
               top_k: int = 0, update_history: bool = True) -> list[int]:
        """Action selection. temperature=1.0 = greedy, >1.0 = more random."""
        if self._arch == "cross_attn":
            return self._select_cross(
                obs_dict,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k,
                update_history=update_history,
            )
        sel = obs_dict.get("select")
        if sel is None:
            raise ValueError("deck selection — return deck directly")

        d = self.encoder.encode(obs_dict)
        n = len(d.opt_type)
        h = self.encode_state(d.board_cards, d.hand_cards, d.state_feats, self._history_arrays())
        opt_feats = self._fit_feat_dim(d.opt_feats, self._opt_feat_dim)

        parts = [
            self.w["card_emb.weight"][d.opt_card],
            self.w["card_emb.weight"][d.opt_card2],
            self.w["attack_emb.weight"][d.opt_attack],
            self.w["opt_type_emb.weight"][d.opt_type],
        ]
        if self._has_option_context:
            ctx = np.rint(opt_feats[:, 3] * 64.0).astype(np.int64).clip(0, 64)
            sel_type = np.rint(opt_feats[:, 4] * 16.0).astype(np.int64).clip(0, 16)
            area = np.rint(opt_feats[:, 7] * 16.0).astype(np.int64).clip(0, 16)
            idx = np.rint(opt_feats[:, 8] * 64.0).astype(np.int64).clip(0, 64)
            inplay_area = np.rint(opt_feats[:, 9] * 16.0).astype(np.int64).clip(0, 16)
            inplay_idx = np.rint(opt_feats[:, 10] * 10.0).astype(np.int64).clip(0, 16)
            parts.extend([
                self.w["context_emb.weight"][ctx],
                self.w["select_type_emb.weight"][sel_type],
                self.w["area_emb.weight"][area],
                self.w["index_emb.weight"][idx],
                self.w["inplay_area_emb.weight"][inplay_area],
                self.w["inplay_index_emb.weight"][inplay_idx],
            ])
        parts.append(opt_feats)
        opt_x = np.concatenate(parts, axis=-1)
        opts = _relu(_linear(self.w["opt_fc.weight"], self.w["opt_fc.bias"], opt_x))

        picks, picked_sum = [], np.zeros(self._oe, dtype=np.float32)
        avail = np.ones(n + 1, dtype=bool)

        while len(picks) < d.max_count:
            avail[n] = len(picks) >= d.min_count
            rows = np.concatenate([opts, self.w["stop_vec"][np.newaxis, :]], axis=0)
            logits = self._score_rows(h, rows, picked_sum)
            logits = np.where(avail, logits, NEG_INF)
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            logits = logits - logits.max()
            if not greedy and top_k > 0:
                legal = np.flatnonzero(avail)
                if len(legal) > top_k:
                    keep = legal[np.argsort(logits[legal])[-top_k:]]
                    top_mask = np.zeros_like(avail)
                    top_mask[keep] = True
                    logits = np.where(top_mask, logits, NEG_INF)
            probs = np.exp(logits) / np.exp(logits).sum()

            if greedy:
                idx = int(np.argmax(probs))
            else:
                idx = int(np.random.choice(n + 1, p=probs))
            if idx >= n:
                break
            picks.append(idx)
            picked_sum += opts[idx]
            avail[idx] = False

        picks = picks[:d.max_count]
        if update_history:
            self._remember_decision(d, picks)
        return picks

    def _option_base(self, d, opt_feats: np.ndarray) -> np.ndarray:
        parts = [
            self.w["card_emb.weight"][d.opt_card],
            self.w["card_emb.weight"][d.opt_card2],
            self.w["attack_emb.weight"][d.opt_attack],
            self.w["opt_type_emb.weight"][d.opt_type],
        ]
        if self._has_option_context:
            ctx = np.rint(opt_feats[:, 3] * 64.0).astype(np.int64).clip(0, 64)
            sel_type = np.rint(opt_feats[:, 4] * 16.0).astype(np.int64).clip(0, 16)
            area = np.rint(opt_feats[:, 7] * 16.0).astype(np.int64).clip(0, 16)
            idx = np.rint(opt_feats[:, 8] * 64.0).astype(np.int64).clip(0, 64)
            inplay_area = np.rint(opt_feats[:, 9] * 16.0).astype(np.int64).clip(0, 16)
            inplay_idx = np.rint(opt_feats[:, 10] * 10.0).astype(np.int64).clip(0, 16)
            parts.extend([
                self.w["context_emb.weight"][ctx],
                self.w["select_type_emb.weight"][sel_type],
                self.w["area_emb.weight"][area],
                self.w["index_emb.weight"][idx],
                self.w["inplay_area_emb.weight"][inplay_area],
                self.w["inplay_index_emb.weight"][inplay_idx],
            ])
        parts.append(opt_feats)
        return _relu(_linear(
            self.w["opt_fc.weight"],
            self.w["opt_fc.bias"],
            np.concatenate(parts, axis=-1),
        ))

    def _cross_options(self, base: np.ndarray, tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
        q = _linear(self.w["cross_q.weight"], self.w["cross_q.bias"], base)
        k = _linear(self.w["cross_k.weight"], self.w["cross_k.bias"], tokens)
        v = _linear(self.w["cross_v.weight"], self.w["cross_v.bias"], tokens)
        scores = q @ k.T / math.sqrt(max(q.shape[-1], 1))
        scores[:, ~mask] = NEG_INF
        attn = _softmax(scores, axis=-1)
        ctx = attn @ v
        return _relu(_linear(
            self.w["cross_out.weight"],
            self.w["cross_out.bias"],
            np.concatenate([base, ctx], axis=-1),
        ))

    def _select_cross(self, obs_dict: dict, greedy: bool = True, temperature: float = 1.0,
                      top_k: int = 0, update_history: bool = True) -> list[int]:
        sel = obs_dict.get("select")
        if sel is None:
            raise ValueError("deck selection — return deck directly")

        d = self.encoder.encode(obs_dict)
        n = len(d.opt_type)
        h, tokens, token_mask = self._encode_state_cross(
            d.board_cards,
            d.hand_cards,
            d.state_feats,
            self._history_arrays(),
        )
        opt_feats = self._fit_feat_dim(d.opt_feats, self._opt_feat_dim)
        opts = self._cross_options(self._option_base(d, opt_feats), tokens, token_mask)

        picks, picked_sum = [], np.zeros(self._oe, dtype=np.float32)
        avail = np.ones(n + 1, dtype=bool)

        while len(picks) < d.max_count:
            avail[n] = len(picks) >= d.min_count
            rows = np.concatenate([opts, self.w["stop_vec"][np.newaxis, :]], axis=0)
            logits = self._score_rows(h, rows, picked_sum)
            logits = np.where(avail, logits, NEG_INF)
            if temperature != 1.0:
                logits = logits / temperature
            if not greedy and top_k > 0:
                legal = np.flatnonzero(avail)
                if len(legal) > top_k:
                    keep = legal[np.argsort(logits[legal])[-top_k:]]
                    top_mask = np.zeros_like(avail)
                    top_mask[keep] = True
                    logits = np.where(top_mask, logits, NEG_INF)
            probs = _softmax(logits, axis=-1)
            idx = int(np.argmax(probs)) if greedy else int(np.random.choice(n + 1, p=probs))
            if idx >= n:
                break
            picks.append(idx)
            picked_sum += opts[idx]
            avail[idx] = False

        picks = picks[:d.max_count]
        if update_history:
            self._remember_decision(d, picks)
        return picks

    def first_step_ranking(self, obs_dict: dict, temperature: float = 1.0) -> list[dict]:
        """Return the first-pick option ranking used by greedy selection.

        This is a diagnostic helper for trace tooling. It includes the STOP row
        as index ``-1`` when STOP is legal at the first pick.
        """
        sel = obs_dict.get("select")
        if sel is None:
            raise ValueError("deck selection has no option ranking")
        if self._arch == "cross_attn":
            return self._first_step_ranking_cross(obs_dict, temperature=temperature)

        d = self.encoder.encode(obs_dict)
        n = len(d.opt_type)
        h = self.encode_state(d.board_cards, d.hand_cards, d.state_feats, self._history_arrays())
        opt_feats = self._fit_feat_dim(d.opt_feats, self._opt_feat_dim)

        parts = [
            self.w["card_emb.weight"][d.opt_card],
            self.w["card_emb.weight"][d.opt_card2],
            self.w["attack_emb.weight"][d.opt_attack],
            self.w["opt_type_emb.weight"][d.opt_type],
        ]
        if self._has_option_context:
            ctx = np.rint(opt_feats[:, 3] * 64.0).astype(np.int64).clip(0, 64)
            sel_type = np.rint(opt_feats[:, 4] * 16.0).astype(np.int64).clip(0, 16)
            area = np.rint(opt_feats[:, 7] * 16.0).astype(np.int64).clip(0, 16)
            idx = np.rint(opt_feats[:, 8] * 64.0).astype(np.int64).clip(0, 64)
            inplay_area = np.rint(opt_feats[:, 9] * 16.0).astype(np.int64).clip(0, 16)
            inplay_idx = np.rint(opt_feats[:, 10] * 10.0).astype(np.int64).clip(0, 16)
            parts.extend([
                self.w["context_emb.weight"][ctx],
                self.w["select_type_emb.weight"][sel_type],
                self.w["area_emb.weight"][area],
                self.w["index_emb.weight"][idx],
                self.w["inplay_area_emb.weight"][inplay_area],
                self.w["inplay_index_emb.weight"][inplay_idx],
            ])
        parts.append(opt_feats)
        opt_x = np.concatenate(parts, axis=-1)
        opts = _relu(_linear(self.w["opt_fc.weight"], self.w["opt_fc.bias"], opt_x))

        rows = np.concatenate([opts, self.w["stop_vec"][np.newaxis, :]], axis=0)
        picked = np.zeros(self._oe, dtype=np.float32)
        logits = self._score_rows(h, rows, picked)
        if temperature != 1.0:
            logits = logits / temperature
        avail = np.ones(n + 1, dtype=bool)
        avail[n] = d.min_count <= 0
        logits = np.where(avail, logits, NEG_INF)
        shifted = logits - np.max(logits)
        probs = np.exp(shifted) / np.exp(shifted).sum()

        ranking = []
        for i in range(n + 1):
            if not avail[i]:
                continue
            ranking.append({
                "index": i if i < n else -1,
                "logit": float(logits[i]),
                "prob": float(probs[i]),
                "type": int(d.opt_type[i]) if i < n else 14,
                "card": int(d.opt_card[i]) if i < n else 0,
            })
        ranking.sort(key=lambda r: (-float(r["prob"]), int(r["index"])))
        return ranking

    def _first_step_ranking_cross(self, obs_dict: dict, temperature: float = 1.0) -> list[dict]:
        d = self.encoder.encode(obs_dict)
        n = len(d.opt_type)
        h, tokens, token_mask = self._encode_state_cross(
            d.board_cards,
            d.hand_cards,
            d.state_feats,
            self._history_arrays(),
        )
        opt_feats = self._fit_feat_dim(d.opt_feats, self._opt_feat_dim)
        opts = self._cross_options(self._option_base(d, opt_feats), tokens, token_mask)
        rows = np.concatenate([opts, self.w["stop_vec"][np.newaxis, :]], axis=0)
        picked = np.zeros(self._oe, dtype=np.float32)
        logits = self._score_rows(h, rows, picked)
        if temperature != 1.0:
            logits = logits / temperature
        avail = np.ones(n + 1, dtype=bool)
        avail[n] = d.min_count <= 0
        logits = np.where(avail, logits, NEG_INF)
        probs = _softmax(logits, axis=-1)

        ranking = []
        for i in range(n + 1):
            if not avail[i]:
                continue
            ranking.append({
                "index": i if i < n else -1,
                "logit": float(logits[i]),
                "prob": float(probs[i]),
                "type": int(d.opt_type[i]) if i < n else 14,
                "card": int(d.opt_card[i]) if i < n else 0,
            })
        ranking.sort(key=lambda r: (-float(r["prob"]), int(r["index"])))
        return ranking

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
            try:
                self._remember_decision(self.encoder.encode(obs_dict), [])
            except Exception:
                pass
            return []

        picks = best["sel"][:mc_sel]
        try:
            self._remember_decision(self.encoder.encode(obs_dict), picks)
        except Exception:
            pass
        return picks
