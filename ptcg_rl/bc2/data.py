from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from ptcg_rl.encoder import OPT_FEAT_DIM, STATE_FEAT_DIM


@dataclass
class BCBatch:
    board: torch.Tensor
    hand: torch.Tensor
    feats: torch.Tensor
    opt_type: torch.Tensor
    opt_card: torch.Tensor
    opt_card2: torch.Tensor
    opt_attack: torch.Tensor
    opt_feats: torch.Tensor
    targets: torch.Tensor
    min_count: torch.Tensor
    max_count: torch.Tensor
    opt_len: torch.Tensor
    sample_weight: torch.Tensor
    actions: list[list[int]]
    n_options: list[int]
    contexts: list[int]
    true_first_types: list[int]
    max_options: int


def discover_npz_paths(corpus: str, archetype: str, score_bands: Iterable[str]) -> list[str]:
    arch = archetype.replace(" ", "_")
    paths: list[str] = []
    for band in score_bands:
        paths.extend(sorted(glob.glob(os.path.join(corpus, arch, band.replace(" ", "_"), "*.npz"))))
    return paths


def _label_status(data: dict[str, np.ndarray], i: int, include_empty: bool) -> str:
    action = np.asarray(data["action"][i], dtype=np.int64)
    n_opt = len(data["ot"][i])
    mn = int(data["min_c"][i])
    mx = int(data["max_c"][i])
    if len(action) == 0:
        return "keep" if include_empty and mn == 0 else "empty"
    if len(action) < mn or len(action) > mx:
        return "bad"
    if len(set(action.tolist())) != len(action):
        return "bad"
    if not ((action >= 0) & (action < n_opt)).all():
        return "bad"
    return "keep"


class BCCorpus:
    """NPZ-backed corpus index with strict label filtering and vectorized collation."""

    def __init__(
        self,
        paths: list[str],
        *,
        include_empty: bool = False,
        option_weight: float = 0.0,
        state_feat_dim: int = STATE_FEAT_DIM,
        opt_feat_dim: int = OPT_FEAT_DIM,
        deck_sigs: Iterable[str] | None = None,
        winner_only: bool = False,
        win_weight: float = 1.0,
        loss_weight: float = 1.0,
        draw_weight: float = 1.0,
    ):
        if not paths:
            raise FileNotFoundError("No BC corpus .npz files found")
        self.include_empty = include_empty
        self.option_weight = float(option_weight)
        self.state_feat_dim = int(state_feat_dim)
        self.opt_feat_dim = int(opt_feat_dim)
        self.deck_sigs = {str(x) for x in (deck_sigs or []) if str(x)}
        self.winner_only = bool(winner_only)
        self.win_weight = float(win_weight)
        self.loss_weight = float(loss_weight)
        self.draw_weight = float(draw_weight)
        self.npz_data: list[dict[str, np.ndarray]] = []
        self.groups: list[list[tuple[int, int]]] = []
        self.stats = {"raw": 0, "kept": 0, "empty": 0, "bad": 0, "deck_filtered": 0, "outcome_filtered": 0}

        for path in paths:
            with np.load(path, allow_pickle=True) as z:
                data = {k: z[k] for k in z.files}
            if self.deck_sigs and "deck_sig" not in data:
                raise ValueError(
                    "Corpus does not contain deck_sig metadata. Re-extract with the updated "
                    "tools/bc_extract_v2.py before using --deck-sig."
                )
            if self.winner_only and "won" not in data:
                raise ValueError(
                    "Corpus does not contain outcome metadata. Re-extract with the updated "
                    "tools/bc_extract_v2.py before using --winner-only."
                )
            di = len(self.npz_data)
            group: list[tuple[int, int]] = []
            for i in range(len(data["board"])):
                self.stats["raw"] += 1
                if self.deck_sigs and str(data["deck_sig"][i]) not in self.deck_sigs:
                    self.stats["deck_filtered"] += 1
                    continue
                if self.winner_only and int(data["won"][i]) != 1:
                    self.stats["outcome_filtered"] += 1
                    continue
                status = _label_status(data, i, include_empty)
                if status == "keep":
                    group.append((di, i))
                    self.stats["kept"] += 1
                else:
                    self.stats[status] += 1
            self.npz_data.append(data)
            if group:
                self.groups.append(group)

    def split_indices(self, val_fraction: float = 0.1, seed: int = 7) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        rng = np.random.default_rng(seed)
        groups = list(self.groups)
        rng.shuffle(groups)
        if len(groups) <= 1:
            flat = [x for g in groups for x in g]
            n_val = max(1, int(len(flat) * val_fraction))
            return flat[:-n_val], flat[-n_val:]
        split = int(round(len(groups) * (1.0 - val_fraction)))
        split = min(max(split, 1), len(groups) - 1)
        train = [x for g in groups[:split] for x in g]
        val = [x for g in groups[split:] for x in g]
        return train, val

    def all_indices(self) -> list[tuple[int, int]]:
        return [x for g in self.groups for x in g]

    def collate(self, indices: list[tuple[int, int]], device: torch.device) -> BCBatch:
        bsz = len(indices)
        n_options = [len(self.npz_data[di]["ot"][si]) for di, si in indices]
        max_options = max(n_options)
        board = np.empty((bsz, 12), dtype=np.int64)
        hand = np.zeros((bsz, 25), dtype=np.int64)
        feats = np.zeros((bsz, self.state_feat_dim), dtype=np.float32)
        opt_type = np.zeros((bsz, max_options), dtype=np.int64)
        opt_card = np.zeros((bsz, max_options), dtype=np.int64)
        opt_card2 = np.zeros((bsz, max_options), dtype=np.int64)
        opt_attack = np.zeros((bsz, max_options), dtype=np.int64)
        opt_feats = np.zeros((bsz, max_options, self.opt_feat_dim), dtype=np.float32)
        min_count = np.empty(bsz, dtype=np.int64)
        max_count = np.empty(bsz, dtype=np.int64)
        weights = np.ones(bsz, dtype=np.float32)
        actions: list[list[int]] = []
        contexts: list[int] = []
        true_first_types: list[int] = []

        for bi, (di, si) in enumerate(indices):
            data = self.npz_data[di]
            n = n_options[bi]
            board[bi] = np.asarray(data["board"][si], dtype=np.int64)
            h = np.asarray(data["hand"][si], dtype=np.int64)
            hand[bi, : min(len(h), 25)] = h[:25]
            ft = np.asarray(data["feats"][si], dtype=np.float32)
            feats[bi, : min(len(ft), self.state_feat_dim)] = ft[: self.state_feat_dim]
            opt_type[bi, :n] = np.asarray(data["ot"][si], dtype=np.int64)
            opt_card[bi, :n] = np.asarray(data["oc"][si], dtype=np.int64)
            opt_card2[bi, :n] = np.asarray(data["oc2"][si], dtype=np.int64)
            opt_attack[bi, :n] = np.asarray(data["oa"][si], dtype=np.int64)
            of = np.asarray(data["of_arr"][si], dtype=np.float32)
            opt_feats[bi, :n, : min(of.shape[-1], self.opt_feat_dim)] = of[:, : self.opt_feat_dim]
            action = np.asarray(data["action"][si], dtype=np.int64).tolist()
            actions.append([int(a) for a in action])
            min_count[bi] = int(data["min_c"][si])
            max_count[bi] = int(data["max_c"][si])
            contexts.append(int(round(float(feats[bi, 17]) * 64.0)))
            first = actions[-1][0] if actions[-1] else -1
            true_first_types.append(int(opt_type[bi, first]) if first >= 0 else -1)
            weights[bi] += self.option_weight * np.log1p(float(n))
            if "won" in data:
                if int(data["won"][si]) == 1:
                    weights[bi] *= self.win_weight
                elif "draw" in data and int(data["draw"][si]) == 1:
                    weights[bi] *= self.draw_weight
                else:
                    weights[bi] *= self.loss_weight

        max_steps = max(len(a) for a in actions) + 1
        targets = np.full((bsz, max_steps), -1, dtype=np.int64)
        for bi, (action, n) in enumerate(zip(actions, n_options)):
            valid = [a for a in action if 0 <= a < n]
            targets[bi, : len(valid)] = valid
            if len(action) < max_count[bi] and len(action) >= min_count[bi] and (action or self.include_empty):
                targets[bi, len(action)] = max_options

        return BCBatch(
            board=torch.as_tensor(board, device=device),
            hand=torch.as_tensor(hand, device=device),
            feats=torch.as_tensor(feats, device=device),
            opt_type=torch.as_tensor(opt_type, device=device),
            opt_card=torch.as_tensor(opt_card, device=device),
            opt_card2=torch.as_tensor(opt_card2, device=device),
            opt_attack=torch.as_tensor(opt_attack, device=device),
            opt_feats=torch.as_tensor(opt_feats, device=device),
            targets=torch.as_tensor(targets, device=device),
            min_count=torch.as_tensor(min_count, device=device),
            max_count=torch.as_tensor(max_count, device=device),
            opt_len=torch.as_tensor(n_options, dtype=torch.long, device=device),
            sample_weight=torch.as_tensor(weights, device=device),
            actions=actions,
            n_options=n_options,
            contexts=contexts,
            true_first_types=true_first_types,
            max_options=max_options,
        )
