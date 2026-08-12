from __future__ import annotations

import glob
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from ptcg_rl.encoder import OPT_FEAT_DIM, STATE_FEAT_DIM
from ptcg_rl.history_features import BOARD_HISTORY_FEAT_DIM, HISTORY_SUMMARY_DIM, history_summary_from_arrays
from ptcg_rl.plan_labels import PLAN_DIM, PLAN_LABELS, label_decision_plan


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


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
    outcome_value: torch.Tensor
    outcome_mask: torch.Tensor
    trajectory_target: torch.Tensor
    trajectory_mask: torch.Tensor
    plan_step_target: torch.Tensor
    plan_step_mask: torch.Tensor
    history: dict[str, torch.Tensor]
    actions: list[list[int]]
    n_options: list[int]
    contexts: list[int]
    true_first_types: list[int]
    max_options: int


def filter_npz_paths_by_date(
    paths: Iterable[str],
    *,
    date_from: str = "",
    date_to: str = "",
) -> list[str]:
    date_from = str(date_from or "").strip()
    date_to = str(date_to or "").strip()
    if not date_from and not date_to:
        return list(paths)
    out: list[str] = []
    for path in paths:
        m = _DATE_RE.search(os.path.basename(path))
        day = m.group(1) if m else ""
        if date_from and day and day < date_from:
            continue
        if date_to and day and day > date_to:
            continue
        out.append(path)
    return out


def discover_npz_paths(
    corpus: str,
    archetype: str,
    score_bands: Iterable[str],
    *,
    date_from: str = "",
    date_to: str = "",
) -> list[str]:
    arch = archetype.replace(" ", "_")
    paths: list[str] = []
    for band in score_bands:
        paths.extend(sorted(glob.glob(os.path.join(corpus, arch, band.replace(" ", "_"), "*.npz"))))
    return filter_npz_paths_by_date(paths, date_from=date_from, date_to=date_to)


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


def _trajectory_game_key(data: dict[str, np.ndarray], i: int) -> str:
    return f"{data['episode_id'][i]}:{data['player_index'][i]}"


def _state_context(data: dict[str, np.ndarray], i: int) -> int:
    try:
        feats = np.asarray(data["feats"][i], dtype=np.float32)
        if len(feats) > 17:
            return int(round(float(feats[17]) * 64.0))
    except Exception:
        pass
    return 0


def _history_event(data: dict[str, np.ndarray], i: int) -> tuple[int, int, int, int, int, int, float]:
    """Return one compact event for the labeled decision at row ``i``.

    Integer ids are offset by +1 where zero is reserved for padding.
    """
    action = np.asarray(data["action"][i], dtype=np.int64)
    max_count = max(int(data["max_c"][i]), 1)
    if len(action) == 0:
        return 15, 0, 0, 0, _state_context(data, i) + 1, 1, 0.0
    first = int(action[0])
    ot = np.asarray(data["ot"][i], dtype=np.int64)
    if first < 0 or first >= len(ot):
        return 0, 0, 0, 0, 0, 0, 0.0
    oc = np.asarray(data["oc"][i], dtype=np.int64)
    oc2 = np.asarray(data["oc2"][i], dtype=np.int64)
    oa = np.asarray(data["oa"][i], dtype=np.int64)
    of = np.asarray(data["of_arr"][i], dtype=np.float32)
    ctx = 0
    sel_type = 0
    if of.ndim == 2 and first < of.shape[0]:
        if of.shape[1] > 3:
            ctx = int(round(float(of[first, 3]) * 64.0))
        if of.shape[1] > 4:
            sel_type = int(round(float(of[first, 4]) * 16.0))
    valid = [int(a) for a in action if 0 <= int(a) < len(ot)]
    count = min(len(valid), max_count) / float(max_count)
    return (
        int(ot[first]) + 1,
        int(oc[first]) if first < len(oc) else 0,
        int(oc2[first]) if first < len(oc2) else 0,
        int(oa[first]) if first < len(oa) else 0,
        max(0, min(ctx, 64)) + 1,
        max(0, min(sel_type, 16)) + 1,
        float(count),
    )


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
        team_names: Iterable[str] | None = None,
        opponent_deck_sigs: Iterable[str] | None = None,
        opponent_archetypes: Iterable[str] | None = None,
        opponent_team_names: Iterable[str] | None = None,
        opponent_deck_sig_weights: dict[str, float] | None = None,
        opponent_archetype_weights: dict[str, float] | None = None,
        winner_only: bool = False,
        win_weight: float = 1.0,
        loss_weight: float = 1.0,
        draw_weight: float = 1.0,
        context_weights: dict[int, float] | None = None,
        type_weights: dict[int, float] | None = None,
        card_weights: dict[int, float] | None = None,
        multi_select_weight: float = 1.0,
        trajectory_filter_keys: set[str] | None = None,
        trajectory_filter_missing: str = "drop",
        trajectory_weights: dict[str, float] | None = None,
        trajectory_default_weight: float = 1.0,
        trajectory_missing: str = "default",
        trajectory_targets: dict[str, np.ndarray] | None = None,
        trajectory_target_dim: int = 0,
        archetype: str = "",
        step_plan: bool = False,
        history_k: int = 0,
        opp_history_k: int = 0,
        log_history_k: int = 0,
        board_history_k: int = 0,
        board_history_feat_dim: int = BOARD_HISTORY_FEAT_DIM,
        history_summary_dim: int = 0,
        split_by_game: bool = False,
        load_progress_every: int = 0,
    ):
        if not paths:
            raise FileNotFoundError("No BC corpus .npz files found")
        self.include_empty = include_empty
        self.option_weight = float(option_weight)
        self.state_feat_dim = int(state_feat_dim)
        self.opt_feat_dim = int(opt_feat_dim)
        self.deck_sigs = {str(x) for x in (deck_sigs or []) if str(x)}
        self.team_names = {str(x).lower() for x in (team_names or []) if str(x)}
        self.opponent_deck_sigs = {str(x) for x in (opponent_deck_sigs or []) if str(x)}
        self.opponent_archetypes = {str(x).lower() for x in (opponent_archetypes or []) if str(x)}
        self.opponent_team_names = {str(x).lower() for x in (opponent_team_names or []) if str(x)}
        self.opponent_deck_sig_weights = {
            str(k): float(v) for k, v in (opponent_deck_sig_weights or {}).items() if str(k)
        }
        self.opponent_archetype_weights = {
            str(k).lower(): float(v) for k, v in (opponent_archetype_weights or {}).items() if str(k)
        }
        self.winner_only = bool(winner_only)
        self.win_weight = float(win_weight)
        self.loss_weight = float(loss_weight)
        self.draw_weight = float(draw_weight)
        self.context_weights = {int(k): float(v) for k, v in (context_weights or {}).items()}
        self.type_weights = {int(k): float(v) for k, v in (type_weights or {}).items()}
        self.card_weights = {int(k): float(v) for k, v in (card_weights or {}).items()}
        self.multi_select_weight = float(multi_select_weight)
        self.trajectory_filter_keys = set(trajectory_filter_keys) if trajectory_filter_keys is not None else None
        self.trajectory_filter_missing = str(trajectory_filter_missing)
        if self.trajectory_filter_missing not in {"keep", "drop"}:
            raise ValueError("trajectory_filter_missing must be 'keep' or 'drop'")
        self.trajectory_weights = {
            str(k): float(v) for k, v in (trajectory_weights or {}).items() if str(k)
        }
        self.trajectory_default_weight = float(trajectory_default_weight)
        self.trajectory_missing = str(trajectory_missing)
        if self.trajectory_missing not in {"default", "drop"}:
            raise ValueError("trajectory_missing must be 'default' or 'drop'")
        self.trajectory_targets = {
            str(k): np.asarray(v, dtype=np.float32)
            for k, v in (trajectory_targets or {}).items()
            if str(k)
        }
        self.trajectory_target_dim = int(trajectory_target_dim)
        if self.trajectory_targets and self.trajectory_target_dim <= 0:
            first = next(iter(self.trajectory_targets.values()))
            self.trajectory_target_dim = int(first.shape[-1])
        self.archetype = str(archetype)
        self.step_plan = bool(step_plan)
        self.step_plan_dim = PLAN_DIM if self.step_plan else 0
        self.history_k = max(0, int(history_k))
        self.opp_history_k = max(0, int(opp_history_k))
        self.log_history_k = max(0, int(log_history_k))
        self.board_history_k = max(0, int(board_history_k))
        self.board_history_feat_dim = max(0, int(board_history_feat_dim))
        self.history_summary_dim = max(0, int(history_summary_dim))
        self.split_by_game = bool(
            split_by_game
            or self.history_k > 0
            or self.opp_history_k > 0
            or self.log_history_k > 0
            or self.board_history_k > 0
            or self.history_summary_dim > 0
            or self.step_plan
        )
        self.npz_data: list[dict[str, np.ndarray]] = []
        self.history_prev: list[np.ndarray] = []
        self.step_plan_targets: list[np.ndarray] = []
        self.groups: list[list[tuple[int, int]]] = []
        global_game_groups: dict[str, list[tuple[int, int]]] = {}
        self.step_plan_counts: Counter[str] = Counter()
        self.stats = {
            "raw": 0,
            "kept": 0,
            "empty": 0,
            "bad": 0,
            "deck_filtered": 0,
            "team_filtered": 0,
            "opponent_deck_filtered": 0,
            "opponent_archetype_filtered": 0,
            "opponent_team_filtered": 0,
            "outcome_filtered": 0,
            "trajectory_keep_filtered": 0,
            "trajectory_filter_matched": 0,
            "trajectory_filtered": 0,
            "trajectory_matched": 0,
            "trajectory_defaulted": 0,
        }

        t0 = time.time()
        for path_i, path in enumerate(paths, 1):
            file_t0 = time.time()
            with np.load(path, allow_pickle=True) as z:
                data = {k: z[k] for k in z.files}
            if self.deck_sigs and "deck_sig" not in data:
                raise ValueError(
                    "Corpus does not contain deck_sig metadata. Re-extract with the updated "
                    "tools/bc_extract_v2.py before using --deck-sig."
                )
            if self.team_names and "team_name" not in data:
                raise ValueError(
                    "Corpus does not contain team_name metadata. Re-extract with the updated "
                    "tools/bc_extract_v2.py before using --team-name."
                )
            if (self.opponent_deck_sigs or self.opponent_deck_sig_weights) and "opponent_deck_sig" not in data:
                raise ValueError(
                    "Corpus does not contain opponent_deck_sig metadata. Re-extract with the updated "
                    "tools/bc_extract_v2.py before using --opponent-deck-sig."
                )
            if (self.opponent_archetypes or self.opponent_archetype_weights) and "opponent_archetype" not in data:
                raise ValueError(
                    "Corpus does not contain opponent_archetype metadata. Re-extract with the updated "
                    "tools/bc_extract_v2.py before using --opponent-archetype."
                )
            if self.opponent_team_names and "opponent_team_name" not in data:
                raise ValueError(
                    "Corpus does not contain opponent_team_name metadata. Re-extract with the updated "
                    "tools/bc_extract_v2.py before using --opponent-team-name."
                )
            if self.winner_only and "won" not in data:
                raise ValueError(
                    "Corpus does not contain outcome metadata. Re-extract with the updated "
                    "tools/bc_extract_v2.py before using --winner-only."
                )
            if (
                self.trajectory_filter_keys is not None
                or self.trajectory_weights
                or self.trajectory_targets
                or self.split_by_game
            ) and (
                "episode_id" not in data or "player_index" not in data
            ):
                raise ValueError(
                    "Corpus does not contain episode_id/player_index metadata. Re-extract with the "
                    "updated tools/bc_extract_v2.py before using trajectory weights or --split-by-game."
                )
            di = len(self.npz_data)
            group: list[tuple[int, int]] = []
            game_groups: dict[str, list[tuple[int, int]]] = {}
            n_rows = len(data["board"])
            file_history_prev = (
                np.full((n_rows, self.history_k), -1, dtype=np.int32)
                if self.history_k > 0 else np.empty((0, 0), dtype=np.int32)
            )
            file_step_plan = (
                np.zeros((n_rows, self.step_plan_dim), dtype=np.float32)
                if self.step_plan else np.empty((0, 0), dtype=np.float32)
            )
            file_raw0 = self.stats["raw"]
            file_kept0 = self.stats["kept"]
            for i in range(n_rows):
                self.stats["raw"] += 1
                if self.deck_sigs and str(data["deck_sig"][i]) not in self.deck_sigs:
                    self.stats["deck_filtered"] += 1
                    continue
                if self.team_names and str(data["team_name"][i]).lower() not in self.team_names:
                    self.stats["team_filtered"] += 1
                    continue
                if self.opponent_deck_sigs and str(data["opponent_deck_sig"][i]) not in self.opponent_deck_sigs:
                    self.stats["opponent_deck_filtered"] += 1
                    continue
                if (
                    self.opponent_archetypes
                    and str(data["opponent_archetype"][i]).lower() not in self.opponent_archetypes
                ):
                    self.stats["opponent_archetype_filtered"] += 1
                    continue
                if (
                    self.opponent_team_names
                    and str(data["opponent_team_name"][i]).lower() not in self.opponent_team_names
                ):
                    self.stats["opponent_team_filtered"] += 1
                    continue
                if self.winner_only and int(data["won"][i]) != 1:
                    self.stats["outcome_filtered"] += 1
                    continue
                game_key = (
                    _trajectory_game_key(data, i)
                    if (
                        self.trajectory_filter_keys is not None
                        or self.trajectory_weights
                        or self.trajectory_targets
                        or self.split_by_game
                    )
                    else ""
                )
                if self.trajectory_filter_keys is not None:
                    in_filter = game_key in self.trajectory_filter_keys
                    if not in_filter and self.trajectory_filter_missing == "drop":
                        self.stats["trajectory_keep_filtered"] += 1
                        continue
                    if in_filter:
                        self.stats["trajectory_filter_matched"] += 1
                trajectory_has_weight = bool(self.trajectory_weights) and game_key in self.trajectory_weights
                if self.trajectory_weights and not trajectory_has_weight and self.trajectory_missing == "drop":
                    self.stats["trajectory_filtered"] += 1
                    continue
                status = _label_status(data, i, include_empty)
                if status == "keep":
                    if self.split_by_game:
                        game_groups.setdefault(game_key, []).append((di, i))
                    else:
                        group.append((di, i))
                    self.stats["kept"] += 1
                    if self.trajectory_weights:
                        if trajectory_has_weight:
                            self.stats["trajectory_matched"] += 1
                        else:
                            self.stats["trajectory_defaulted"] += 1
                else:
                    self.stats[status] += 1
                if load_progress_every and (
                    i + 1 == 1 or (i + 1) % load_progress_every == 0 or i + 1 == n_rows
                ):
                    done = self.stats["raw"]
                    rate = done / max(time.time() - t0, 1e-9)
                    file_rate = (i + 1) / max(time.time() - file_t0, 1e-9)
                    print(
                        f"  load {path_i}/{len(paths)} {os.path.basename(path)} "
                        f"{i+1}/{n_rows} rows kept={self.stats['kept']} "
                        f"file={file_rate:.0f}/s total={rate:.0f}/s",
                        flush=True,
                    )
            file_groups: list[list[tuple[int, int]]] = []
            if self.split_by_game and game_groups:
                for rows in game_groups.values():
                    rows = sorted(rows, key=lambda x: x[1])
                    if self.history_k > 0:
                        for pos, (_, row_i) in enumerate(rows):
                            prev = [r for _, r in rows[max(0, pos - self.history_k) : pos]]
                            if prev:
                                file_history_prev[row_i, -len(prev):] = np.asarray(prev, dtype=np.int32)
                    if self.step_plan:
                        for pos, (_, row_i) in enumerate(rows):
                            target = label_decision_plan(
                                data,
                                row_i,
                                archetype=self.archetype,
                                position=pos,
                                game_len=len(rows),
                            )
                            file_step_plan[row_i] = target
                            for label_i, label_name in enumerate(PLAN_LABELS):
                                if float(target[label_i]) > 0.0:
                                    self.step_plan_counts[label_name] += 1
                    file_groups.append(rows)
            elif group:
                file_groups.append(group)

            if not file_groups:
                if load_progress_every:
                    print(
                        f"  loaded {path_i}/{len(paths)} {os.path.basename(path)} "
                        f"raw={self.stats['raw']-file_raw0} kept={self.stats['kept']-file_kept0} "
                        f"stored=0 {time.time()-file_t0:.1f}s",
                        flush=True,
                    )
                continue

            kept_rows = sorted({int(si) for rows in file_groups for _, si in rows})
            if kept_rows and len(kept_rows) < n_rows:
                old_to_new = {old: new for new, old in enumerate(kept_rows)}
                idx = np.asarray(kept_rows, dtype=np.int64)
                compacted: dict[str, np.ndarray] = {}
                for key, value in data.items():
                    arr = np.asarray(value)
                    if arr.ndim > 0 and arr.shape[0] == n_rows:
                        compacted[key] = arr[idx]
                    else:
                        compacted[key] = value
                data = compacted
                file_groups = [
                    [(di, old_to_new[int(si)]) for _, si in rows if int(si) in old_to_new]
                    for rows in file_groups
                ]
                if self.history_k > 0:
                    prev_old = file_history_prev[idx]
                    prev_new = np.full(prev_old.shape, -1, dtype=np.int32)
                    for r in range(prev_old.shape[0]):
                        for c in range(prev_old.shape[1]):
                            old = int(prev_old[r, c])
                            if old >= 0:
                                prev_new[r, c] = old_to_new.get(old, -1)
                    file_history_prev = prev_new
                if self.step_plan:
                    file_step_plan = file_step_plan[idx]

            self.npz_data.append(data)
            if self.split_by_game:
                for rows in file_groups:
                    if not rows:
                        continue
                    _, first_si = rows[0]
                    game_key = _trajectory_game_key(data, int(first_si))
                    global_game_groups.setdefault(game_key, []).extend(rows)
            else:
                self.groups.extend(rows for rows in file_groups if rows)
            if self.history_k > 0:
                self.history_prev.append(file_history_prev)
            if self.step_plan:
                self.step_plan_targets.append(file_step_plan)
            if load_progress_every:
                print(
                    f"  loaded {path_i}/{len(paths)} {os.path.basename(path)} "
                    f"raw={self.stats['raw']-file_raw0} kept={self.stats['kept']-file_kept0} "
                    f"stored={len(kept_rows)} {time.time()-file_t0:.1f}s",
                    flush=True,
                )
        if self.split_by_game:
            self.groups = [rows for rows in global_game_groups.values() if rows]

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

    @staticmethod
    def _augment_masked_sequence(
        mask: np.ndarray,
        value_arrays: list[np.ndarray],
        *,
        stream_drop_prob: float,
        event_drop_prob: float,
        tail_drop_prob: float,
    ) -> None:
        """In-place history corruption used during training only.

        BC history is extracted from expert prefixes, while live play uses the
        model's own prefixes. Randomly hiding streams, individual events, and
        the most recent suffix makes the model robust to that exposure gap.
        """
        if mask.size == 0:
            return
        bsz = mask.shape[0]
        stream_drop_prob = max(0.0, min(1.0, float(stream_drop_prob)))
        event_drop_prob = max(0.0, min(1.0, float(event_drop_prob)))
        tail_drop_prob = max(0.0, min(1.0, float(tail_drop_prob)))
        for bi in range(bsz):
            valid = np.flatnonzero(mask[bi] > 0)
            if valid.size == 0:
                continue
            drop = False
            if stream_drop_prob > 0 and np.random.random() < stream_drop_prob:
                drop = True
                keep = np.empty(0, dtype=np.int64)
            else:
                keep = valid
                if tail_drop_prob > 0 and np.random.random() < tail_drop_prob:
                    keep_len = int(np.random.randint(0, valid.size + 1))
                    keep = valid[-keep_len:] if keep_len > 0 else np.empty(0, dtype=np.int64)
                if event_drop_prob > 0 and keep.size:
                    event_keep = np.random.random(keep.size) >= event_drop_prob
                    keep = keep[event_keep]
            if drop or keep.size != valid.size:
                drop_idx = np.setdiff1d(valid, keep, assume_unique=True)
                if drop_idx.size:
                    mask[bi, drop_idx] = 0.0
                    for arr in value_arrays:
                        arr[bi, drop_idx] = 0

    def collate(
        self,
        indices: list[tuple[int, int]],
        device: torch.device,
        *,
        history_augment: bool = False,
        history_stream_drop_prob: float = 0.0,
        history_event_drop_prob: float = 0.0,
        history_tail_drop_prob: float = 0.0,
    ) -> BCBatch:
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
        outcome_value = np.zeros(bsz, dtype=np.float32)
        outcome_mask = np.zeros(bsz, dtype=np.float32)
        trajectory_target = np.zeros((bsz, self.trajectory_target_dim), dtype=np.float32)
        trajectory_mask = np.zeros((bsz, self.trajectory_target_dim), dtype=np.float32)
        plan_step_target = np.zeros((bsz, self.step_plan_dim), dtype=np.float32)
        plan_step_mask = np.zeros((bsz, self.step_plan_dim), dtype=np.float32)
        history_type = np.zeros((bsz, self.history_k), dtype=np.int64)
        history_card = np.zeros((bsz, self.history_k), dtype=np.int64)
        history_card2 = np.zeros((bsz, self.history_k), dtype=np.int64)
        history_attack = np.zeros((bsz, self.history_k), dtype=np.int64)
        history_context = np.zeros((bsz, self.history_k), dtype=np.int64)
        history_select_type = np.zeros((bsz, self.history_k), dtype=np.int64)
        history_count = np.zeros((bsz, self.history_k), dtype=np.float32)
        history_mask = np.zeros((bsz, self.history_k), dtype=np.float32)
        opp_history_type = np.zeros((bsz, self.opp_history_k), dtype=np.int64)
        opp_history_card = np.zeros((bsz, self.opp_history_k), dtype=np.int64)
        opp_history_card2 = np.zeros((bsz, self.opp_history_k), dtype=np.int64)
        opp_history_attack = np.zeros((bsz, self.opp_history_k), dtype=np.int64)
        opp_history_context = np.zeros((bsz, self.opp_history_k), dtype=np.int64)
        opp_history_select_type = np.zeros((bsz, self.opp_history_k), dtype=np.int64)
        opp_history_count = np.zeros((bsz, self.opp_history_k), dtype=np.float32)
        opp_history_mask = np.zeros((bsz, self.opp_history_k), dtype=np.float32)
        log_history_type = np.zeros((bsz, self.log_history_k), dtype=np.int64)
        log_history_player = np.zeros((bsz, self.log_history_k), dtype=np.int64)
        log_history_card = np.zeros((bsz, self.log_history_k), dtype=np.int64)
        log_history_card2 = np.zeros((bsz, self.log_history_k), dtype=np.int64)
        log_history_attack = np.zeros((bsz, self.log_history_k), dtype=np.int64)
        log_history_serial = np.zeros((bsz, self.log_history_k), dtype=np.int64)
        log_history_serial2 = np.zeros((bsz, self.log_history_k), dtype=np.int64)
        log_history_from_area = np.zeros((bsz, self.log_history_k), dtype=np.int64)
        log_history_to_area = np.zeros((bsz, self.log_history_k), dtype=np.int64)
        log_history_value = np.zeros((bsz, self.log_history_k), dtype=np.float32)
        log_history_mask = np.zeros((bsz, self.log_history_k), dtype=np.float32)
        board_history_cards = np.zeros((bsz, self.board_history_k, 12), dtype=np.int64)
        board_history_feats = np.zeros(
            (bsz, self.board_history_k, self.board_history_feat_dim),
            dtype=np.float32,
        )
        board_history_mask = np.zeros((bsz, self.board_history_k), dtype=np.float32)
        history_summary = np.zeros((bsz, self.history_summary_dim), dtype=np.float32)
        actions: list[list[int]] = []
        contexts: list[int] = []
        true_first_types: list[int] = []
        true_first_cards: list[int] = []

        def _copy_1d(data: dict[str, np.ndarray], key: str, si: int,
                     dest: np.ndarray, bi: int, *, dtype=np.float32) -> bool:
            if key not in data or dest.shape[1] <= 0:
                return False
            arr = np.asarray(data[key][si], dtype=dtype).reshape(-1)
            n = min(arr.shape[0], dest.shape[1])
            if n:
                dest[bi, -n:] = arr[-n:]
            return True

        def _copy_action_prefix(data: dict[str, np.ndarray], prefix: str, si: int, bi: int,
                                type_dst: np.ndarray, card_dst: np.ndarray,
                                card2_dst: np.ndarray, attack_dst: np.ndarray,
                                context_dst: np.ndarray, select_dst: np.ndarray,
                                count_dst: np.ndarray, mask_dst: np.ndarray) -> bool:
            if type_dst.shape[1] <= 0 or f"{prefix}_type" not in data:
                return False
            _copy_1d(data, f"{prefix}_type", si, type_dst, bi, dtype=np.int64)
            _copy_1d(data, f"{prefix}_card", si, card_dst, bi, dtype=np.int64)
            _copy_1d(data, f"{prefix}_card2", si, card2_dst, bi, dtype=np.int64)
            _copy_1d(data, f"{prefix}_attack", si, attack_dst, bi, dtype=np.int64)
            _copy_1d(data, f"{prefix}_context", si, context_dst, bi, dtype=np.int64)
            _copy_1d(data, f"{prefix}_select_type", si, select_dst, bi, dtype=np.int64)
            _copy_1d(data, f"{prefix}_count", si, count_dst, bi, dtype=np.float32)
            _copy_1d(data, f"{prefix}_mask", si, mask_dst, bi, dtype=np.float32)
            return True

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
            true_first_cards.append(int(opt_card[bi, first]) if first >= 0 else -1)
            weights[bi] += self.option_weight * np.log1p(float(n))
            weights[bi] *= self.context_weights.get(contexts[-1], 1.0)
            weights[bi] *= self.type_weights.get(true_first_types[-1], 1.0)
            weights[bi] *= self.card_weights.get(true_first_cards[-1], 1.0)
            if self.opponent_deck_sig_weights:
                weights[bi] *= self.opponent_deck_sig_weights.get(str(data["opponent_deck_sig"][si]), 1.0)
            if self.opponent_archetype_weights:
                weights[bi] *= self.opponent_archetype_weights.get(
                    str(data["opponent_archetype"][si]).lower(),
                    1.0,
                )
            if len(actions[-1]) > 1:
                weights[bi] *= self.multi_select_weight
            if "won" in data:
                outcome_mask[bi] = 1.0
                if int(data["won"][si]) == 1:
                    weights[bi] *= self.win_weight
                    outcome_value[bi] = 1.0
                elif "draw" in data and int(data["draw"][si]) == 1:
                    weights[bi] *= self.draw_weight
                    outcome_value[bi] = 0.0
                else:
                    weights[bi] *= self.loss_weight
                    outcome_value[bi] = -1.0
            if self.trajectory_weights:
                key = _trajectory_game_key(data, si)
                weights[bi] *= self.trajectory_weights.get(key, self.trajectory_default_weight)
            if self.trajectory_targets:
                key = _trajectory_game_key(data, si)
                target = self.trajectory_targets.get(key)
                if target is not None:
                    n_target = min(len(target), self.trajectory_target_dim)
                    trajectory_target[bi, :n_target] = target[:n_target]
                    trajectory_mask[bi, :n_target] = 1.0
            if self.step_plan:
                plan_step_target[bi] = self.step_plan_targets[di][si]
                plan_step_mask[bi] = 1.0
            if self.history_k > 0:
                copied = _copy_action_prefix(
                    data, "own_hist", si, bi,
                    history_type, history_card, history_card2, history_attack,
                    history_context, history_select_type, history_count, history_mask,
                )
                if not copied:
                    for hi, prev_si in enumerate(self.history_prev[di][si]):
                        if prev_si < 0:
                            continue
                        (
                            history_type[bi, hi],
                            history_card[bi, hi],
                            history_card2[bi, hi],
                            history_attack[bi, hi],
                            history_context[bi, hi],
                            history_select_type[bi, hi],
                            history_count[bi, hi],
                        ) = _history_event(data, int(prev_si))
                        history_mask[bi, hi] = 1.0
            if self.opp_history_k > 0:
                _copy_action_prefix(
                    data, "opp_hist", si, bi,
                    opp_history_type, opp_history_card, opp_history_card2, opp_history_attack,
                    opp_history_context, opp_history_select_type, opp_history_count, opp_history_mask,
                )
            if self.log_history_k > 0:
                _copy_1d(data, "log_hist_type", si, log_history_type, bi, dtype=np.int64)
                _copy_1d(data, "log_hist_player", si, log_history_player, bi, dtype=np.int64)
                _copy_1d(data, "log_hist_card", si, log_history_card, bi, dtype=np.int64)
                _copy_1d(data, "log_hist_card2", si, log_history_card2, bi, dtype=np.int64)
                _copy_1d(data, "log_hist_attack", si, log_history_attack, bi, dtype=np.int64)
                _copy_1d(data, "log_hist_serial", si, log_history_serial, bi, dtype=np.int64)
                _copy_1d(data, "log_hist_serial2", si, log_history_serial2, bi, dtype=np.int64)
                _copy_1d(data, "log_hist_from_area", si, log_history_from_area, bi, dtype=np.int64)
                _copy_1d(data, "log_hist_to_area", si, log_history_to_area, bi, dtype=np.int64)
                _copy_1d(data, "log_hist_value", si, log_history_value, bi, dtype=np.float32)
                _copy_1d(data, "log_hist_mask", si, log_history_mask, bi, dtype=np.float32)
            if self.board_history_k > 0:
                if "board_hist_cards" in data:
                    arr = np.asarray(data["board_hist_cards"][si], dtype=np.int64)
                    if arr.ndim == 2:
                        n0 = min(arr.shape[0], self.board_history_k)
                        n1 = min(arr.shape[1], 12)
                        if n0 and n1:
                            board_history_cards[bi, -n0:, :n1] = arr[-n0:, :n1]
                if "board_hist_feats" in data:
                    arr = np.asarray(data["board_hist_feats"][si], dtype=np.float32)
                    if arr.ndim == 2:
                        n0 = min(arr.shape[0], self.board_history_k)
                        n1 = min(arr.shape[1], self.board_history_feat_dim)
                        if n0 and n1:
                            board_history_feats[bi, -n0:, :n1] = arr[-n0:, :n1]
                if "board_hist_mask" in data:
                    _copy_1d(data, "board_hist_mask", si, board_history_mask, bi, dtype=np.float32)
            if self.history_summary_dim > 0:
                if "history_summary" in data:
                    arr = np.asarray(data["history_summary"][si], dtype=np.float32).reshape(-1)
                    n = min(arr.shape[0], self.history_summary_dim)
                    if n:
                        history_summary[bi, :n] = arr[:n]
                else:
                    history_summary[bi] = history_summary_from_arrays(
                        own_hist={
                            "type": history_type[bi],
                            "card": history_card[bi],
                            "card2": history_card2[bi],
                            "attack": history_attack[bi],
                            "context": history_context[bi],
                            "select_type": history_select_type[bi],
                            "count": history_count[bi],
                            "mask": history_mask[bi],
                        },
                        opp_hist={
                            "type": opp_history_type[bi],
                            "card": opp_history_card[bi],
                            "card2": opp_history_card2[bi],
                            "attack": opp_history_attack[bi],
                            "context": opp_history_context[bi],
                            "select_type": opp_history_select_type[bi],
                            "count": opp_history_count[bi],
                            "mask": opp_history_mask[bi],
                        },
                        log_hist={
                            "type": log_history_type[bi],
                            "player": log_history_player[bi],
                            "card": log_history_card[bi],
                            "card2": log_history_card2[bi],
                            "attack": log_history_attack[bi],
                            "serial": log_history_serial[bi],
                            "serial2": log_history_serial2[bi],
                            "from_area": log_history_from_area[bi],
                            "to_area": log_history_to_area[bi],
                            "value": log_history_value[bi],
                            "mask": log_history_mask[bi],
                        },
                        board_hist={
                            "cards": board_history_cards[bi],
                            "feats": board_history_feats[bi],
                            "mask": board_history_mask[bi],
                        },
                        dim=self.history_summary_dim,
                    ).astype(np.float32)

        if history_augment:
            self._augment_masked_sequence(
                history_mask,
                [
                    history_type, history_card, history_card2, history_attack,
                    history_context, history_select_type, history_count,
                ],
                stream_drop_prob=history_stream_drop_prob,
                event_drop_prob=history_event_drop_prob,
                tail_drop_prob=history_tail_drop_prob,
            )
            self._augment_masked_sequence(
                opp_history_mask,
                [
                    opp_history_type, opp_history_card, opp_history_card2, opp_history_attack,
                    opp_history_context, opp_history_select_type, opp_history_count,
                ],
                stream_drop_prob=history_stream_drop_prob,
                event_drop_prob=history_event_drop_prob,
                tail_drop_prob=history_tail_drop_prob,
            )
            self._augment_masked_sequence(
                log_history_mask,
                [
                    log_history_type, log_history_player, log_history_card, log_history_card2,
                    log_history_attack, log_history_serial, log_history_serial2,
                    log_history_from_area, log_history_to_area, log_history_value,
                ],
                stream_drop_prob=history_stream_drop_prob,
                event_drop_prob=history_event_drop_prob,
                tail_drop_prob=history_tail_drop_prob,
            )
            self._augment_masked_sequence(
                board_history_mask,
                [board_history_cards, board_history_feats],
                stream_drop_prob=history_stream_drop_prob,
                event_drop_prob=history_event_drop_prob,
                tail_drop_prob=history_tail_drop_prob,
            )
            if self.history_summary_dim > 0:
                # Recompute summary after corruption so sensitivity/consistency
                # losses see the same information as the forward pass.
                for bi in range(bsz):
                    history_summary[bi] = history_summary_from_arrays(
                        own_hist={
                            "type": history_type[bi],
                            "card": history_card[bi],
                            "card2": history_card2[bi],
                            "attack": history_attack[bi],
                            "context": history_context[bi],
                            "select_type": history_select_type[bi],
                            "count": history_count[bi],
                            "mask": history_mask[bi],
                        },
                        opp_hist={
                            "type": opp_history_type[bi],
                            "card": opp_history_card[bi],
                            "card2": opp_history_card2[bi],
                            "attack": opp_history_attack[bi],
                            "context": opp_history_context[bi],
                            "select_type": opp_history_select_type[bi],
                            "count": opp_history_count[bi],
                            "mask": opp_history_mask[bi],
                        },
                        log_hist={
                            "type": log_history_type[bi],
                            "player": log_history_player[bi],
                            "card": log_history_card[bi],
                            "card2": log_history_card2[bi],
                            "attack": log_history_attack[bi],
                            "serial": log_history_serial[bi],
                            "serial2": log_history_serial2[bi],
                            "from_area": log_history_from_area[bi],
                            "to_area": log_history_to_area[bi],
                            "value": log_history_value[bi],
                            "mask": log_history_mask[bi],
                        },
                        board_hist={
                            "cards": board_history_cards[bi],
                            "feats": board_history_feats[bi],
                            "mask": board_history_mask[bi],
                        },
                        dim=self.history_summary_dim,
                    ).astype(np.float32)

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
            outcome_value=torch.as_tensor(outcome_value, device=device),
            outcome_mask=torch.as_tensor(outcome_mask, device=device),
            trajectory_target=torch.as_tensor(trajectory_target, device=device),
            trajectory_mask=torch.as_tensor(trajectory_mask, device=device),
            plan_step_target=torch.as_tensor(plan_step_target, device=device),
            plan_step_mask=torch.as_tensor(plan_step_mask, device=device),
            history={
                "type": torch.as_tensor(history_type, device=device),
                "card": torch.as_tensor(history_card, device=device),
                "card2": torch.as_tensor(history_card2, device=device),
                "attack": torch.as_tensor(history_attack, device=device),
                "context": torch.as_tensor(history_context, device=device),
                "select_type": torch.as_tensor(history_select_type, device=device),
                "count": torch.as_tensor(history_count, device=device),
                "mask": torch.as_tensor(history_mask, device=device),
                "opp_type": torch.as_tensor(opp_history_type, device=device),
                "opp_card": torch.as_tensor(opp_history_card, device=device),
                "opp_card2": torch.as_tensor(opp_history_card2, device=device),
                "opp_attack": torch.as_tensor(opp_history_attack, device=device),
                "opp_context": torch.as_tensor(opp_history_context, device=device),
                "opp_select_type": torch.as_tensor(opp_history_select_type, device=device),
                "opp_count": torch.as_tensor(opp_history_count, device=device),
                "opp_mask": torch.as_tensor(opp_history_mask, device=device),
                "log_type": torch.as_tensor(log_history_type, device=device),
                "log_player": torch.as_tensor(log_history_player, device=device),
                "log_card": torch.as_tensor(log_history_card, device=device),
                "log_card2": torch.as_tensor(log_history_card2, device=device),
                "log_attack": torch.as_tensor(log_history_attack, device=device),
                "log_serial": torch.as_tensor(log_history_serial, device=device),
                "log_serial2": torch.as_tensor(log_history_serial2, device=device),
                "log_from_area": torch.as_tensor(log_history_from_area, device=device),
                "log_to_area": torch.as_tensor(log_history_to_area, device=device),
                "log_value": torch.as_tensor(log_history_value, device=device),
                "log_mask": torch.as_tensor(log_history_mask, device=device),
                "board_cards": torch.as_tensor(board_history_cards, device=device),
                "board_feats": torch.as_tensor(board_history_feats, device=device),
                "board_mask": torch.as_tensor(board_history_mask, device=device),
                "summary": torch.as_tensor(history_summary, device=device),
            },
            actions=actions,
            n_options=n_options,
            contexts=contexts,
            true_first_types=true_first_types,
            max_options=max_options,
        )
