from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from ptcg_rl.encoder import BOARD_SLOTS, MAX_HAND, OPT_FEAT_DIM, STATE_FEAT_DIM, STATE_TOKEN_FEAT_DIM
from ptcg_rl.v15.constants import (
    DEFAULT_HISTORY_K,
    DEFAULT_MAX_OPTIONS,
    DEFAULT_PLAN_STEPS,
    EVENT_FIELDS,
    KEY_ACTION_TYPES,
    KNOWN_OPP_CARDS,
    MAX_SELECT_COUNT,
    N_ACTION_TYPES,
    TYPE_ATTACH,
    TYPE_ATTACK,
    TYPE_EVOLVE,
)

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass
class V15Batch:
    board: torch.Tensor
    hand: torch.Tensor
    feats: torch.Tensor
    state_token_feats: torch.Tensor
    event_type: torch.Tensor
    event_source: torch.Tensor
    event_owner: torch.Tensor
    event_card: torch.Tensor
    event_card2: torch.Tensor
    event_attack: torch.Tensor
    event_context: torch.Tensor
    event_select_type: torch.Tensor
    event_from_area: torch.Tensor
    event_to_area: torch.Tensor
    event_value: torch.Tensor
    event_turn_delta: torch.Tensor
    event_step_delta: torch.Tensor
    event_same_turn: torch.Tensor
    event_mask: torch.Tensor
    known_cards: torch.Tensor
    known_counts: torch.Tensor
    known_age: torch.Tensor
    known_mask: torch.Tensor
    opt_type: torch.Tensor
    opt_card: torch.Tensor
    opt_card2: torch.Tensor
    opt_attack: torch.Tensor
    opt_feats: torch.Tensor
    option_mask: torch.Tensor
    target_first: torch.Tensor
    target_multi: torch.Tensor
    target_order: torch.Tensor
    target_type: torch.Tensor
    target_context: torch.Tensor
    plan_mask: torch.Tensor
    plan_type: torch.Tensor
    plan_card: torch.Tensor
    plan_card2: torch.Tensor
    plan_attack: torch.Tensor
    plan_context: torch.Tensor
    plan_mode: torch.Tensor
    turn_continue: torch.Tensor
    turn_remaining: torch.Tensor
    block_pos: torch.Tensor
    block_len: torch.Tensor
    block_remaining: torch.Tensor
    block_type_counts: torch.Tensor
    dca_mask: torch.Tensor
    dca_group_unique: torch.Tensor
    dca_group_focus: torch.Tensor
    min_count: torch.Tensor
    max_count: torch.Tensor
    sample_weight: torch.Tensor
    won: torch.Tensor
    game_keys: list[str]

    def to(self, device: torch.device) -> "V15Batch":
        values = {}
        for key, value in self.__dict__.items():
            values[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return V15Batch(**values)


def filter_paths_by_date(paths: Iterable[str], *, date_from: str = "", date_to: str = "") -> list[str]:
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


def discover_v15_npz(
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
    return filter_paths_by_date(paths, date_from=date_from, date_to=date_to)


def _hash_unit(value: str) -> float:
    h = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") / float(1 << 64)


def split_indices_by_game(game_keys: list[str], *, val_frac: float = 0.08, seed: int = 17) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(game_keys):
        groups.setdefault(str(key), []).append(i)
    salt = f"|{int(seed)}"
    ordered = sorted(groups.items(), key=lambda kv: _hash_unit(kv[0] + salt))
    n_total = len(game_keys)
    target = max(1, int(round(n_total * float(val_frac))))
    val_groups: set[str] = set()
    val_count = 0
    for key, ids in ordered:
        if len(val_groups) >= max(len(ordered) - 1, 1):
            break
        val_groups.add(key)
        val_count += len(ids)
        if val_count >= target:
            break
    train: list[int] = []
    val: list[int] = []
    for key, ids in groups.items():
        if key in val_groups:
            val.extend(ids)
        else:
            train.extend(ids)
    train.sort()
    val.sort()
    if not val and train:
        val.append(train.pop())
    if not train and val:
        train.append(val.pop())
    return train, val


class V15RowCorpus:
    """Row-level v15 corpus with explicit event history and turn-block labels."""

    def __init__(
        self,
        paths: list[str],
        *,
        max_options: int = DEFAULT_MAX_OPTIONS,
        history_k: int = DEFAULT_HISTORY_K,
        plan_steps: int = DEFAULT_PLAN_STEPS,
        deck_sigs: Iterable[str] | None = None,
        team_names: Iterable[str] | None = None,
        opponent_archetypes: Iterable[str] | None = None,
        opponent_deck_sigs: Iterable[str] | None = None,
        winner_only: bool = False,
        min_score: float = 0.0,
        max_score: float = 0.0,
        win_weight: float = 1.0,
        loss_weight: float = 1.0,
        draw_weight: float = 1.0,
    ):
        if not paths:
            raise FileNotFoundError("No v15 corpus files found")
        self.max_options = int(max_options)
        self.history_k = int(history_k)
        self.plan_steps = int(plan_steps)
        self.deck_sigs = {str(x) for x in (deck_sigs or []) if str(x)}
        self.team_names = {str(x).lower() for x in (team_names or []) if str(x)}
        self.opponent_archetypes = {str(x).lower() for x in (opponent_archetypes or []) if str(x)}
        self.opponent_deck_sigs = {str(x) for x in (opponent_deck_sigs or []) if str(x)}
        self.winner_only = bool(winner_only)
        self.min_score = float(min_score)
        self.max_score = float(max_score)
        self.win_weight = float(win_weight)
        self.loss_weight = float(loss_weight)
        self.draw_weight = float(draw_weight)
        self.files: list[dict[str, np.ndarray]] = []
        self.rows: list[tuple[int, int]] = []
        self.game_keys: list[str] = []
        self.stats: dict[str, float | int] = {
            "files": 0,
            "raw_rows": 0,
            "kept_rows": 0,
            "deck_filtered": 0,
            "team_filtered": 0,
            "opponent_arch_filtered": 0,
            "opponent_deck_filtered": 0,
            "winner_filtered": 0,
            "score_filtered": 0,
            "win_rows": 0,
            "draw_rows": 0,
            "target_k_sum": 0,
            "multi_rows": 0,
            "option_sum": 0,
            "event_rows": 0,
            "event_slots": 0,
            "public_log_slots": 0,
            "same_turn_event_slots": 0,
            "known_rows": 0,
            "known_slots": 0,
            "plan_rows": 0,
            "plan_slots": 0,
            "plan_card_slots": 0,
            "plan_attack_slots": 0,
            "turn_continue_rows": 0,
            "dca_rows": 0,
            "dca_spread_rows": 0,
            "dca_focus_sum": 0.0,
        }
        for typ in KEY_ACTION_TYPES:
            self.stats[f"type_{typ}_rows"] = 0
        for path in paths:
            with np.load(path, allow_pickle=True) as z:
                data = {k: z[k] for k in z.files}
            fi = len(self.files)
            self.files.append(data)
            self.stats["files"] += 1
            n = len(data["board"])
            self.stats["raw_rows"] += n
            for i in range(n):
                if not self._keep(data, i):
                    continue
                self.rows.append((fi, i))
                self.game_keys.append(str(data["game_key"][i]))
                self.stats["kept_rows"] += 1
                self._add_stats(data, i)
        if not self.rows:
            raise FileNotFoundError("v15 filters kept no rows")
        self._finalize_stats()

    def __len__(self) -> int:
        return len(self.rows)

    def _keep(self, data: dict[str, np.ndarray], i: int) -> bool:
        if self.deck_sigs and str(data["deck_sig"][i]) not in self.deck_sigs:
            self.stats["deck_filtered"] += 1
            return False
        if self.team_names and str(data["team_name"][i]).lower() not in self.team_names:
            self.stats["team_filtered"] += 1
            return False
        if self.opponent_archetypes and str(data["opponent_archetype"][i]).lower() not in self.opponent_archetypes:
            self.stats["opponent_arch_filtered"] += 1
            return False
        if self.opponent_deck_sigs and str(data["opponent_deck_sig"][i]) not in self.opponent_deck_sigs:
            self.stats["opponent_deck_filtered"] += 1
            return False
        won = int(data["won"][i]) if "won" in data else int(float(data.get("reward", [0])[i]) > 0)
        if self.winner_only and not won:
            self.stats["winner_filtered"] += 1
            return False
        score = float(data["score"][i]) if "score" in data else 0.0
        if self.min_score and score < self.min_score:
            self.stats["score_filtered"] += 1
            return False
        if self.max_score and score > self.max_score:
            self.stats["score_filtered"] += 1
            return False
        return True

    def _add_stats(self, data: dict[str, np.ndarray], i: int) -> None:
        action = np.asarray(data["action"][i], dtype=np.int64).reshape(-1)
        opt_type = np.asarray(data["ot"][i], dtype=np.int64).reshape(-1)
        plan_mask = np.asarray(data["plan_mask"][i], dtype=np.float32).reshape(-1)
        event_mask = np.asarray(data["event_mask"][i], dtype=np.float32).reshape(-1)
        event_source = np.asarray(data["event_source"][i], dtype=np.int64).reshape(-1)
        known_mask = np.asarray(data["known_mask"][i], dtype=np.float32).reshape(-1)
        self.stats["target_k_sum"] += int(len(action))
        self.stats["multi_rows"] += int(len(action) > 1)
        self.stats["option_sum"] += int(len(opt_type))
        self.stats["event_rows"] += int(event_mask.sum() > 0)
        self.stats["event_slots"] += int(event_mask.sum())
        self.stats["public_log_slots"] += int(((event_source == 2) & (event_mask > 0)).sum())
        if "event_same_turn" in data:
            same = np.asarray(data["event_same_turn"][i], dtype=np.float32).reshape(-1)
            self.stats["same_turn_event_slots"] += int(((same > 0) & (event_mask > 0)).sum())
        self.stats["known_rows"] += int(known_mask.sum() > 0)
        self.stats["known_slots"] += int(known_mask.sum())
        self.stats["plan_rows"] += int(plan_mask.sum() > 0)
        self.stats["plan_slots"] += int(plan_mask.sum())
        self.stats["plan_card_slots"] += int((np.asarray(data["plan_card"][i]).reshape(-1) > 0).sum())
        self.stats["plan_attack_slots"] += int((np.asarray(data["plan_attack"][i]).reshape(-1) > 0).sum())
        self.stats["turn_continue_rows"] += int(float(data["turn_continue"][i]) > 0)
        dca = int(data["dca_mask"][i]) if "dca_mask" in data else 0
        self.stats["dca_rows"] += dca
        if dca:
            self.stats["dca_focus_sum"] += float(data["dca_group_focus"][i])
            self.stats["dca_spread_rows"] += int(float(data["dca_group_focus"][i]) < 0.999)
        won = int(data["won"][i]) if "won" in data else 0
        draw = int(data["draw"][i]) if "draw" in data else 0
        self.stats["win_rows"] += won
        self.stats["draw_rows"] += draw
        typ = int(data["act_type"][i]) if "act_type" in data else -1
        if f"type_{typ}_rows" in self.stats:
            self.stats[f"type_{typ}_rows"] += 1

    def _finalize_stats(self) -> None:
        n = max(int(self.stats["kept_rows"]), 1)
        self.stats["win_rate"] = float(self.stats["win_rows"]) / n
        self.stats["draw_rate"] = float(self.stats["draw_rows"]) / n
        self.stats["target_k_mean"] = float(self.stats["target_k_sum"]) / n
        self.stats["multi_rate"] = float(self.stats["multi_rows"]) / n
        self.stats["option_mean"] = float(self.stats["option_sum"]) / n
        self.stats["event_rate"] = float(self.stats["event_rows"]) / n
        self.stats["event_slots_mean"] = float(self.stats["event_slots"]) / n
        self.stats["public_log_slots_mean"] = float(self.stats["public_log_slots"]) / n
        self.stats["same_turn_event_slots_mean"] = float(self.stats["same_turn_event_slots"]) / n
        self.stats["known_rate"] = float(self.stats["known_rows"]) / n
        self.stats["known_slots_mean"] = float(self.stats["known_slots"]) / n
        self.stats["plan_rate"] = float(self.stats["plan_rows"]) / n
        self.stats["plan_slots_mean"] = float(self.stats["plan_slots"]) / n
        self.stats["plan_card_rate"] = float(self.stats["plan_card_slots"]) / max(float(self.stats["plan_slots"]), 1.0)
        self.stats["plan_attack_rate"] = float(self.stats["plan_attack_slots"]) / max(float(self.stats["plan_slots"]), 1.0)
        self.stats["turn_continue_rate"] = float(self.stats["turn_continue_rows"]) / n
        self.stats["dca_rate"] = float(self.stats["dca_rows"]) / n
        self.stats["dca_spread_rate"] = float(self.stats["dca_spread_rows"]) / max(float(self.stats["dca_rows"]), 1.0)
        self.stats["dca_focus_mean"] = float(self.stats["dca_focus_sum"]) / max(float(self.stats["dca_rows"]), 1.0)
        for typ in KEY_ACTION_TYPES:
            self.stats[f"type_{typ}_rate"] = float(self.stats[f"type_{typ}_rows"]) / n

    def sample_weight_at(self, data: dict[str, np.ndarray], i: int) -> float:
        won = int(data["won"][i]) if "won" in data else 0
        draw = int(data["draw"][i]) if "draw" in data else 0
        if draw:
            return self.draw_weight
        return self.win_weight if won else self.loss_weight

    def get(self, sample_id: int) -> dict[str, np.ndarray | float | int | str]:
        fi, i = self.rows[int(sample_id)]
        data = self.files[fi]
        out: dict[str, np.ndarray | float | int | str] = {}
        for name in (
            "board",
            "hand",
            "feats",
            "state_token_feats",
            "known_cards",
            "known_counts",
            "known_age",
            "known_mask",
            "plan_mask",
            "plan_type",
            "plan_card",
            "plan_card2",
            "plan_attack",
            "plan_context",
            "block_type_counts",
        ):
            out[name] = np.asarray(data[name][i])
        for name in (
            "event_type",
            "event_source",
            "event_owner",
            "event_card",
            "event_card2",
            "event_attack",
            "event_context",
            "event_select_type",
            "event_from_area",
            "event_to_area",
            "event_value",
            "event_turn_delta",
            "event_step_delta",
            "event_same_turn",
            "event_mask",
        ):
            out[name] = np.asarray(data[name][i])
        out["ot"] = np.asarray(data["ot"][i], dtype=np.int64)
        out["oc"] = np.asarray(data["oc"][i], dtype=np.int64)
        out["oc2"] = np.asarray(data["oc2"][i], dtype=np.int64)
        out["oa"] = np.asarray(data["oa"][i], dtype=np.int64)
        out["of_arr"] = np.asarray(data["of_arr"][i], dtype=np.float32)
        out["action"] = np.asarray(data["action"][i], dtype=np.int64)
        for name in (
            "act_type",
            "act_context",
            "plan_mode",
            "turn_continue",
            "turn_remaining",
            "block_pos",
            "block_len",
            "block_remaining",
            "dca_mask",
            "dca_group_unique",
            "min_c",
            "max_c",
            "won",
        ):
            out[name] = int(data[name][i])
        out["dca_group_focus"] = float(data["dca_group_focus"][i])
        out["sample_weight"] = float(self.sample_weight_at(data, i))
        out["game_key"] = str(data["game_key"][i])
        return out

    def make_batch(self, ids: list[int]) -> V15Batch:
        rows = [self.get(i) for i in ids]
        b = len(rows)
        max_options = self.max_options
        hist_k = self.history_k
        plan_steps = self.plan_steps
        opt_type = np.zeros((b, max_options), dtype=np.int64)
        opt_card = np.zeros((b, max_options), dtype=np.int64)
        opt_card2 = np.zeros((b, max_options), dtype=np.int64)
        opt_attack = np.zeros((b, max_options), dtype=np.int64)
        opt_feats = np.zeros((b, max_options, OPT_FEAT_DIM), dtype=np.float32)
        option_mask = np.zeros((b, max_options), dtype=np.float32)
        target_multi = np.zeros((b, max_options), dtype=np.float32)
        target_order = np.full((b, MAX_SELECT_COUNT), -1, dtype=np.int64)
        target_first = np.full(b, -1, dtype=np.int64)
        for bi, row in enumerate(rows):
            n = min(len(row["ot"]), max_options)
            if n:
                opt_type[bi, :n] = np.asarray(row["ot"], dtype=np.int64)[:n]
                opt_card[bi, :n] = np.asarray(row["oc"], dtype=np.int64)[:n]
                opt_card2[bi, :n] = np.asarray(row["oc2"], dtype=np.int64)[:n]
                opt_attack[bi, :n] = np.asarray(row["oa"], dtype=np.int64)[:n]
                feats = np.asarray(row["of_arr"], dtype=np.float32)
                opt_feats[bi, : min(n, feats.shape[0]), : min(OPT_FEAT_DIM, feats.shape[1])] = feats[:n, :OPT_FEAT_DIM]
                option_mask[bi, :n] = 1.0
            action = np.asarray(row["action"], dtype=np.int64).reshape(-1)
            valid_action = [int(a) for a in action if 0 <= int(a) < max_options]
            if valid_action:
                target_first[bi] = valid_action[0]
                for pos, a in enumerate(valid_action[:MAX_SELECT_COUNT]):
                    target_multi[bi, a] = 1.0
                    target_order[bi, pos] = a

        def stack(name: str, dtype=np.float32):
            return torch.as_tensor(np.stack([np.asarray(r[name]) for r in rows]), dtype=dtype)

        return V15Batch(
            board=stack("board", torch.long),
            hand=stack("hand", torch.long),
            feats=stack("feats", torch.float32),
            state_token_feats=stack("state_token_feats", torch.float32),
            event_type=stack("event_type", torch.long)[:, -hist_k:],
            event_source=stack("event_source", torch.long)[:, -hist_k:],
            event_owner=stack("event_owner", torch.long)[:, -hist_k:],
            event_card=stack("event_card", torch.long)[:, -hist_k:],
            event_card2=stack("event_card2", torch.long)[:, -hist_k:],
            event_attack=stack("event_attack", torch.long)[:, -hist_k:],
            event_context=stack("event_context", torch.long)[:, -hist_k:],
            event_select_type=stack("event_select_type", torch.long)[:, -hist_k:],
            event_from_area=stack("event_from_area", torch.long)[:, -hist_k:],
            event_to_area=stack("event_to_area", torch.long)[:, -hist_k:],
            event_value=stack("event_value", torch.float32)[:, -hist_k:],
            event_turn_delta=stack("event_turn_delta", torch.float32)[:, -hist_k:],
            event_step_delta=stack("event_step_delta", torch.float32)[:, -hist_k:],
            event_same_turn=stack("event_same_turn", torch.float32)[:, -hist_k:],
            event_mask=stack("event_mask", torch.float32)[:, -hist_k:],
            known_cards=stack("known_cards", torch.long),
            known_counts=stack("known_counts", torch.float32),
            known_age=stack("known_age", torch.float32),
            known_mask=stack("known_mask", torch.float32),
            opt_type=torch.as_tensor(opt_type, dtype=torch.long),
            opt_card=torch.as_tensor(opt_card, dtype=torch.long),
            opt_card2=torch.as_tensor(opt_card2, dtype=torch.long),
            opt_attack=torch.as_tensor(opt_attack, dtype=torch.long),
            opt_feats=torch.as_tensor(opt_feats, dtype=torch.float32),
            option_mask=torch.as_tensor(option_mask, dtype=torch.float32),
            target_first=torch.as_tensor(target_first, dtype=torch.long),
            target_multi=torch.as_tensor(target_multi, dtype=torch.float32),
            target_order=torch.as_tensor(target_order, dtype=torch.long),
            target_type=torch.as_tensor([int(r["act_type"]) for r in rows], dtype=torch.long),
            target_context=torch.as_tensor([int(r["act_context"]) for r in rows], dtype=torch.long),
            plan_mask=stack("plan_mask", torch.float32)[:, :plan_steps],
            plan_type=stack("plan_type", torch.long)[:, :plan_steps],
            plan_card=stack("plan_card", torch.long)[:, :plan_steps],
            plan_card2=stack("plan_card2", torch.long)[:, :plan_steps],
            plan_attack=stack("plan_attack", torch.long)[:, :plan_steps],
            plan_context=stack("plan_context", torch.long)[:, :plan_steps],
            plan_mode=torch.as_tensor([int(r["plan_mode"]) for r in rows], dtype=torch.long),
            turn_continue=torch.as_tensor([int(r["turn_continue"]) for r in rows], dtype=torch.float32),
            turn_remaining=torch.as_tensor([int(r["turn_remaining"]) for r in rows], dtype=torch.float32),
            block_pos=torch.as_tensor([int(r["block_pos"]) for r in rows], dtype=torch.long),
            block_len=torch.as_tensor([int(r["block_len"]) for r in rows], dtype=torch.float32),
            block_remaining=torch.as_tensor([int(r["block_remaining"]) for r in rows], dtype=torch.float32),
            block_type_counts=stack("block_type_counts", torch.float32),
            dca_mask=torch.as_tensor([int(r["dca_mask"]) for r in rows], dtype=torch.float32),
            dca_group_unique=torch.as_tensor([int(r["dca_group_unique"]) for r in rows], dtype=torch.float32),
            dca_group_focus=torch.as_tensor([float(r["dca_group_focus"]) for r in rows], dtype=torch.float32),
            min_count=torch.as_tensor([int(r["min_c"]) for r in rows], dtype=torch.long),
            max_count=torch.as_tensor([int(r["max_c"]) for r in rows], dtype=torch.long),
            sample_weight=torch.as_tensor([float(r["sample_weight"]) for r in rows], dtype=torch.float32),
            won=torch.as_tensor([int(r["won"]) for r in rows], dtype=torch.float32),
            game_keys=[str(r["game_key"]) for r in rows],
        )


def signal_stats_line(stats: dict[str, float | int]) -> str:
    keys = [
        ("kept_rows", 0),
        ("win_rate", 3),
        ("target_k_mean", 3),
        ("multi_rate", 3),
        ("option_mean", 2),
        ("event_rate", 3),
        ("event_slots_mean", 2),
        ("public_log_slots_mean", 2),
        ("same_turn_event_slots_mean", 2),
        ("known_rate", 3),
        ("known_slots_mean", 2),
        ("plan_rate", 3),
        ("plan_slots_mean", 2),
        ("plan_card_rate", 3),
        ("plan_attack_rate", 3),
        ("turn_continue_rate", 3),
        ("dca_rate", 3),
        ("dca_spread_rate", 3),
        ("dca_focus_mean", 3),
    ]
    parts = []
    for key, digits in keys:
        value = stats.get(key, 0)
        if digits:
            parts.append(f"{key}={float(value):.{digits}f}")
        else:
            parts.append(f"{key}={int(value)}")
    type_parts = []
    for typ in KEY_ACTION_TYPES:
        type_parts.append(f"{typ}:{float(stats.get(f'type_{typ}_rate', 0.0)):.2f}")
    return "Signal stats: " + " ".join(parts) + " type_rates=" + ",".join(type_parts)


def signal_warnings(stats: dict[str, float | int]) -> list[str]:
    warnings: list[str] = []
    if float(stats.get("event_slots_mean", 0.0)) < 2.0:
        warnings.append("history_too_sparse")
    if float(stats.get("plan_rate", 0.0)) < 0.20:
        warnings.append("plan_labels_sparse")
    if float(stats.get("turn_continue_rate", 0.0)) < 0.20:
        warnings.append("same_turn_continue_sparse")
    if float(stats.get("known_rate", 0.0)) == 0.0:
        warnings.append("known_info_zero")
    if float(stats.get("multi_rate", 0.0)) < 0.02:
        warnings.append("multi_select_rare")
    return warnings
