from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from ptcg_rl.encoder import BOARD_SLOTS, MAX_HAND, OPT_FEAT_DIM, STATE_FEAT_DIM, STATE_TOKEN_FEAT_DIM
from ptcg_rl.seq.constants import DAMAGE_COUNTER_ANY_CONTEXT, FUTURE_PLAN_DIM, KNOWN_OPP_CARDS, LEDGER_FEAT_DIM, MAX_SELECT_COUNT, TYPE_END
from ptcg_rl.seq.constants import TYPE_ABILITY, TYPE_ATTACH, TYPE_ATTACK, TYPE_EVOLVE, TYPE_PLAY, TYPE_RETREAT

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

_STAT_TYPES = (
    (TYPE_PLAY, "play"),
    (TYPE_ATTACH, "attach"),
    (TYPE_EVOLVE, "evolve"),
    (TYPE_ABILITY, "ability"),
    (TYPE_RETREAT, "retreat"),
    (TYPE_ATTACK, "attack"),
    (TYPE_END, "end"),
)


@dataclass
class SequenceBatch:
    board: torch.Tensor
    hand: torch.Tensor
    feats: torch.Tensor
    state_token_feats: torch.Tensor
    ledger_feats: torch.Tensor
    known_opp_cards: torch.Tensor
    known_opp_counts: torch.Tensor
    known_opp_mask: torch.Tensor
    prev_type: torch.Tensor
    prev_card: torch.Tensor
    prev_card2: torch.Tensor
    prev_attack: torch.Tensor
    prev_context: torch.Tensor
    prev_select_type: torch.Tensor
    prev_count: torch.Tensor
    opt_type: torch.Tensor
    opt_card: torch.Tensor
    opt_card2: torch.Tensor
    opt_attack: torch.Tensor
    opt_feats: torch.Tensor
    option_mask: torch.Tensor
    target_first: torch.Tensor
    target_order: torch.Tensor
    target_multi: torch.Tensor
    target_type: torch.Tensor
    target_context: torch.Tensor
    dca_group_index: torch.Tensor
    dca_pos: torch.Tensor
    dca_len: torch.Tensor
    dca_remaining: torch.Tensor
    dca_prior_same_slot: torch.Tensor
    dca_prior_unique_slots: torch.Tensor
    dca_prior_max_repeat: torch.Tensor
    dca_group_unique_slots: torch.Tensor
    dca_group_focus_frac: torch.Tensor
    min_count: torch.Tensor
    max_count: torch.Tensor
    step_mask: torch.Tensor
    sample_weight: torch.Tensor
    future_plan: torch.Tensor
    outcome: torch.Tensor
    game_keys: list[str]
    row_refs: list[list[tuple[int, int]]]

    def to(self, device: torch.device) -> "SequenceBatch":
        values = {}
        for key, value in self.__dict__.items():
            values[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return SequenceBatch(**values)


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


def discover_sequence_npz(
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


class SequenceCorpus:
    """NPZ-backed game/window corpus for v14 sequence policy training."""

    def __init__(
        self,
        paths: list[str],
        *,
        seq_len: int = 32,
        stride: int = 1,
        state_feat_dim: int = STATE_FEAT_DIM,
        opt_feat_dim: int = OPT_FEAT_DIM,
        state_token_feat_dim: int = STATE_TOKEN_FEAT_DIM,
        ledger_feat_dim: int = LEDGER_FEAT_DIM,
        future_plan_dim: int = FUTURE_PLAN_DIM,
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
        min_game_decisions: int = 2,
    ):
        if not paths:
            raise FileNotFoundError("No v14 sequence corpus .npz files found")
        self.seq_len = max(1, int(seq_len))
        self.stride = max(1, int(stride))
        self.state_feat_dim = int(state_feat_dim)
        self.opt_feat_dim = int(opt_feat_dim)
        self.state_token_feat_dim = int(state_token_feat_dim)
        self.ledger_feat_dim = int(ledger_feat_dim)
        self.future_plan_dim = int(future_plan_dim)
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
        self.games: list[list[tuple[int, int]]] = []
        self.samples: list[tuple[int, int]] = []
        self._game_lengths: list[int] = []
        self.stats = {
            "files": 0,
            "raw_rows": 0,
            "kept_rows": 0,
            "games": 0,
            "samples": 0,
            "deck_filtered": 0,
            "team_filtered": 0,
            "opponent_arch_filtered": 0,
            "opponent_deck_filtered": 0,
            "winner_filtered": 0,
            "score_filtered": 0,
            "target_rows": 0,
            "target_k_sum": 0,
            "multi_target_rows": 0,
            "capable_multi_rows": 0,
            "dca_rows": 0,
            "dca_groups": 0,
            "dca_spread_rows": 0,
            "dca_focus_sum": 0.0,
            "dca_unique_sum": 0,
            "known_opp_rows": 0,
            "known_opp_slots": 0.0,
            "known_opp_count_sum": 0.0,
            "known_opp_max_slots": 0.0,
            "plan_rows": 0,
            "plan_pos_sum": 0,
            "plan_value_sum": 0.0,
            "win_rows": 0,
        }
        for _, name in _STAT_TYPES:
            self.stats[f"type_{name}_rows"] = 0

        for path in paths:
            with np.load(path, allow_pickle=True) as z:
                data = {k: z[k] for k in z.files}
            self.stats["files"] += 1
            n_rows = len(data["board"])
            self.stats["raw_rows"] += n_rows
            di = len(self.files)
            game_rows: dict[str, list[int]] = {}
            for i in range(n_rows):
                if not self._keep(data, i):
                    continue
                key = str(data["game_key"][i]) if "game_key" in data else f"{data['episode_id'][i]}:{data['player_index'][i]}"
                game_rows.setdefault(key, []).append(i)
                self.stats["kept_rows"] += 1
                self._add_signal_stats(data, i)
            self.files.append(data)
            for rows in game_rows.values():
                rows = sorted(rows, key=lambda r: int(data["decision_index"][r]) if "decision_index" in data else r)
                if len(rows) < int(min_game_decisions):
                    continue
                self._game_lengths.append(len(rows))
                gi = len(self.games)
                refs = [(di, int(r)) for r in rows]
                self.games.append(refs)
                for end_pos in range(len(refs)):
                    if end_pos % self.stride == 0 or end_pos == len(refs) - 1:
                        self.samples.append((gi, end_pos))
        self.stats["games"] = len(self.games)
        self.stats["samples"] = len(self.samples)
        if not self.samples:
            raise FileNotFoundError("v14 sequence corpus filters kept no samples")
        self._finalize_signal_stats()

    def _keep(self, data: dict[str, np.ndarray], i: int) -> bool:
        if self.deck_sigs and str(data.get("deck_sig", [""])[i]) not in self.deck_sigs:
            self.stats["deck_filtered"] += 1
            return False
        if self.team_names and str(data.get("team_name", [""])[i]).lower() not in self.team_names:
            self.stats["team_filtered"] += 1
            return False
        if self.opponent_archetypes and str(data.get("opponent_archetype", [""])[i]).lower() not in self.opponent_archetypes:
            self.stats["opponent_arch_filtered"] += 1
            return False
        if self.opponent_deck_sigs and str(data.get("opponent_deck_sig", [""])[i]) not in self.opponent_deck_sigs:
            self.stats["opponent_deck_filtered"] += 1
            return False
        if self.winner_only and int(data.get("won", np.zeros(len(data["board"]), dtype=np.int8))[i]) != 1:
            self.stats["winner_filtered"] += 1
            return False
        if self.min_score > 0.0 or self.max_score > 0.0:
            score = float(data.get("score", np.zeros(len(data["board"]), dtype=np.float32))[i])
            if self.min_score > 0.0 and score < self.min_score:
                self.stats["score_filtered"] += 1
                return False
            if self.max_score > 0.0 and score > self.max_score:
                self.stats["score_filtered"] += 1
                return False
        return True

    def _add_signal_stats(self, data: dict[str, np.ndarray], i: int) -> None:
        action = np.asarray(data.get("action", [])[i], dtype=np.int64).reshape(-1)
        opt_n = len(data.get("ot", [])[i]) if "ot" in data else 0
        valid = action[(action >= 0) & (action < opt_n)]
        target_k = int(valid.size)
        if target_k > 0:
            self.stats["target_rows"] += 1
            self.stats["target_k_sum"] += target_k
            if target_k > 1:
                self.stats["multi_target_rows"] += 1
        if int(data.get("max_c", np.zeros(len(data["board"]), dtype=np.int16))[i]) > 1:
            self.stats["capable_multi_rows"] += 1
        typ = int(data.get("act_type", np.zeros(len(data["board"]), dtype=np.int16))[i])
        for type_id, name in _STAT_TYPES:
            if typ == type_id:
                self.stats[f"type_{name}_rows"] += 1
                break
        if int(data.get("won", np.zeros(len(data["board"]), dtype=np.int8))[i]) == 1:
            self.stats["win_rows"] += 1
        if "future_plan" in data:
            plan = np.asarray(data["future_plan"][i], dtype=np.float32).reshape(-1)
            if plan.size:
                plan01 = np.clip(plan, 0.0, 1.0)
                self.stats["plan_rows"] += 1
                self.stats["plan_pos_sum"] += int((plan01 > 0.05).sum())
                self.stats["plan_value_sum"] += float(plan01.mean())
        if "known_opp_mask" in data:
            mask = np.asarray(data["known_opp_mask"][i], dtype=np.float32).reshape(-1)
            counts = np.asarray(data.get("known_opp_counts", np.zeros(len(data["board"]), dtype=object))[i], dtype=np.float32).reshape(-1)
            slots = float((mask > 0).sum())
            if slots > 0:
                self.stats["known_opp_rows"] += 1
                self.stats["known_opp_slots"] += slots
                self.stats["known_opp_count_sum"] += float((np.clip(counts, 0.0, 1.0) * mask).sum())
                self.stats["known_opp_max_slots"] = max(float(self.stats["known_opp_max_slots"]), slots)
        ctx = int(data.get("act_context", np.zeros(len(data["board"]), dtype=np.int16))[i])
        if ctx == DAMAGE_COUNTER_ANY_CONTEXT:
            self.stats["dca_rows"] += 1
            if int(data.get("dca_pos", np.full(len(data["board"]), -1, dtype=np.int16))[i]) == 0:
                self.stats["dca_groups"] += 1
            unique = int(data.get("dca_group_unique_slots", np.zeros(len(data["board"]), dtype=np.int16))[i])
            if unique > 1:
                self.stats["dca_spread_rows"] += 1
            self.stats["dca_unique_sum"] += unique
            self.stats["dca_focus_sum"] += float(data.get("dca_group_focus_frac", np.zeros(len(data["board"]), dtype=np.float16))[i])

    def _finalize_signal_stats(self) -> None:
        kept = max(float(self.stats.get("kept_rows", 0)), 1.0)
        target = max(float(self.stats.get("target_rows", 0)), 1.0)
        dca = max(float(self.stats.get("dca_rows", 0)), 1.0)
        self.stats["target_k_mean"] = float(self.stats.get("target_k_sum", 0)) / target
        self.stats["multi_target_rate"] = float(self.stats.get("multi_target_rows", 0)) / target
        self.stats["capable_multi_rate"] = float(self.stats.get("capable_multi_rows", 0)) / kept
        self.stats["dca_rate"] = float(self.stats.get("dca_rows", 0)) / kept
        self.stats["dca_spread_rate"] = float(self.stats.get("dca_spread_rows", 0)) / dca
        self.stats["dca_focus_mean"] = float(self.stats.get("dca_focus_sum", 0.0)) / dca
        self.stats["dca_unique_mean"] = float(self.stats.get("dca_unique_sum", 0)) / dca
        self.stats["win_rate"] = float(self.stats.get("win_rows", 0)) / kept
        plan_rows = max(float(self.stats.get("plan_rows", 0)), 1.0)
        self.stats["plan_label_density"] = float(self.stats.get("plan_pos_sum", 0)) / (plan_rows * max(self.future_plan_dim, 1))
        self.stats["plan_value_mean"] = float(self.stats.get("plan_value_sum", 0.0)) / plan_rows
        known_rows = max(float(self.stats.get("known_opp_rows", 0)), 1.0)
        self.stats["known_opp_rate"] = float(self.stats.get("known_opp_rows", 0)) / kept
        self.stats["known_opp_slots_mean"] = float(self.stats.get("known_opp_slots", 0.0)) / known_rows
        self.stats["known_opp_count_mean"] = float(self.stats.get("known_opp_count_sum", 0.0)) / known_rows
        if self._game_lengths:
            lengths = np.asarray(self._game_lengths, dtype=np.float32)
            self.stats["game_len_mean"] = float(lengths.mean())
            self.stats["game_len_p50"] = float(np.percentile(lengths, 50))
            self.stats["game_len_p90"] = float(np.percentile(lengths, 90))
        else:
            self.stats["game_len_mean"] = 0.0
            self.stats["game_len_p50"] = 0.0
            self.stats["game_len_p90"] = 0.0
        for _, name in _STAT_TYPES:
            self.stats[f"type_{name}_rate"] = float(self.stats.get(f"type_{name}_rows", 0)) / target

    def split_samples(self, val_fraction: float = 0.1, seed: int = 7) -> tuple[list[int], list[int]]:
        rng = np.random.default_rng(seed)
        game_order = np.arange(len(self.games))
        rng.shuffle(game_order)
        if len(game_order) <= 1:
            ids = np.arange(len(self.samples))
            rng.shuffle(ids)
            n_val = max(1, int(round(len(ids) * val_fraction)))
            return ids[n_val:].tolist(), ids[:n_val].tolist()
        split = int(round(len(game_order) * (1.0 - val_fraction)))
        split = min(max(split, 1), len(game_order) - 1)
        train_games = set(int(x) for x in game_order[:split])
        train: list[int] = []
        val: list[int] = []
        for si, (gi, _) in enumerate(self.samples):
            (train if gi in train_games else val).append(si)
        return train, val

    def collate(self, sample_ids: list[int]) -> SequenceBatch:
        bsz = len(sample_ids)
        windows: list[list[tuple[int, int]]] = []
        max_options = 1
        for sample_id in sample_ids:
            gi, end_pos = self.samples[int(sample_id)]
            refs = self.games[gi][max(0, end_pos - self.seq_len + 1): end_pos + 1]
            windows.append(refs)
            for di, ri in refs:
                max_options = max(max_options, len(self.files[di]["ot"][ri]))

        shape_bt = (bsz, self.seq_len)
        board = np.zeros((*shape_bt, BOARD_SLOTS), dtype=np.int64)
        hand = np.zeros((*shape_bt, MAX_HAND), dtype=np.int64)
        feats = np.zeros((*shape_bt, self.state_feat_dim), dtype=np.float32)
        state_token_feats = np.zeros((*shape_bt, BOARD_SLOTS + MAX_HAND, self.state_token_feat_dim), dtype=np.float32)
        ledger_feats = np.zeros((*shape_bt, self.ledger_feat_dim), dtype=np.float32)
        known_opp_cards = np.zeros((*shape_bt, KNOWN_OPP_CARDS), dtype=np.int64)
        known_opp_counts = np.zeros((*shape_bt, KNOWN_OPP_CARDS), dtype=np.float32)
        known_opp_mask = np.zeros((*shape_bt, KNOWN_OPP_CARDS), dtype=np.float32)
        prev_type = np.zeros(shape_bt, dtype=np.int64)
        prev_card = np.zeros(shape_bt, dtype=np.int64)
        prev_card2 = np.zeros(shape_bt, dtype=np.int64)
        prev_attack = np.zeros(shape_bt, dtype=np.int64)
        prev_context = np.zeros(shape_bt, dtype=np.int64)
        prev_select_type = np.zeros(shape_bt, dtype=np.int64)
        prev_count = np.zeros(shape_bt, dtype=np.float32)
        opt_type = np.zeros((*shape_bt, max_options), dtype=np.int64)
        opt_card = np.zeros((*shape_bt, max_options), dtype=np.int64)
        opt_card2 = np.zeros((*shape_bt, max_options), dtype=np.int64)
        opt_attack = np.zeros((*shape_bt, max_options), dtype=np.int64)
        opt_feats = np.zeros((*shape_bt, max_options, self.opt_feat_dim), dtype=np.float32)
        option_mask = np.zeros((*shape_bt, max_options), dtype=np.float32)
        target_first = np.full(shape_bt, -1, dtype=np.int64)
        target_order = np.full((*shape_bt, MAX_SELECT_COUNT), -1, dtype=np.int64)
        target_multi = np.zeros((*shape_bt, max_options), dtype=np.float32)
        target_type = np.zeros(shape_bt, dtype=np.int64)
        target_context = np.zeros(shape_bt, dtype=np.int64)
        dca_group_index = np.full(shape_bt, -1, dtype=np.int64)
        dca_pos = np.full(shape_bt, -1, dtype=np.int64)
        dca_len = np.zeros(shape_bt, dtype=np.int64)
        dca_remaining = np.zeros(shape_bt, dtype=np.int64)
        dca_prior_same_slot = np.zeros(shape_bt, dtype=np.int64)
        dca_prior_unique_slots = np.zeros(shape_bt, dtype=np.int64)
        dca_prior_max_repeat = np.zeros(shape_bt, dtype=np.int64)
        dca_group_unique_slots = np.zeros(shape_bt, dtype=np.int64)
        dca_group_focus_frac = np.zeros(shape_bt, dtype=np.float32)
        min_count = np.zeros(shape_bt, dtype=np.int64)
        max_count = np.zeros(shape_bt, dtype=np.int64)
        step_mask = np.zeros(shape_bt, dtype=np.float32)
        sample_weight = np.ones(shape_bt, dtype=np.float32)
        future_plan = np.zeros((*shape_bt, self.future_plan_dim), dtype=np.float32)
        outcome = np.zeros(shape_bt, dtype=np.float32)
        game_keys: list[str] = []

        for bi, refs in enumerate(windows):
            offset = self.seq_len - len(refs)
            if refs:
                di0, ri0 = refs[-1]
                game_keys.append(str(self.files[di0]["game_key"][ri0]) if "game_key" in self.files[di0] else "")
            else:
                game_keys.append("")
            for local_t, (di, ri) in enumerate(refs, offset):
                data = self.files[di]
                board[bi, local_t] = _fit_1d(data["board"][ri], BOARD_SLOTS, dtype=np.int64)
                hand[bi, local_t] = _fit_1d(data["hand"][ri], MAX_HAND, dtype=np.int64)
                feats[bi, local_t] = _fit_1d(data["feats"][ri], self.state_feat_dim, dtype=np.float32)
                if self.state_token_feat_dim > 0 and "state_token_feats" in data:
                    state_token_feats[bi, local_t] = _fit_2d(
                        data["state_token_feats"][ri],
                        BOARD_SLOTS + MAX_HAND,
                        self.state_token_feat_dim,
                    )
                if "ledger_feats" in data:
                    ledger_feats[bi, local_t] = _fit_1d(data["ledger_feats"][ri], self.ledger_feat_dim, dtype=np.float32)
                if "known_opp_cards" in data:
                    known_opp_cards[bi, local_t] = _fit_1d(data["known_opp_cards"][ri], KNOWN_OPP_CARDS, dtype=np.int64)
                if "known_opp_counts" in data:
                    known_opp_counts[bi, local_t] = _fit_1d(data["known_opp_counts"][ri], KNOWN_OPP_CARDS, dtype=np.float32)
                if "known_opp_mask" in data:
                    known_opp_mask[bi, local_t] = _fit_1d(data["known_opp_mask"][ri], KNOWN_OPP_CARDS, dtype=np.float32)
                prev_type[bi, local_t] = int(_get_row(data, "prev_type", ri, 0))
                prev_card[bi, local_t] = int(_get_row(data, "prev_card", ri, 0))
                prev_card2[bi, local_t] = int(_get_row(data, "prev_card2", ri, 0))
                prev_attack[bi, local_t] = int(_get_row(data, "prev_attack", ri, 0))
                prev_context[bi, local_t] = int(_get_row(data, "prev_context", ri, 0))
                prev_select_type[bi, local_t] = int(_get_row(data, "prev_select_type", ri, 0))
                prev_count[bi, local_t] = float(_get_row(data, "prev_count", ri, 0.0))

                ot = np.asarray(data["ot"][ri], dtype=np.int64).reshape(-1)
                nopt = len(ot)
                opt_type[bi, local_t, :nopt] = ot
                opt_card[bi, local_t, :nopt] = _fit_1d(data["oc"][ri], nopt, dtype=np.int64)
                opt_card2[bi, local_t, :nopt] = _fit_1d(data["oc2"][ri], nopt, dtype=np.int64)
                opt_attack[bi, local_t, :nopt] = _fit_1d(data["oa"][ri], nopt, dtype=np.int64)
                opt_feats[bi, local_t, :nopt] = _fit_2d(data["of_arr"][ri], nopt, self.opt_feat_dim)
                option_mask[bi, local_t, :nopt] = 1.0

                action = np.asarray(data["action"][ri], dtype=np.int64).reshape(-1)
                valid = action[(action >= 0) & (action < nopt)]
                if valid.size:
                    target_first[bi, local_t] = int(valid[0])
                    upto = min(valid.size, MAX_SELECT_COUNT)
                    target_order[bi, local_t, :upto] = valid[:upto]
                    target_multi[bi, local_t, valid] = 1.0
                    target_type[bi, local_t] = int(ot[int(valid[0])])
                else:
                    target_type[bi, local_t] = TYPE_END
                if "act_context" in data:
                    target_context[bi, local_t] = int(data["act_context"][ri])
                else:
                    target_context[bi, local_t] = int(round(float(feats[bi, local_t, 17]) * 64.0)) if feats.shape[-1] > 17 else 0
                dca_group_index[bi, local_t] = int(_get_row(data, "dca_group_index", ri, -1))
                dca_pos[bi, local_t] = int(_get_row(data, "dca_pos", ri, -1))
                dca_len[bi, local_t] = int(_get_row(data, "dca_len", ri, 0))
                dca_remaining[bi, local_t] = int(_get_row(data, "dca_remaining", ri, 0))
                dca_prior_same_slot[bi, local_t] = int(_get_row(data, "dca_prior_same_slot", ri, 0))
                dca_prior_unique_slots[bi, local_t] = int(_get_row(data, "dca_prior_unique_slots", ri, 0))
                dca_prior_max_repeat[bi, local_t] = int(_get_row(data, "dca_prior_max_repeat", ri, 0))
                dca_group_unique_slots[bi, local_t] = int(_get_row(data, "dca_group_unique_slots", ri, 0))
                dca_group_focus_frac[bi, local_t] = float(_get_row(data, "dca_group_focus_frac", ri, 0.0))
                min_count[bi, local_t] = int(data["min_c"][ri])
                max_count[bi, local_t] = int(data["max_c"][ri])
                step_mask[bi, local_t] = 1.0
                sample_weight[bi, local_t] = self._row_weight(data, ri)
                if "future_plan" in data:
                    future_plan[bi, local_t] = _fit_1d(data["future_plan"][ri], self.future_plan_dim, dtype=np.float32)
                outcome[bi, local_t] = float(data.get("won", np.zeros(len(data["board"]), dtype=np.float32))[ri])

        return SequenceBatch(
            board=torch.from_numpy(board),
            hand=torch.from_numpy(hand),
            feats=torch.from_numpy(feats),
            state_token_feats=torch.from_numpy(state_token_feats),
            ledger_feats=torch.from_numpy(ledger_feats),
            known_opp_cards=torch.from_numpy(known_opp_cards),
            known_opp_counts=torch.from_numpy(known_opp_counts),
            known_opp_mask=torch.from_numpy(known_opp_mask),
            prev_type=torch.from_numpy(prev_type),
            prev_card=torch.from_numpy(prev_card),
            prev_card2=torch.from_numpy(prev_card2),
            prev_attack=torch.from_numpy(prev_attack),
            prev_context=torch.from_numpy(prev_context),
            prev_select_type=torch.from_numpy(prev_select_type),
            prev_count=torch.from_numpy(prev_count),
            opt_type=torch.from_numpy(opt_type),
            opt_card=torch.from_numpy(opt_card),
            opt_card2=torch.from_numpy(opt_card2),
            opt_attack=torch.from_numpy(opt_attack),
            opt_feats=torch.from_numpy(opt_feats),
            option_mask=torch.from_numpy(option_mask),
            target_first=torch.from_numpy(target_first),
            target_order=torch.from_numpy(target_order),
            target_multi=torch.from_numpy(target_multi),
            target_type=torch.from_numpy(target_type),
            target_context=torch.from_numpy(target_context),
            dca_group_index=torch.from_numpy(dca_group_index),
            dca_pos=torch.from_numpy(dca_pos),
            dca_len=torch.from_numpy(dca_len),
            dca_remaining=torch.from_numpy(dca_remaining),
            dca_prior_same_slot=torch.from_numpy(dca_prior_same_slot),
            dca_prior_unique_slots=torch.from_numpy(dca_prior_unique_slots),
            dca_prior_max_repeat=torch.from_numpy(dca_prior_max_repeat),
            dca_group_unique_slots=torch.from_numpy(dca_group_unique_slots),
            dca_group_focus_frac=torch.from_numpy(dca_group_focus_frac),
            min_count=torch.from_numpy(min_count),
            max_count=torch.from_numpy(max_count),
            step_mask=torch.from_numpy(step_mask),
            sample_weight=torch.from_numpy(sample_weight),
            future_plan=torch.from_numpy(future_plan),
            outcome=torch.from_numpy(outcome),
            game_keys=game_keys,
            row_refs=windows,
        )

    def _row_weight(self, data: dict[str, np.ndarray], i: int) -> float:
        won = int(data.get("won", np.zeros(len(data["board"]), dtype=np.int8))[i])
        draw = int(data.get("draw", np.zeros(len(data["board"]), dtype=np.int8))[i])
        if draw:
            return self.draw_weight
        return self.win_weight if won else self.loss_weight


def _get_row(data: dict[str, np.ndarray], key: str, i: int, default: float | int) -> float | int:
    if key not in data:
        return default
    return data[key][i]


def _fit_1d(value: object, dim: int, *, dtype=np.float32) -> np.ndarray:
    out = np.zeros(dim, dtype=dtype)
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    n = min(dim, arr.size)
    if n:
        out[:n] = arr[:n]
    return out


def _fit_2d(value: object, rows: int, cols: int, *, dtype=np.float32) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=dtype)
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return out
    r = min(rows, arr.shape[0])
    c = min(cols, arr.shape[1])
    if r and c:
        out[:r, :c] = arr[:r, :c]
    return out
