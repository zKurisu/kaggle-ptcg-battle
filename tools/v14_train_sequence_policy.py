#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import math
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.encoder import OPT_FEAT_DIM, STATE_FEAT_DIM, STATE_TOKEN_FEAT_DIM
from ptcg_rl.seq.constants import DEFAULT_SEQ_LEN, FEATURE_VERSION, FUTURE_PLAN_DIM, LEDGER_FEAT_DIM
from ptcg_rl.seq.data import SequenceBatch
from ptcg_rl.seq.data import SequenceCorpus, discover_sequence_npz
from ptcg_rl.seq.model import SequenceLossConfig, SequencePolicyNet, sequence_accuracy, sequence_policy_loss

_TYPE_LOG_KEYS = ("play", "attach", "evolve", "ability", "attack", "end")


def _split_csv(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _mean_parts(parts: list[dict[str, float]]) -> dict[str, float]:
    if not parts:
        return {}
    keys = sorted({k for p in parts for k in p})
    return {k: float(np.mean([p.get(k, 0.0) for p in parts])) for k in keys}


def _iter_batches(ids: list[int], batch_size: int, *, shuffle: bool, seed: int, max_batches: int = 0):
    order = list(ids)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)
    batches = 0
    for start in range(0, len(order), batch_size):
        if max_batches and batches >= max_batches:
            break
        yield order[start:start + batch_size]
        batches += 1


def _total_batches(n_items: int, batch_size: int, max_batches: int = 0) -> int:
    total = max(1, int(math.ceil(max(n_items, 1) / max(batch_size, 1))))
    return min(total, int(max_batches)) if max_batches else total


def _cuda_mem(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(idx) / 1024**3
    reserved = torch.cuda.memory_reserved(idx) / 1024**3
    return f" mem={alloc:.1f}/{reserved:.1f}G"


def _nonfinite_names(parts: dict[str, float]) -> str:
    bad = [k for k, v in sorted(parts.items()) if isinstance(v, (int, float)) and not np.isfinite(float(v))]
    return ",".join(bad) if bad else "loss_tensor"


def _fmt(stats: dict[str, float], key: str, digits: int = 3) -> str:
    return f"{stats.get(key, 0.0):.{digits}f}"


def _signal_stats_line(stats: dict[str, object]) -> str:
    keys = [
        ("game_len_mean", 1),
        ("game_len_p90", 1),
        ("win_rate", 3),
        ("target_k_mean", 3),
        ("multi_target_rate", 3),
        ("capable_multi_rate", 3),
        ("plan_label_density", 3),
        ("plan_value_mean", 3),
        ("known_opp_rate", 3),
        ("known_opp_slots_mean", 2),
        ("known_opp_count_mean", 2),
        ("turn_continue_rate", 3),
        ("turn_remaining_mean", 2),
        ("turn_future_type_density", 3),
        ("turn_next_rate", 3),
        ("turn_next_card_rate", 3),
        ("turn_next_attack_rate", 3),
        ("turn_next_context_rate", 3),
        ("turn_plan_seq_density", 3),
        ("turn_plan_seq_card_rate", 3),
        ("turn_plan_seq_attack_rate", 3),
        ("dca_rate", 3),
        ("dca_groups", 0),
        ("dca_spread_rate", 3),
        ("dca_focus_mean", 3),
        ("dca_unique_mean", 3),
    ]
    parts: list[str] = []
    for key, digits in keys:
        value = stats.get(key, 0)
        if isinstance(value, float):
            parts.append(f"{key}={value:.{digits}f}")
        else:
            parts.append(f"{key}={value}")
    type_parts = []
    for name in _TYPE_LOG_KEYS:
        value = stats.get(f"type_{name}_rate", 0.0)
        if isinstance(value, float):
            type_parts.append(f"{name}:{value:.2f}")
    return "Signal stats: " + " ".join(parts) + " type_rates=" + ",".join(type_parts)


def _current_only_batch(batch: SequenceBatch) -> SequenceBatch:
    """Keep only the rightmost/current decision row for validation ablation."""
    values = {}
    seq_len = int(batch.step_mask.shape[1])
    last = max(seq_len - 1, 0)
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == seq_len:
            out = torch.zeros_like(value)
            out[:, last] = value[:, last]
            values[field.name] = out
        else:
            values[field.name] = value
    return SequenceBatch(**values)


def _no_action_history_batch(batch: SequenceBatch) -> SequenceBatch:
    """Keep board/hand history, but remove live action-ledger channels."""
    values = {field.name: getattr(batch, field.name) for field in dataclasses.fields(batch)}
    for name in (
        "ledger_feats",
        "prev_type",
        "prev_card",
        "prev_card2",
        "prev_attack",
        "prev_context",
        "prev_select_type",
        "prev_count",
    ):
        values[name] = torch.zeros_like(values[name])
    return SequenceBatch(**values)


def _stateless_current_batch(batch: SequenceBatch) -> SequenceBatch:
    """Keep only current board/hand/options and remove all explicit history ledgers."""
    current = _current_only_batch(batch)
    values = {field.name: getattr(current, field.name) for field in dataclasses.fields(current)}
    for name in (
        "ledger_feats",
        "prev_type",
        "prev_card",
        "prev_card2",
        "prev_attack",
        "prev_context",
        "prev_select_type",
        "prev_count",
        "known_opp_cards",
        "known_opp_counts",
        "known_opp_mask",
    ):
        values[name] = torch.zeros_like(values[name])
    return SequenceBatch(**values)


def _no_known_info_batch(batch: SequenceBatch) -> SequenceBatch:
    """Remove public known-opponent-hand memory while keeping other history."""
    values = {field.name: getattr(batch, field.name) for field in dataclasses.fields(batch)}
    values["known_opp_cards"] = torch.zeros_like(values["known_opp_cards"])
    values["known_opp_counts"] = torch.zeros_like(values["known_opp_counts"])
    values["known_opp_mask"] = torch.zeros_like(values["known_opp_mask"])
    ledger = values["ledger_feats"].clone()
    if ledger.shape[-1] > 80:
        ledger[..., 80:] = 0
    values["ledger_feats"] = ledger
    return SequenceBatch(**values)


def _reversed_prefix_batch(batch: SequenceBatch) -> SequenceBatch:
    """Reverse historical prefix rows while preserving the live/current row."""
    values = {}
    seq_len = int(batch.step_mask.shape[1])
    if seq_len <= 2:
        return batch
    order = torch.arange(seq_len, device=batch.step_mask.device)
    order[:-1] = torch.flip(order[:-1], dims=[0])
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == seq_len:
            values[field.name] = value.index_select(1, order)
        else:
            values[field.name] = value
    return SequenceBatch(**values)


def _current_logit_compare(
    *,
    name: str,
    batch: SequenceBatch,
    full_outputs: dict[str, torch.Tensor],
    ab_outputs: dict[str, torch.Tensor],
) -> dict[str, float]:
    ab_acc = sequence_accuracy(ab_outputs, batch)
    full_acc = sequence_accuracy(full_outputs, batch)
    valid = (batch.target_first >= 0) & (batch.step_mask > 0)
    current = torch.zeros_like(valid, dtype=torch.bool)
    if current.shape[1] > 0:
        current[:, -1] = True
    mask = valid & current
    if not bool(mask.any()):
        return {
            f"{name}_top1": 0.0,
            f"{name}_delta": 0.0,
            f"{name}_agree": 0.0,
            f"{name}_kl": 0.0,
        }
    full_logits = full_outputs["action_logits"][mask]
    ab_logits = ab_outputs["action_logits"][mask]
    opt_mask = batch.option_mask[mask].float()
    full_logits = full_logits.masked_fill(opt_mask <= 0, -1e4)
    ab_logits = ab_logits.masked_fill(opt_mask <= 0, -1e4)
    full_pred = full_logits.argmax(dim=-1)
    ab_pred = ab_logits.argmax(dim=-1)
    full_lp = torch.log_softmax(full_logits, dim=-1)
    ab_lp = torch.log_softmax(ab_logits, dim=-1)
    full_p = full_lp.exp()
    kl = (full_p * (full_lp - ab_lp)).sum(dim=-1).mean().item()
    out = {
        f"{name}_top1": float(ab_acc.get("cur_top1", 0.0)),
        f"{name}_delta": float(ab_acc.get("cur_top1", 0.0) - full_acc.get("cur_top1", 0.0)),
        f"{name}_agree": float((full_pred == ab_pred).float().mean().item()),
        f"{name}_kl": float(kl),
        f"{name}_seq_top1": float(ab_acc.get("top1", 0.0)),
        f"{name}_seq_delta": float(ab_acc.get("top1", 0.0) - full_acc.get("top1", 0.0)),
        f"{name}_next_acc": float(ab_acc.get("next_type_acc", 0.0)),
        f"{name}_next_delta": float(ab_acc.get("next_type_acc", 0.0) - full_acc.get("next_type_acc", 0.0)),
        f"{name}_plan_f1": float(ab_acc.get("plan_f1", 0.0)),
        f"{name}_plan_delta": float(ab_acc.get("plan_f1", 0.0) - full_acc.get("plan_f1", 0.0)),
        f"{name}_dplan_f1": float(ab_acc.get("dca_plan_spread_f1", 0.0)),
        f"{name}_dplan_delta": float(ab_acc.get("dca_plan_spread_f1", 0.0) - full_acc.get("dca_plan_spread_f1", 0.0)),
    }
    if "plan_logits" in full_outputs and "plan_logits" in ab_outputs:
        full_plan = torch.sigmoid(full_outputs["plan_logits"][mask])
        ab_plan = torch.sigmoid(ab_outputs["plan_logits"][mask])
        out[f"{name}_plan_l1"] = float((full_plan - ab_plan).abs().mean().item())
    if "next_type_logits" in full_outputs and "next_type_logits" in ab_outputs:
        full_next = full_outputs["next_type_logits"][mask][:, 0, :]
        ab_next = ab_outputs["next_type_logits"][mask][:, 0, :]
        full_next_lp = torch.log_softmax(full_next, dim=-1)
        ab_next_lp = torch.log_softmax(ab_next, dim=-1)
        full_next_p = full_next_lp.exp()
        out[f"{name}_next_kl"] = float((full_next_p * (full_next_lp - ab_next_lp)).sum(dim=-1).mean().item())
    return out


@torch.no_grad()
def _current_ablation_metrics(
    *,
    model: SequencePolicyNet,
    batch: SequenceBatch,
    full_outputs: dict[str, torch.Tensor],
) -> dict[str, float]:
    out: dict[str, float] = {}
    current_only = _current_only_batch(batch)
    out.update(_current_logit_compare(
        name="cur1",
        batch=current_only,
        full_outputs=full_outputs,
        ab_outputs=model(current_only),
    ))
    stateless = _stateless_current_batch(batch)
    out.update(_current_logit_compare(
        name="stateless",
        batch=stateless,
        full_outputs=full_outputs,
        ab_outputs=model(stateless),
    ))
    no_action = _no_action_history_batch(batch)
    out.update(_current_logit_compare(
        name="noact",
        batch=batch,
        full_outputs=full_outputs,
        ab_outputs=model(no_action),
    ))
    no_known = _no_known_info_batch(batch)
    out.update(_current_logit_compare(
        name="noknown",
        batch=batch,
        full_outputs=full_outputs,
        ab_outputs=model(no_known),
    ))
    reversed_prefix = _reversed_prefix_batch(batch)
    out.update(_current_logit_compare(
        name="revhist",
        batch=batch,
        full_outputs=full_outputs,
        ab_outputs=model(reversed_prefix),
    ))
    return out


def _signal_warnings(
    *,
    train_loss: dict[str, float],
    train_acc: dict[str, float],
    val_loss: dict[str, float],
    val_acc: dict[str, float],
    args: argparse.Namespace,
) -> list[str]:
    warnings: list[str] = []
    if val_acc.get("multi2_rate", 0.0) < 0.05 and val_acc.get("capable2_rate", 0.0) > 0.05:
        warnings.append("single_step_dominated")
    if val_loss.get("current_weight_share", 1.0) < 0.50:
        warnings.append(f"current_step_underweighted={val_loss.get('current_weight_share', 0.0):.3f}")
    if val_loss.get("current_row_share", 0.0) < 0.08:
        warnings.append(f"sequence_prefix_dominant_raw={val_loss.get('current_row_share', 0.0):.3f}")
    if train_loss.get("grad_norm", 0.0) > 1000.0:
        warnings.append(f"large_preclip_grad={train_loss.get('grad_norm', 0.0):.1f}")
    if val_acc.get("cur_n", 0.0) > 50 and val_acc.get("cur_top1", 0.0) + 0.05 < val_acc.get("top1", 0.0):
        warnings.append("current_step_weaker_than_prefix")
    if val_acc.get("cur_n", 0.0) > 50 and val_acc.get("cur_forced_rate", 0.0) > 0.20:
        warnings.append(f"current_top1_forced_inflated={val_acc.get('cur_forced_rate', 0.0):.3f}")
    if val_acc.get("cur_target_margin_best", 0.0) < 0.10 and val_acc.get("cur_n", 0.0) > 50:
        warnings.append(f"current_exact_rank_margin_low={val_acc.get('cur_target_margin_best', 0.0):.3f}")
    if val_acc.get("cur_rank_violation_025", 0.0) > 0.35 and val_acc.get("cur_n", 0.0) > 50:
        warnings.append(f"current_exact_rank_violation={val_acc.get('cur_rank_violation_025', 0.0):.3f}")
    if (
        val_acc.get("cur_n", 0.0) > 50
        and val_acc.get("cur_nonforced_top1", 0.0) > 0
        and val_acc.get("cur_nonforced_top1", 0.0) + 0.08 < val_acc.get("cur_top1", 0.0)
    ):
        warnings.append(f"nonforced_current_weak={val_acc.get('cur_nonforced_top1', 0.0):.3f}")
    if val_acc.get("cur_pred_end_when_nonend_legal", 0.0) > 0.08:
        warnings.append(f"premature_end_when_nonend_legal={val_acc.get('cur_pred_end_when_nonend_legal', 0.0):.3f}")
    if val_acc.get("cur_pred_end_when_target_nonend", 0.0) > 0.05:
        warnings.append(f"miss_nonend_by_end={val_acc.get('cur_pred_end_when_target_nonend', 0.0):.3f}")
    for name in ("attach", "evolve", "ability", "attack"):
        legal = val_acc.get(f"cur_{name}_legal_rate", 0.0)
        target = val_acc.get(f"cur_{name}_target_if_legal", 0.0)
        pred = val_acc.get(f"cur_{name}_pred_if_legal", 0.0)
        miss = val_acc.get(f"cur_{name}_miss_if_target", 0.0)
        mass = val_acc.get(f"cur_{name}_target_mass", 0.0)
        if legal > 0.08 and target > 0.08 and pred + 0.12 < target:
            warnings.append(f"current_{name}_opportunity_underused={pred:.3f}<{target:.3f}")
        if miss > 0.25 and target > 0.08:
            warnings.append(f"current_{name}_target_missed={miss:.3f}")
        if target > 0.08 and mass < 0.45:
            warnings.append(f"current_{name}_target_mass_low={mass:.3f}")
        margin = val_acc.get(f"cur_{name}_target_margin_other", 0.0)
        if target > 0.08 and margin < 0.10:
            warnings.append(f"current_{name}_target_margin_low={margin:.3f}")
    if not args.diagnostic_ablation:
        warnings.append("diagnostic_ablation_disabled")
    else:
        if val_acc.get("cur1_agree", 0.0) > 0.90 and abs(val_acc.get("cur1_delta", 0.0)) < 0.02:
            warnings.append("history_not_affecting_current")
        if val_acc.get("stateless_agree", 0.0) > 0.88 and abs(val_acc.get("stateless_delta", 0.0)) < 0.03:
            warnings.append("stateless_current_matches_full")
        if val_acc.get("noact_agree", 0.0) > 0.92 and abs(val_acc.get("noact_delta", 0.0)) < 0.02:
            warnings.append("action_ledger_not_affecting_current")
        if val_acc.get("revhist_agree", 0.0) > 0.92 and abs(val_acc.get("revhist_delta", 0.0)) < 0.02:
            warnings.append("history_order_not_affecting_current")
        if (
            val_acc.get("next_type_n", 0.0) > 50
            and abs(val_acc.get("revhist_next_delta", 0.0)) < 0.02
            and val_acc.get("revhist_next_kl", 0.0) < 0.02
        ):
            warnings.append("history_order_not_affecting_future")
    if val_acc.get("plan_pos_rate", 0.0) > 0.10 and val_acc.get("plan_f1", 0.0) < 0.20:
        warnings.append("future_plan_weak")
    if args.history_condition_scale > 0.0 and val_loss.get("hist_query_ratio", 0.0) < 0.05:
        warnings.append(f"history_condition_inactive={val_loss.get('hist_query_ratio', 0.0):.3f}")
    if args.plan_condition_scale > 0.0 and val_loss.get("plan_query_ratio", 0.0) < 0.05:
        warnings.append(f"plan_condition_inactive={val_loss.get('plan_query_ratio', 0.0):.3f}")
    if args.next_type_condition_scale > 0.0 and val_loss.get("next_query_ratio", 0.0) < 0.05:
        warnings.append(f"next_condition_inactive={val_loss.get('next_query_ratio', 0.0):.3f}")
    if args.known_condition_scale > 0.0 and val_loss.get("known_query_ratio", 0.0) < 0.03 and val_acc.get("known_opp_rate", 0.0) > 0.01:
        warnings.append(f"known_condition_inactive={val_loss.get('known_query_ratio', 0.0):.3f}")
    if val_acc.get("known_opp_rate", 0.0) > 0.05 and val_acc.get("noknown_agree", 0.0) > 0.95 and abs(val_acc.get("noknown_delta", 0.0)) < 0.02:
        warnings.append("known_info_not_affecting_current")
    if args.known_action_weight > 0.0 and val_acc.get("known_action_n", 0.0) > 50 and val_acc.get("known_action_top1", 0.0) < 0.35:
        warnings.append(f"known_action_weak={val_acc.get('known_action_top1', 0.0):.3f}")
    if args.known_logit_scale > 0.0 and val_acc.get("known_opp_rate", 0.0) > 0.05 and val_acc.get("noknown_agree", 0.0) > 0.95 and abs(val_acc.get("noknown_delta", 0.0)) < 0.02:
        warnings.append("known_logit_not_affecting_current")
    if args.turn_condition_scale > 0.0 and val_loss.get("turn_query_ratio", 0.0) < 0.05:
        warnings.append(f"turn_condition_inactive={val_loss.get('turn_query_ratio', 0.0):.3f}")
    if args.turn_next_condition_scale > 0.0 and val_loss.get("turn_next_query_ratio", 0.0) < 0.05:
        warnings.append(f"turn_next_condition_inactive={val_loss.get('turn_next_query_ratio', 0.0):.3f}")
    if args.turn_seq_condition_scale > 0.0 and val_loss.get("turn_seq_query_ratio", 0.0) < 0.05:
        warnings.append(f"turn_seq_condition_inactive={val_loss.get('turn_seq_query_ratio', 0.0):.3f}")
    if val_acc.get("turn_continue_target_rate", 0.0) > 0.10 and val_acc.get("turn_continue_f1", 0.0) < 0.45:
        warnings.append(f"turn_continue_weak={val_acc.get('turn_continue_f1', 0.0):.3f}")
    if val_acc.get("turn_future_type_target_rate", 0.0) > 0.03 and val_acc.get("turn_future_type_f1", 0.0) < 0.25:
        warnings.append(f"turn_future_type_weak={val_acc.get('turn_future_type_f1', 0.0):.3f}")
    if val_acc.get("cur_turn_continue_target_rate", 0.0) > 0.10 and val_acc.get("cur_turn_continue_miss", 0.0) > 0.25:
        warnings.append(f"current_turn_continue_missed={val_acc.get('cur_turn_continue_miss', 0.0):.3f}")
    if val_acc.get("cur_turn_continue_target_rate", 0.0) > 0.10 and val_acc.get("cur_terminal_when_continue", 0.0) > 0.10:
        warnings.append(f"terminal_during_continue={val_acc.get('cur_terminal_when_continue', 0.0):.3f}")
    if val_acc.get("cur_turn_continue_target_rate", 0.0) > 0.10 and val_acc.get("cur_terminal_prob_when_continue", 0.0) > 0.20:
        warnings.append(f"terminal_mass_during_continue={val_acc.get('cur_terminal_prob_when_continue', 0.0):.3f}")
    if val_acc.get("next_type_n", 0.0) > 50 and val_acc.get("next_type_acc", 0.0) < 0.45:
        warnings.append(f"next_type_weak={val_acc.get('next_type_acc', 0.0):.3f}")
    if val_acc.get("turn_next_exists_rate", 0.0) > 0.10 and val_acc.get("turn_next_type_pos_acc", 0.0) < 0.35:
        warnings.append(f"turn_next_type_weak={val_acc.get('turn_next_type_pos_acc', 0.0):.3f}")
    if (
        val_acc.get("turn_next_exists_rate", 0.0) > 0.10
        and val_acc.get("turn_next_none_acc", 0.0) > 0.90
        and val_acc.get("turn_next_type_pos_acc", 0.0) < 0.45
    ):
        warnings.append("turn_next_collapsed_to_none")
    if val_acc.get("turn_next_card_n", 0.0) > 50 and val_acc.get("turn_next_card_acc", 0.0) < 0.20:
        warnings.append(f"turn_next_card_weak={val_acc.get('turn_next_card_acc', 0.0):.3f}")
    if val_acc.get("turn_next_attack_n", 0.0) > 50 and val_acc.get("turn_next_attack_acc", 0.0) < 0.20:
        warnings.append(f"turn_next_attack_weak={val_acc.get('turn_next_attack_acc', 0.0):.3f}")
    if val_loss.get("turn_seq_slots", 0.0) > 50 and val_acc.get("turn_seq_type_pos_acc", 0.0) < 0.30:
        warnings.append(f"turn_seq_type_weak={val_acc.get('turn_seq_type_pos_acc', 0.0):.3f}")
    if (
        val_loss.get("turn_seq_slots", 0.0) > 50
        and val_acc.get("turn_seq_none_acc", 0.0) > 0.90
        and val_acc.get("turn_seq_type_pos_acc", 0.0) < 0.45
    ):
        warnings.append("turn_seq_collapsed_to_none")
    if val_acc.get("turn_seq_card_n", 0.0) > 50 and val_acc.get("turn_seq_card_acc", 0.0) < 0.15:
        warnings.append(f"turn_seq_card_weak={val_acc.get('turn_seq_card_acc', 0.0):.3f}")
    if val_acc.get("turn_seq_attack_n", 0.0) > 50 and val_acc.get("turn_seq_attack_acc", 0.0) < 0.20:
        warnings.append(f"turn_seq_attack_weak={val_acc.get('turn_seq_attack_acc', 0.0):.3f}")
    if val_acc.get("ambig_type_rate", 0.0) > 0.10 and val_acc.get("ambig_type_top1", 0.0) + 0.05 < val_acc.get("top1", 0.0):
        warnings.append("ambiguous_options_weak")
    for name in ("attach", "evolve", "attack"):
        if val_acc.get(f"{name}_rate", 0.0) > 0.04 and val_acc.get(f"{name}_top1", 1.0) < 0.45:
            warnings.append(f"{name}_decision_weak")
    if args.damage_counter_weight > 1.0 and val_acc.get("dca_n", 0.0) <= 0:
        warnings.append("no_val_dca_signal")
    if val_acc.get("dca_n", 0.0) > 50:
        dca_gap = train_acc.get("dca_top1", 0.0) - val_acc.get("dca_top1", 0.0)
        if dca_gap > 0.10:
            warnings.append(f"dca_overfit_gap={dca_gap:.3f}")
        if val_acc.get("dca_spread_rate", 0.0) < 0.15:
            warnings.append("dca_labels_focus_dominated")
        if val_acc.get("dca_plan_spread_rate", 0.0) > 0.10 and val_acc.get("dca_plan_spread_f1", 0.0) < 0.30:
            warnings.append("dca_block_plan_weak")
        if (
            val_acc.get("dca_plan_spread_rate", 0.0) > 0.10
            and val_acc.get("dca_plan_pred_spread_rate", 0.0) < val_acc.get("dca_plan_spread_rate", 0.0) * 0.50
        ):
            warnings.append("dca_spread_head_collapsed")
        if args.damage_counter_weight > 1.0 and val_loss.get("dca_weight_share", 0.0) < max(0.05, val_acc.get("dca_rate", 0.0)):
            warnings.append("dca_underweighted")
        if val_acc.get("dca_late_top1", 0.0) + 0.05 < val_acc.get("dca_first_top1", 0.0):
            warnings.append("dca_late_sequence_weak")
    if val_acc.get("multi2_n", 0.0) > 50:
        m2_gap = train_acc.get("multi2_f1", 0.0) - val_acc.get("multi2_f1", 0.0)
        if m2_gap > 0.10:
            warnings.append(f"multi_overfit_gap={m2_gap:.3f}")
    if not np.isfinite(val_loss.get("loss", 0.0)):
        warnings.append("nonfinite_val_loss")
    return warnings


def run_epoch(
    *,
    model: SequencePolicyNet,
    corpus: SequenceCorpus,
    ids: list[int],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    loss_cfg: SequenceLossConfig,
    batch_size: int,
    epoch: int,
    amp: bool,
    grad_clip: float,
    max_batches: int,
    progress_every: int,
    diagnostic_ablation: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    train = optimizer is not None
    model.train(train)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and train)
    loss_parts: list[dict[str, float]] = []
    acc_parts: list[dict[str, float]] = []
    t0 = time.time()
    total_batches = _total_batches(len(ids), batch_size, max_batches)
    target_seen = min(len(ids), total_batches * max(batch_size, 1))
    seen = 0
    for bi, sample_ids in enumerate(
        _iter_batches(ids, batch_size, shuffle=train, seed=epoch * 100003 + 17, max_batches=max_batches),
        1,
    ):
        seen += len(sample_ids)
        batch = corpus.collate(sample_ids).to(device)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(batch)
                loss, parts = sequence_policy_loss(outputs, batch, loss_cfg)
            if not torch.isfinite(loss):
                print(
                    f"FATAL nonfinite_loss mode={'train' if train else 'val'} epoch={epoch} "
                    f"batch={bi}/{total_batches} bad={_nonfinite_names(parts)} parts={parts}",
                    flush=True,
                )
                raise RuntimeError("nonfinite sequence policy loss")
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                grad_norm = 0.0
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    grad_norm = float(grad.detach().cpu()) if torch.is_tensor(grad) else float(grad)
                parts = dict(parts)
                if not np.isfinite(grad_norm):
                    parts["grad_norm"] = 0.0
                    parts["grad_nonfinite"] = 1.0
                    if amp:
                        print(
                            f"WARN nonfinite_grad_skip mode=train epoch={epoch} batch={bi}/{total_batches} "
                            f"grad_norm={grad_norm} parts={parts}",
                            flush=True,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                    else:
                        print(
                            f"FATAL nonfinite_grad mode=train epoch={epoch} batch={bi}/{total_batches} "
                            f"grad_norm={grad_norm} parts={parts}",
                            flush=True,
                        )
                        raise RuntimeError("nonfinite sequence policy gradient")
                else:
                    parts["grad_norm"] = grad_norm
                    parts["grad_nonfinite"] = 0.0
                    scaler.step(optimizer)
                    scaler.update()
        loss_parts.append(parts)
        acc = sequence_accuracy(outputs, batch)
        if diagnostic_ablation and not train:
            acc.update(_current_ablation_metrics(model=model, batch=batch, full_outputs=outputs))
        acc_parts.append(acc)
        if progress_every and (bi == 1 or bi % progress_every == 0 or bi == total_batches):
            elapsed = time.time() - t0
            rate = seen / max(elapsed, 1e-9)
            eta = max(total_batches - bi, 0) * max(batch_size, 1) / max(rate, 1e-9)
            mp = _mean_parts(loss_parts[-max(progress_every, 1):])
            ma = _mean_parts(acc_parts[-max(progress_every, 1):])
            mode = "train" if train else "val"
            print(
                f"  {mode} epoch={epoch} batch={bi}/{total_batches} "
                f"seen={seen}/{target_seen} loss={mp.get('loss', 0):.4f} "
                f"act={mp.get('action', 0):.4f} ord={mp.get('order', 0):.4f} "
                f"cnt_loss={mp.get('count', 0):.4f} multi={mp.get('multi', 0):.4f} "
                f"plan={mp.get('plan', 0):.4f} out={mp.get('outcome', 0):.4f} "
                f"top1={ma.get('top1', 0):.3f} type={ma.get('type_acc', 0):.3f} "
                f"cnt={ma.get('count_acc', 0):.3f}/{ma.get('count_mae', 0):.2f} "
                f"k={ma.get('pred_k', 0):.2f}/{ma.get('target_k', 0):.2f} "
                f"cur={ma.get('cur_top1', 0):.3f}/{ma.get('cur_type_acc', 0):.3f}/"
                f"{mp.get('current_weight_share', 0):.2f}/{mp.get('current_row_share', 0):.2f} "
                f"cmp={mp.get('current_complexity_mean', 0):.2f}/{mp.get('current_complexity_max', 0):.2f} "
                f"curQ=H{ma.get('cur_entropy', 0):.2f}/M{ma.get('cur_margin', 0):.2f}/"
                f"R{ma.get('cur_target_margin_best', 0):.2f}/{ma.get('cur_rank_violation_025', 0):.2f}/"
                f"AR{ma.get('cur_ambig_target_margin_best', 0):.2f}/"
                f"F{ma.get('cur_forced_rate', 0):.2f}/NF{ma.get('cur_nonforced_top1', 0):.3f}/"
                f"B{ma.get('cur_bigopt_top1', 0):.3f}/A{ma.get('cur_ambig_type_top1', 0):.3f}/"
                f"End{ma.get('cur_pred_end_rate', 0):.2f}/{ma.get('cur_pred_end_when_nonend_legal', 0):.2f} "
                f"opp=at{ma.get('cur_attach_legal_rate', 0):.2f}/{ma.get('cur_attach_target_if_legal', 0):.2f}/{ma.get('cur_attach_pred_if_legal', 0):.2f} "
                f"ev{ma.get('cur_evolve_legal_rate', 0):.2f}/{ma.get('cur_evolve_target_if_legal', 0):.2f}/{ma.get('cur_evolve_pred_if_legal', 0):.2f} "
                f"ak{ma.get('cur_attack_legal_rate', 0):.2f}/{ma.get('cur_attack_target_if_legal', 0):.2f}/{ma.get('cur_attack_pred_if_legal', 0):.2f} "
                f"oppM=at{ma.get('cur_attach_target_mass', 0):.2f}/"
                f"ev{ma.get('cur_evolve_target_mass', 0):.2f}/"
                f"ab{ma.get('cur_ability_target_mass', 0):.2f}/"
                f"ak{ma.get('cur_attack_target_mass', 0):.2f}/"
                f"{mp.get('opportunity_type', 0):.3f} "
                f"oppMargin=at{ma.get('cur_attach_target_margin_other', 0):.2f}/"
                f"ev{ma.get('cur_evolve_target_margin_other', 0):.2f}/"
                f"ab{ma.get('cur_ability_target_margin_other', 0):.2f}/"
                f"ak{ma.get('cur_attack_target_margin_other', 0):.2f}/"
                f"{mp.get('opportunity_margin', 0):.3f}/"
                f"{mp.get('opportunity_margin_violation', 0):.2f} "
                f"seq={ma.get('seq_len_mean', 0):.1f}/{ma.get('seq_full_rate', 0):.2f}/{ma.get('history_present_rate', 0):.2f} "
                f"known={ma.get('known_opp_rate', 0):.2f}/{ma.get('known_opp_slots_mean', 0):.1f}/"
                f"{mp.get('known_opp_slots_mean', 0):.1f}/"
                f"{ma.get('known_action_top1', 0):.3f}/{mp.get('known_action', 0):.3f} "
                f"turn={ma.get('turn_continue_f1', 0):.3f}/"
                f"{ma.get('turn_continue_target_rate', 0):.2f}->{ma.get('turn_continue_pred_rate', 0):.2f}/"
                f"{ma.get('turn_remaining_mae', 0):.2f}/"
                f"{ma.get('turn_future_type_f1', 0):.3f}/"
                f"{mp.get('turn_plan', 0):.3f} "
                f"curTurn={ma.get('cur_turn_continue_f1', 0):.3f}/"
                f"{ma.get('cur_turn_continue_target_rate', 0):.2f}->{ma.get('cur_turn_continue_pred_rate', 0):.2f}/"
                f"miss{ma.get('cur_turn_continue_miss', 0):.2f}/"
                f"term{ma.get('cur_terminal_when_continue', 0):.2f}/"
                f"pT{ma.get('cur_terminal_prob_when_continue', 0):.2f}/"
                f"stopT{ma.get('cur_terminal_prob_when_stop', 0):.2f} "
                f"turnNext={ma.get('turn_next_exists_rate', 0):.2f}/"
                f"{ma.get('turn_next_type_pos_acc', 0):.3f}/"
                f"none{ma.get('turn_next_none_acc', 0):.3f}/"
                f"card{ma.get('turn_next_card_acc', 0):.3f}/"
                f"atk{ma.get('turn_next_attack_acc', 0):.3f}/"
                f"{mp.get('turn_next_plan', 0):.3f} "
                f"setF1={ma.get('set_f1', 0):.3f} ordAcc={ma.get('order_acc', 0):.3f} "
                f"atype={ma.get('action_type_acc', 0):.3f} "
                f"pln={ma.get('plan_f1', 0):.3f}/{ma.get('plan_mae', 0):.3f}/"
                f"{ma.get('plan_pos_rate', 0):.2f}->{ma.get('plan_pred_pos_rate', 0):.2f} "
                f"nxt={ma.get('next_type_acc', 0):.3f}/{ma.get('next1_acc', 0):.3f}/"
                f"{ma.get('next2_acc', 0):.3f}/{ma.get('next3_acc', 0):.3f} "
                f"opt={ma.get('option_n', 0):.1f}/{ma.get('ambig_type_rate', 0):.2f}/"
                f"{ma.get('ambig_type_top1', 0):.3f} "
                f"m2n={ma.get('multi2_n', 0):.0f} m2r={ma.get('multi2_rate', 0):.3f} "
                f"cap2={ma.get('capable2_rate', 0):.3f} m2F1={ma.get('multi2_f1', 0):.3f} "
                f"m2Ord={ma.get('multi2_order_acc', 0):.3f} "
                f"dcaN={ma.get('dca_n', 0):.0f} dcaR={ma.get('dca_rate', 0):.3f} "
                f"dcaTop1={ma.get('dca_top1', 0):.3f} dcaF1={ma.get('dca_f1', 0):.3f} "
                f"dcaCnt={ma.get('dca_count_acc', 0):.3f}/{ma.get('dca_count_mae', 0):.2f} "
                f"dcaK={ma.get('dca_pred_k', 0):.2f}/{ma.get('dca_target_k', 0):.2f} "
                f"dcaBlk={ma.get('dca_focus_mean', 0):.2f}/{ma.get('dca_spread_rate', 0):.2f}/"
                f"{ma.get('dca_prior_unique_mean', 0):.2f} "
                f"dcaSplit={ma.get('dca_first_top1', 0):.3f}/{ma.get('dca_late_top1', 0):.3f}/"
                f"{ma.get('dca_spread_top1', 0):.3f} "
                f"dplan={ma.get('dca_plan_spread_f1', 0):.3f}/{ma.get('dca_plan_unique_acc', 0):.3f}/"
                f"{ma.get('dca_plan_focus_mae', 0):.3f}/"
                f"{ma.get('dca_plan_spread_rate', 0):.2f}->{ma.get('dca_plan_pred_spread_rate', 0):.2f} "
                f"boost={mp.get('weight_boost', 0):.2f} dcaW={mp.get('dca_weight_share', 0):.2f} "
                f"m2W={mp.get('multi_weight_share', 0):.2f} nxtL={mp.get('next_type', 0):.3f} "
                f"dplanL={mp.get('dca_plan', 0):.3f} dposW={mp.get('dca_spread_pos_weight', 0):.1f} "
                f"curA={mp.get('current_action', 0):.3f}/{mp.get('prefix_action', 0):.3f} "
                f"curHead={mp.get('action_current_head', 0):.3f}/{mp.get('action_prefix_head', 0):.3f} "
                f"dcaA={mp.get('dca_action', 0):.3f}/{mp.get('non_dca_action', 0):.3f} "
                f"cond=h{mp.get('hist_query_ratio', 0):.2f}/p{mp.get('plan_query_ratio', 0):.2f}/"
                f"n{mp.get('next_query_ratio', 0):.2f}/d{mp.get('dca_query_ratio', 0):.2f}/"
                f"k{mp.get('known_query_ratio', 0):.2f}/u{mp.get('turn_query_ratio', 0):.2f}/"
                f"tn{mp.get('turn_next_query_ratio', 0):.2f}/"
                f"q{mp.get('conditioned_query_ratio', 0):.2f}/"
                f"t{mp.get('type_prior_abs', 0):.2f} "
                f"out={ma.get('outcome_acc', 0):.3f}/{ma.get('outcome_brier', 0):.3f}/"
                f"{ma.get('outcome_pos_rate', 0):.2f}->{ma.get('outcome_pred_pos_rate', 0):.2f} "
                f"types=p{ma.get('play_top1', 0):.2f},at{ma.get('attach_top1', 0):.2f},"
                f"ev{ma.get('evolve_top1', 0):.2f},ab{ma.get('ability_top1', 0):.2f},"
                f"ak{ma.get('attack_top1', 0):.2f},en{ma.get('end_top1', 0):.2f} "
                f"grad={mp.get('grad_norm', 0):.2f}/skip{mp.get('grad_nonfinite', 0):.2f} "
                f"{rate:.0f} samples/s eta={eta:.0f}s{_cuda_mem(device)}",
                flush=True,
            )
    return _mean_parts(loss_parts), _mean_parts(acc_parts)


def save_checkpoint(
    path: str,
    *,
    model: SequencePolicyNet,
    args: argparse.Namespace,
    corpus: SequenceCorpus,
    epoch: int,
    val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "format": "ptcg_seq_v14_torch",
            "feature_version": FEATURE_VERSION,
            "model_state": model.state_dict(),
            "model_config": model.config(),
            "train_args": vars(args),
            "corpus_stats": corpus.stats,
            "epoch": int(epoch),
            "val_loss": float(val_loss),
        },
        path,
    )


def _run_random_smoke(args: argparse.Namespace, checkpoint_path: str, epoch: int) -> dict[str, float]:
    if not args.random_smoke_deck or int(args.random_smoke_games) <= 0:
        return {}
    if int(args.random_smoke_every) > 1 and epoch % int(args.random_smoke_every) != 0:
        return {}
    cmd = [
        sys.executable,
        str(_HERE / "v14_eval_random.py"),
        checkpoint_path,
        "--deck",
        args.random_smoke_deck,
        "--games",
        str(int(args.random_smoke_games)),
        "--workers",
        str(int(args.random_smoke_workers)),
        "--device",
        args.random_smoke_device,
        "--seed",
        str(int(args.seed) + epoch * 1009),
        "--max-turns",
        str(int(args.random_smoke_max_turns)),
        "--progress-every",
        str(int(args.random_smoke_progress_every)),
    ]
    print(
        "random_smoke_start "
        f"epoch={epoch} games={args.random_smoke_games} workers={args.random_smoke_workers} "
        f"deck={args.random_smoke_deck}",
        flush=True,
    )
    wins = games = errors = policy_errors = 0
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if line:
            print(f"random_smoke {line}", flush=True)
        m = re.search(r"Win rate vs Random:\s*([0-9.]+)%\s*\((\d+)/(\d+)\)", line)
        if m:
            wins = int(m.group(2))
            games = int(m.group(3))
        m = re.search(r"Timeout/error games:\s*(\d+)/(\d+)", line)
        if m:
            errors = int(m.group(1))
        m = re.search(r"Policy fallback errors:\s*(\d+)", line)
        if m:
            policy_errors = int(m.group(1))
    rc = proc.wait()
    elapsed = time.time() - t0
    wr = float(wins / max(games, 1)) if games else 0.0
    status = "ok" if rc == 0 else f"failed_rc={rc}"
    warn = ""
    if games and wr < float(args.random_smoke_min_wr):
        warn = f" warning=random_smoke_below_min_{float(args.random_smoke_min_wr):.3f}"
    if rc != 0:
        warn += " warning=random_smoke_failed"
    print(
        f"random_smoke_result epoch={epoch} status={status} wr={wr:.3f} "
        f"wins={wins}/{games} errors={errors} policy_errors={policy_errors} "
        f"elapsed={elapsed:.1f}s{warn}",
        flush=True,
    )
    return {
        "random_smoke_wr": wr,
        "random_smoke_games": float(games),
        "random_smoke_errors": float(errors),
        "random_smoke_policy_errors": float(policy_errors),
        "random_smoke_rc": float(rc),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--archetype", required=True)
    p.add_argument("--score-bands", nargs="+", default=["900-999", "1000-1099", "1100-1199", "1200+"])
    p.add_argument("--date-from", default="")
    p.add_argument("--date-to", default="")
    p.add_argument("--deck-sig", action="append", default=[])
    p.add_argument("--team-name", action="append", default=[])
    p.add_argument("--opponent-archetype", action="append", default=[])
    p.add_argument("--opponent-deck-sig", action="append", default=[])
    p.add_argument("--winner-only", action="store_true")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--max-score", type=float, default=0.0)
    p.add_argument("--win-weight", type=float, default=1.5)
    p.add_argument("--loss-weight", type=float, default=0.5)
    p.add_argument("--draw-weight", type=float, default=0.8)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--allow-amp", action="store_true",
                   help="explicitly allow v14 AMP despite known unstable signal diagnostics")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=200)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--diagnostic-ablation", action="store_true",
                   help="during validation, compare full sequence against current-only ablation")
    p.add_argument("--action-weight", type=float, default=1.0)
    p.add_argument("--current-action-weight", type=float, default=1.0,
                   help="main CE weight for the rightmost decision row used by live inference")
    p.add_argument("--prefix-action-weight", type=float, default=0.10,
                   help="auxiliary CE weight for historical prefix rows; keep small so history is context")
    p.add_argument("--order-weight", type=float, default=0.15,
                   help="ordered multi-select CE weight; applied only when target_k>1")
    p.add_argument("--multi-weight", type=float, default=0.15)
    p.add_argument("--count-weight", type=float, default=0.20)
    p.add_argument("--plan-weight", type=float, default=0.35)
    p.add_argument("--next-type-weight", type=float, default=0.25,
                   help="predict future action-type sequence from each prefix state")
    p.add_argument("--dca-plan-weight", type=float, default=0.25,
                   help="predict block-level DamageCounterAny spread/unique/focus labels")
    p.add_argument("--known-action-weight", type=float, default=0.0,
                   help="auxiliary CE for known-only option scoring on rows with public known opponent cards")
    p.add_argument("--turn-plan-weight", type=float, default=0.0,
                   help="auxiliary same-turn continuation/future-action plan loss")
    p.add_argument("--turn-terminal-weight", type=float, default=0.0,
                   help="bind same-turn continuation labels to END/ATTACK terminal action mass")
    p.add_argument("--turn-next-plan-weight", type=float, default=0.0,
                   help="auxiliary same-turn next concrete action plan loss")
    p.add_argument("--turn-next-type-weight", type=float, default=1.0,
                   help="relative same-turn next action type loss weight")
    p.add_argument("--turn-next-card-weight", type=float, default=0.25,
                   help="relative same-turn next card/card2 loss weight")
    p.add_argument("--turn-next-attack-weight", type=float, default=0.25,
                   help="relative same-turn next attack loss weight")
    p.add_argument("--turn-next-context-weight", type=float, default=0.10,
                   help="relative same-turn next context loss weight")
    p.add_argument("--turn-seq-plan-weight", type=float, default=0.0,
                   help="auxiliary ordered same-turn K-step action plan loss")
    p.add_argument("--turn-seq-type-weight", type=float, default=1.0,
                   help="relative ordered same-turn K-step type loss weight")
    p.add_argument("--turn-seq-card-weight", type=float, default=0.15,
                   help="relative ordered same-turn K-step card loss weight")
    p.add_argument("--turn-seq-attack-weight", type=float, default=0.15,
                   help="relative ordered same-turn K-step attack loss weight")
    p.add_argument("--turn-seq-context-weight", type=float, default=0.05,
                   help="relative ordered same-turn K-step context loss weight")
    p.add_argument("--opportunity-type-weight", type=float, default=0.0,
                   help="train current action logits to put mass on the demonstrated legal action type")
    p.add_argument("--opportunity-margin-weight", type=float, default=0.0,
                   help="penalize key current actions when target logit is below best other-type option")
    p.add_argument("--opportunity-margin", type=float, default=0.25,
                   help="desired target-vs-other-type logit margin for opportunity-margin loss")
    p.add_argument("--current-rank-margin-weight", type=float, default=0.0,
                   help="penalize current target action when best non-target legal action is too close")
    p.add_argument("--current-rank-margin", type=float, default=0.25,
                   help="desired target-vs-best-non-target logit margin on the live current row")
    p.add_argument("--current-rank-margin-min-options", type=int, default=2,
                   help="minimum legal option count for current-rank-margin loss")
    p.add_argument("--history-condition-scale", type=float, default=0.0,
                   help="inject previous causal sequence state into the live action scorer")
    p.add_argument("--plan-condition-scale", type=float, default=0.0,
                   help="inject predicted future-plan embedding into the live action scorer")
    p.add_argument("--next-type-condition-scale", type=float, default=0.0,
                   help="inject predicted next-action-type embedding into the live action scorer")
    p.add_argument("--dca-condition-scale", type=float, default=0.0,
                   help="inject predicted DamageCounterAny block-plan embedding into the action scorer")
    p.add_argument("--known-condition-scale", type=float, default=0.0,
                   help="inject public known-opponent-hand memory into the action scorer")
    p.add_argument("--known-logit-scale", type=float, default=0.0,
                   help="add known-only option scorer logits to final action logits")
    p.add_argument("--turn-condition-scale", type=float, default=0.0,
                   help="inject predicted same-turn continuation/future-action plan into the live action scorer")
    p.add_argument("--turn-next-condition-scale", type=float, default=0.0,
                   help="inject predicted same-turn next concrete action plan into the live action scorer")
    p.add_argument("--turn-seq-condition-scale", type=float, default=0.0,
                   help="inject predicted ordered same-turn K-step plan into the live action scorer")
    p.add_argument("--type-prior-scale", type=float, default=0.0,
                   help="add predicted action-type log-probability as an option-level prior")
    p.add_argument("--current-complexity-weight", type=float, default=0.0,
                   help="upweight current decisions with many legal options so forced rows do not dominate training")
    p.add_argument("--multi-target-weight", type=float, default=1.0,
                   help="boost decision losses on target_k>1 rows; default keeps historical weighting")
    p.add_argument("--damage-counter-weight", type=float, default=1.0,
                   help="boost DamageCounterAny resolution rows such as Dragapult Phantom Dive")
    p.add_argument("--random-smoke-deck", default="",
                   help="optional deck CSV for per-epoch small random rollout smoke test")
    p.add_argument("--random-smoke-games", type=int, default=0)
    p.add_argument("--random-smoke-workers", type=int, default=4)
    p.add_argument("--random-smoke-every", type=int, default=1)
    p.add_argument("--random-smoke-max-turns", type=int, default=700)
    p.add_argument("--random-smoke-device", default="cpu")
    p.add_argument("--random-smoke-progress-every", type=int, default=20)
    p.add_argument("--random-smoke-min-wr", type=float, default=0.90,
                   help="print a warning when epoch smoke random WR falls below this value")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.amp and not args.allow_amp:
        raise SystemExit(
            "v14 AMP is disabled by default: previous diagnostics produced "
            "nonfinite/degenerate training signals. Remove --amp, or pass "
            "--allow-amp only for an explicit AMP repair experiment."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    paths = discover_sequence_npz(
        args.corpus,
        args.archetype,
        _split_csv(args.score_bands),
        date_from=args.date_from,
        date_to=args.date_to,
    )
    print(f"Sequence v14 train: arch={args.archetype} paths={len(paths)}", flush=True)
    corpus = SequenceCorpus(
        paths,
        seq_len=args.seq_len,
        stride=args.stride,
        state_feat_dim=STATE_FEAT_DIM,
        opt_feat_dim=OPT_FEAT_DIM,
        state_token_feat_dim=STATE_TOKEN_FEAT_DIM,
        ledger_feat_dim=LEDGER_FEAT_DIM,
        future_plan_dim=FUTURE_PLAN_DIM,
        deck_sigs=_split_csv(args.deck_sig),
        team_names=_split_csv(args.team_name),
        opponent_archetypes=_split_csv(args.opponent_archetype),
        opponent_deck_sigs=_split_csv(args.opponent_deck_sig),
        winner_only=args.winner_only,
        min_score=args.min_score,
        max_score=args.max_score,
        win_weight=args.win_weight,
        loss_weight=args.loss_weight,
        draw_weight=args.draw_weight,
    )
    print("Corpus stats:", json.dumps(corpus.stats, ensure_ascii=False, sort_keys=True), flush=True)
    print(_signal_stats_line(corpus.stats), flush=True)
    train_ids, val_ids = corpus.split_samples(args.val_fraction, args.seed)
    print(f"Split: train={len(train_ids)} val={len(val_ids)} seq_len={args.seq_len}", flush=True)

    device = torch.device(args.device)
    model = SequencePolicyNet(
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        state_feat_dim=STATE_FEAT_DIM,
        opt_feat_dim=OPT_FEAT_DIM,
        state_token_feat_dim=STATE_TOKEN_FEAT_DIM,
        ledger_feat_dim=LEDGER_FEAT_DIM,
        future_plan_dim=FUTURE_PLAN_DIM,
        max_seq_len=args.seq_len,
        history_condition_scale=args.history_condition_scale,
        plan_condition_scale=args.plan_condition_scale,
        next_type_condition_scale=args.next_type_condition_scale,
        dca_condition_scale=args.dca_condition_scale,
        known_condition_scale=args.known_condition_scale,
        known_logit_scale=args.known_logit_scale,
        turn_condition_scale=args.turn_condition_scale,
        turn_next_condition_scale=args.turn_next_condition_scale,
        type_prior_scale=args.type_prior_scale,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model params: {n_params/1e6:.2f}M device={device} "
        f"condition_scales=hist:{args.history_condition_scale} plan:{args.plan_condition_scale} "
        f"next:{args.next_type_condition_scale} dca:{args.dca_condition_scale} "
        f"known:{args.known_condition_scale} known_logit:{args.known_logit_scale} "
        f"turn:{args.turn_condition_scale} turn_next:{args.turn_next_condition_scale} "
        f"type:{args.type_prior_scale} "
        f"cur_complex:{args.current_complexity_weight}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_cfg = SequenceLossConfig(
        action_weight=args.action_weight,
        current_action_weight=args.current_action_weight,
        prefix_action_weight=args.prefix_action_weight,
        order_weight=args.order_weight,
        multi_weight=args.multi_weight,
        count_weight=args.count_weight,
        plan_weight=args.plan_weight,
        next_type_weight=args.next_type_weight,
        dca_plan_weight=args.dca_plan_weight,
        known_action_weight=args.known_action_weight,
        turn_plan_weight=args.turn_plan_weight,
        turn_terminal_weight=args.turn_terminal_weight,
        turn_next_plan_weight=args.turn_next_plan_weight,
        turn_next_type_weight=args.turn_next_type_weight,
        turn_next_card_weight=args.turn_next_card_weight,
        turn_next_attack_weight=args.turn_next_attack_weight,
        turn_next_context_weight=args.turn_next_context_weight,
        opportunity_type_weight=args.opportunity_type_weight,
        opportunity_margin_weight=args.opportunity_margin_weight,
        opportunity_margin=args.opportunity_margin,
        current_rank_margin_weight=args.current_rank_margin_weight,
        current_rank_margin=args.current_rank_margin,
        current_rank_margin_min_options=args.current_rank_margin_min_options,
        current_complexity_weight=args.current_complexity_weight,
        outcome_weight=0.10,
        type_weight=0.10,
        multi_target_weight=args.multi_target_weight,
        damage_counter_weight=args.damage_counter_weight,
    )
    best = float("inf")
    best_path = args.out
    last_path = str(Path(args.out).with_suffix(".last.pt"))
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            corpus=corpus,
            ids=train_ids,
            device=device,
            optimizer=optimizer,
            loss_cfg=loss_cfg,
            batch_size=args.batch_size,
            epoch=epoch,
            amp=args.amp and device.type == "cuda",
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
            progress_every=args.progress_every,
            diagnostic_ablation=False,
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                model=model,
                corpus=corpus,
                ids=val_ids,
                device=device,
                optimizer=None,
                loss_cfg=loss_cfg,
                batch_size=args.batch_size,
                epoch=epoch,
                amp=args.amp and device.type == "cuda",
                grad_clip=0.0,
                max_batches=args.max_val_batches,
                progress_every=0,
                diagnostic_ablation=args.diagnostic_ablation,
            )
        val = val_loss.get("loss", float("inf"))
        print(
            f"done epoch {epoch}/{args.epochs} "
            f"train={train_loss.get('loss', 0):.4f} val={val:.4f} "
            f"train_top1={train_acc.get('top1', 0):.3f} val_top1={val_acc.get('top1', 0):.3f} "
            f"val_cur={val_acc.get('cur_top1', 0):.3f}/{val_acc.get('cur_type_acc', 0):.3f} "
            f"val_curW={val_loss.get('current_weight_share', 0):.3f} "
            f"val_curRow={val_loss.get('current_row_share', 0):.3f} "
            f"val_curCmp={val_loss.get('current_complexity_mean', 0):.2f}/"
            f"{val_loss.get('current_complexity_max', 0):.2f} "
            f"val_curQ=H{val_acc.get('cur_entropy', 0):.2f}/M{val_acc.get('cur_margin', 0):.2f}/"
            f"R{val_acc.get('cur_target_margin_best', 0):.2f}/"
            f"{val_acc.get('cur_rank_violation_025', 0):.2f}/"
            f"AR{val_acc.get('cur_ambig_target_margin_best', 0):.2f}/"
            f"F{val_acc.get('cur_forced_rate', 0):.2f}/NF{val_acc.get('cur_nonforced_top1', 0):.3f}/"
            f"B{val_acc.get('cur_bigopt_rate', 0):.2f}:{val_acc.get('cur_bigopt_top1', 0):.3f}/"
            f"A{val_acc.get('cur_ambig_type_rate', 0):.2f}:{val_acc.get('cur_ambig_type_top1', 0):.3f}/"
            f"EndT{val_acc.get('cur_target_end_rate', 0):.2f}/P{val_acc.get('cur_pred_end_rate', 0):.2f}/"
            f"NL{val_acc.get('cur_pred_end_when_nonend_legal', 0):.2f}/TN{val_acc.get('cur_pred_end_when_target_nonend', 0):.2f} "
            f"val_opp=at{val_acc.get('cur_attach_legal_rate', 0):.2f}/"
            f"{val_acc.get('cur_attach_target_if_legal', 0):.2f}/{val_acc.get('cur_attach_pred_if_legal', 0):.2f}/"
            f"{val_acc.get('cur_attach_miss_if_target', 0):.2f} "
            f"ev{val_acc.get('cur_evolve_legal_rate', 0):.2f}/"
            f"{val_acc.get('cur_evolve_target_if_legal', 0):.2f}/{val_acc.get('cur_evolve_pred_if_legal', 0):.2f}/"
            f"{val_acc.get('cur_evolve_miss_if_target', 0):.2f} "
            f"ak{val_acc.get('cur_attack_legal_rate', 0):.2f}/"
            f"{val_acc.get('cur_attack_target_if_legal', 0):.2f}/{val_acc.get('cur_attack_pred_if_legal', 0):.2f}/"
            f"{val_acc.get('cur_attack_miss_if_target', 0):.2f} "
            f"ab{val_acc.get('cur_ability_legal_rate', 0):.2f}/"
            f"{val_acc.get('cur_ability_target_if_legal', 0):.2f}/{val_acc.get('cur_ability_pred_if_legal', 0):.2f}/"
            f"{val_acc.get('cur_ability_miss_if_target', 0):.2f} "
            f"val_oppMass=at{val_acc.get('cur_attach_target_mass', 0):.2f}/"
            f"ev{val_acc.get('cur_evolve_target_mass', 0):.2f}/"
            f"ak{val_acc.get('cur_attack_target_mass', 0):.2f}/"
            f"ab{val_acc.get('cur_ability_target_mass', 0):.2f} "
            f"val_oppMargin=at{val_acc.get('cur_attach_target_margin_other', 0):.2f}/"
            f"ev{val_acc.get('cur_evolve_target_margin_other', 0):.2f}/"
            f"ak{val_acc.get('cur_attack_target_margin_other', 0):.2f}/"
            f"ab{val_acc.get('cur_ability_target_margin_other', 0):.2f} "
            f"val_seq={val_acc.get('seq_len_mean', 0):.1f}/{val_acc.get('seq_full_rate', 0):.2f}/"
            f"{val_acc.get('history_present_rate', 0):.2f} "
            f"val_known={val_acc.get('known_opp_rate', 0):.3f}/{val_acc.get('known_opp_slots_mean', 0):.2f}/"
            f"{val_acc.get('known_opp_count_mean', 0):.2f}/"
            f"{val_acc.get('known_action_top1', 0):.3f}/{val_loss.get('known_action', 0):.3f} "
            f"val_turn={val_acc.get('turn_continue_acc', 0):.3f}/"
            f"{val_acc.get('turn_continue_f1', 0):.3f}/"
            f"{val_acc.get('turn_continue_target_rate', 0):.2f}->{val_acc.get('turn_continue_pred_rate', 0):.2f}/"
            f"rem{val_acc.get('turn_remaining_mae', 0):.2f}/"
            f"typ{val_acc.get('turn_future_type_f1', 0):.3f}/"
            f"{val_loss.get('turn_plan', 0):.3f} "
            f"val_curTurn={val_acc.get('cur_turn_continue_f1', 0):.3f}/"
            f"{val_acc.get('cur_turn_continue_target_rate', 0):.2f}->{val_acc.get('cur_turn_continue_pred_rate', 0):.2f}/"
            f"miss{val_acc.get('cur_turn_continue_miss', 0):.2f}/"
            f"term{val_acc.get('cur_terminal_when_continue', 0):.2f}/"
            f"pT{val_acc.get('cur_terminal_prob_when_continue', 0):.2f}/"
            f"stopT{val_acc.get('cur_terminal_prob_when_stop', 0):.2f}/"
            f"over{val_acc.get('cur_nonterminal_when_stop', 0):.2f} "
            f"val_turnNext={val_acc.get('turn_next_exists_rate', 0):.2f}/"
            f"{val_acc.get('turn_next_type_acc', 0):.3f}/"
            f"{val_acc.get('turn_next_type_pos_acc', 0):.3f}/"
            f"none{val_acc.get('turn_next_none_acc', 0):.3f}/"
            f"card{val_acc.get('turn_next_card_acc', 0):.3f}/"
            f"atk{val_acc.get('turn_next_attack_acc', 0):.3f}/"
            f"ctx{val_acc.get('turn_next_context_acc', 0):.3f} "
            f"val_curNext={val_acc.get('cur_turn_next_exists_rate', 0):.2f}/"
            f"{val_acc.get('cur_turn_next_type_acc', 0):.3f}/"
            f"{val_acc.get('cur_turn_next_type_pos_acc', 0):.3f}/"
            f"card{val_acc.get('cur_turn_next_card_acc', 0):.3f}/"
            f"atk{val_acc.get('cur_turn_next_attack_acc', 0):.3f} "
            f"val_cur1={val_acc.get('cur1_top1', 0):.3f}/{val_acc.get('cur1_delta', 0):+.3f}/"
            f"{val_acc.get('cur1_agree', 0):.3f}/{val_acc.get('cur1_kl', 0):.3f} "
            f"val_stateless={val_acc.get('stateless_top1', 0):.3f}/{val_acc.get('stateless_delta', 0):+.3f}/"
            f"{val_acc.get('stateless_agree', 0):.3f}/{val_acc.get('stateless_kl', 0):.3f} "
            f"val_noact={val_acc.get('noact_top1', 0):.3f}/{val_acc.get('noact_delta', 0):+.3f}/"
            f"{val_acc.get('noact_agree', 0):.3f}/{val_acc.get('noact_kl', 0):.3f} "
            f"val_noknown={val_acc.get('noknown_top1', 0):.3f}/{val_acc.get('noknown_delta', 0):+.3f}/"
            f"{val_acc.get('noknown_agree', 0):.3f}/{val_acc.get('noknown_kl', 0):.3f} "
            f"val_revhist={val_acc.get('revhist_top1', 0):.3f}/{val_acc.get('revhist_delta', 0):+.3f}/"
            f"{val_acc.get('revhist_agree', 0):.3f}/{val_acc.get('revhist_kl', 0):.3f} "
            f"val_plan={val_loss.get('plan', 0):.4f} val_type={val_acc.get('type_acc', 0):.3f} "
            f"val_planSig={val_acc.get('plan_f1', 0):.3f}/{val_acc.get('plan_mae', 0):.3f}/"
            f"{val_acc.get('plan_pos_rate', 0):.2f}->{val_acc.get('plan_pred_pos_rate', 0):.2f} "
            f"val_next={val_acc.get('next_type_acc', 0):.3f}/{val_acc.get('next1_acc', 0):.3f}/"
            f"{val_acc.get('next2_acc', 0):.3f}/{val_acc.get('next3_acc', 0):.3f}/"
            f"{val_acc.get('next4_acc', 0):.3f} "
            f"val_atype={val_acc.get('action_type_acc', 0):.3f} "
            f"val_count={val_acc.get('count_acc', 0):.3f} "
            f"val_setF1={val_acc.get('set_f1', 0):.3f} val_order={val_acc.get('order_acc', 0):.3f} "
            f"val_k={val_acc.get('pred_k', 0):.2f}/{val_acc.get('target_k', 0):.2f} "
            f"val_opt={val_acc.get('option_n', 0):.1f}/{val_acc.get('ambig_type_rate', 0):.2f}/"
            f"{val_acc.get('ambig_type_top1', 0):.3f} "
            f"val_m2n={val_acc.get('multi2_n', 0):.0f} val_m2r={val_acc.get('multi2_rate', 0):.3f} "
            f"val_cap2={val_acc.get('capable2_rate', 0):.3f} "
            f"val_m2F1={val_acc.get('multi2_f1', 0):.3f} val_m2Order={val_acc.get('multi2_order_acc', 0):.3f} "
            f"val_dcaN={val_acc.get('dca_n', 0):.0f} val_dcaR={val_acc.get('dca_rate', 0):.3f} "
            f"val_dcaTop1={val_acc.get('dca_top1', 0):.3f} val_dcaF1={val_acc.get('dca_f1', 0):.3f} "
            f"val_dcaCnt={val_acc.get('dca_count_acc', 0):.3f}/{val_acc.get('dca_count_mae', 0):.2f} "
            f"val_dcaK={val_acc.get('dca_pred_k', 0):.2f}/{val_acc.get('dca_target_k', 0):.2f} "
            f"val_dcaBlk={val_acc.get('dca_focus_mean', 0):.2f}/{val_acc.get('dca_spread_rate', 0):.2f}/"
            f"{val_acc.get('dca_prior_unique_mean', 0):.2f} "
            f"val_dcaSplit={val_acc.get('dca_first_top1', 0):.3f}/{val_acc.get('dca_late_top1', 0):.3f}/"
            f"{val_acc.get('dca_spread_top1', 0):.3f} "
            f"val_dplan={val_acc.get('dca_plan_spread_f1', 0):.3f}/"
            f"{val_acc.get('dca_plan_unique_acc', 0):.3f}/"
            f"{val_acc.get('dca_plan_focus_mae', 0):.3f}/"
            f"{val_acc.get('dca_plan_spread_rate', 0):.2f}->{val_acc.get('dca_plan_pred_spread_rate', 0):.2f} "
            f"val_boost={val_loss.get('weight_boost', 0):.2f} "
            f"val_dcaW={val_loss.get('dca_weight_share', 0):.2f} val_m2W={val_loss.get('multi_weight_share', 0):.2f} "
            f"val_nextL={val_loss.get('next_type', 0):.3f} val_dplanL={val_loss.get('dca_plan', 0):.3f} "
            f"val_knownL={val_loss.get('known_action', 0):.3f} "
            f"val_turnTermL={val_loss.get('turn_terminal', 0):.3f} "
            f"val_turnNextL={val_loss.get('turn_next_plan', 0):.3f}/"
            f"{val_loss.get('turn_next_type', 0):.3f}/"
            f"{val_loss.get('turn_next_card', 0):.3f}/"
            f"{val_loss.get('turn_next_attack', 0):.3f}/"
            f"{val_loss.get('turn_next_context', 0):.3f} "
            f"val_oppTypeL={val_loss.get('opportunity_type', 0):.3f}/"
            f"{val_loss.get('opportunity_type_rows', 0):.0f}/"
            f"{val_loss.get('opportunity_key_share', 0):.2f} "
            f"val_oppMarginL={val_loss.get('opportunity_margin', 0):.3f}/"
            f"{val_loss.get('opportunity_margin_rows', 0):.0f}/"
            f"{val_loss.get('opportunity_margin_violation', 0):.2f} "
            f"val_rankMarginL={val_loss.get('current_rank_margin', 0):.3f}/"
            f"{val_loss.get('current_rank_margin_rows', 0):.0f}/"
            f"{val_loss.get('current_rank_margin_violation', 0):.2f} "
            f"val_dposW={val_loss.get('dca_spread_pos_weight', 0):.1f} "
            f"val_curA={val_loss.get('current_action', 0):.3f}/{val_loss.get('prefix_action', 0):.3f} "
            f"val_curHead={val_loss.get('action_current_head', 0):.3f}/{val_loss.get('action_prefix_head', 0):.3f} "
            f"val_dcaA={val_loss.get('dca_action', 0):.3f}/{val_loss.get('non_dca_action', 0):.3f} "
            f"val_cond=h{val_loss.get('hist_query_ratio', 0):.2f}/p{val_loss.get('plan_query_ratio', 0):.2f}/"
            f"n{val_loss.get('next_query_ratio', 0):.2f}/d{val_loss.get('dca_query_ratio', 0):.2f}/"
            f"k{val_loss.get('known_query_ratio', 0):.2f}/u{val_loss.get('turn_query_ratio', 0):.2f}/"
            f"tn{val_loss.get('turn_next_query_ratio', 0):.2f}/"
            f"q{val_loss.get('conditioned_query_ratio', 0):.2f}/"
            f"t{val_loss.get('type_prior_abs', 0):.2f} "
            f"val_out={val_acc.get('outcome_acc', 0):.3f}/{val_acc.get('outcome_brier', 0):.3f}/"
            f"{val_acc.get('outcome_pos_rate', 0):.2f}->{val_acc.get('outcome_pred_pos_rate', 0):.2f} "
            f"val_types=p{val_acc.get('play_top1', 0):.2f},at{val_acc.get('attach_top1', 0):.2f},"
            f"ev{val_acc.get('evolve_top1', 0):.2f},ab{val_acc.get('ability_top1', 0):.2f},"
            f"ak{val_acc.get('attack_top1', 0):.2f},en{val_acc.get('end_top1', 0):.2f}",
            flush=True,
        )
        warnings = _signal_warnings(
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            args=args,
        )
        print(
            "signal_health "
            f"top1_gap={train_acc.get('top1', 0) - val_acc.get('top1', 0):.3f} "
            f"m2_gap={train_acc.get('multi2_f1', 0) - val_acc.get('multi2_f1', 0):.3f} "
            f"dca_gap={train_acc.get('dca_top1', 0) - val_acc.get('dca_top1', 0):.3f} "
            f"cur_gap={train_acc.get('cur_top1', 0) - val_acc.get('cur_top1', 0):.3f} "
            f"cur1_delta={val_acc.get('cur1_delta', 0):+.3f} "
            f"cur1_agree={val_acc.get('cur1_agree', 0):.3f} "
            f"stateless_delta={val_acc.get('stateless_delta', 0):+.3f} "
            f"stateless_agree={val_acc.get('stateless_agree', 0):.3f} "
            f"noact_delta={val_acc.get('noact_delta', 0):+.3f} "
            f"noact_agree={val_acc.get('noact_agree', 0):.3f} "
            f"noknown_delta={val_acc.get('noknown_delta', 0):+.3f} "
            f"noknown_agree={val_acc.get('noknown_agree', 0):.3f} "
            f"revhist_delta={val_acc.get('revhist_delta', 0):+.3f} "
            f"revhist_agree={val_acc.get('revhist_agree', 0):.3f} "
            f"noact_nextD={val_acc.get('noact_next_delta', 0):+.3f} "
            f"revhist_nextD={val_acc.get('revhist_next_delta', 0):+.3f} "
            f"revhist_nextKL={val_acc.get('revhist_next_kl', 0):.4f} "
            f"revhist_planD={val_acc.get('revhist_plan_delta', 0):+.3f} "
            f"revhist_planL1={val_acc.get('revhist_plan_l1', 0):.4f} "
            f"revhist_dplanD={val_acc.get('revhist_dplan_delta', 0):+.3f} "
            f"plan_f1={val_acc.get('plan_f1', 0):.3f} "
            f"next_acc={val_acc.get('next_type_acc', 0):.3f} "
            f"dplan_f1={val_acc.get('dca_plan_spread_f1', 0):.3f} "
            f"ambig_top1={val_acc.get('ambig_type_top1', 0):.3f} "
            f"train_grad={train_loss.get('grad_norm', 0):.2f} "
            f"grad_skip={train_loss.get('grad_nonfinite', 0):.3f} "
            f"dca_weight_share={val_loss.get('dca_weight_share', 0):.3f} "
            f"multi_weight_share={val_loss.get('multi_weight_share', 0):.3f} "
            f"current_weight_share={val_loss.get('current_weight_share', 0):.3f} "
            f"current_row_share={val_loss.get('current_row_share', 0):.3f} "
            f"current_complexity={val_loss.get('current_complexity_mean', 0):.3f}/"
            f"{val_loss.get('current_complexity_max', 0):.3f} "
            f"cur_entropy={val_acc.get('cur_entropy', 0):.3f} "
            f"cur_margin={val_acc.get('cur_margin', 0):.3f} "
            f"cur_rank_margin={val_acc.get('cur_target_margin_best', 0):.3f}/"
            f"{val_acc.get('cur_rank_violation_025', 0):.3f}/"
            f"{val_acc.get('cur_ambig_target_margin_best', 0):.3f} "
            f"cur_forced={val_acc.get('cur_forced_rate', 0):.3f} "
            f"cur_nonforced={val_acc.get('cur_nonforced_top1', 0):.3f} "
            f"cur_bigopt={val_acc.get('cur_bigopt_rate', 0):.3f}/{val_acc.get('cur_bigopt_top1', 0):.3f} "
            f"cur_ambig={val_acc.get('cur_ambig_type_rate', 0):.3f}/{val_acc.get('cur_ambig_type_top1', 0):.3f} "
            f"cur_end={val_acc.get('cur_target_end_rate', 0):.3f}/{val_acc.get('cur_pred_end_rate', 0):.3f}/"
            f"{val_acc.get('cur_pred_end_when_nonend_legal', 0):.3f}/{val_acc.get('cur_pred_end_when_target_nonend', 0):.3f} "
            f"cur_opp_at={val_acc.get('cur_attach_legal_rate', 0):.3f}/"
            f"{val_acc.get('cur_attach_target_if_legal', 0):.3f}/{val_acc.get('cur_attach_pred_if_legal', 0):.3f}/"
            f"{val_acc.get('cur_attach_miss_if_target', 0):.3f} "
            f"cur_opp_ev={val_acc.get('cur_evolve_legal_rate', 0):.3f}/"
            f"{val_acc.get('cur_evolve_target_if_legal', 0):.3f}/{val_acc.get('cur_evolve_pred_if_legal', 0):.3f}/"
            f"{val_acc.get('cur_evolve_miss_if_target', 0):.3f} "
            f"cur_opp_ak={val_acc.get('cur_attack_legal_rate', 0):.3f}/"
            f"{val_acc.get('cur_attack_target_if_legal', 0):.3f}/{val_acc.get('cur_attack_pred_if_legal', 0):.3f}/"
            f"{val_acc.get('cur_attack_miss_if_target', 0):.3f} "
            f"cur_opp_mass=at{val_acc.get('cur_attach_target_mass', 0):.3f}/"
            f"ev{val_acc.get('cur_evolve_target_mass', 0):.3f}/"
            f"ab{val_acc.get('cur_ability_target_mass', 0):.3f}/"
            f"ak{val_acc.get('cur_attack_target_mass', 0):.3f} "
            f"cur_opp_margin=at{val_acc.get('cur_attach_target_margin_other', 0):.3f}/"
            f"ev{val_acc.get('cur_evolve_target_margin_other', 0):.3f}/"
            f"ab{val_acc.get('cur_ability_target_margin_other', 0):.3f}/"
            f"ak{val_acc.get('cur_attack_target_margin_other', 0):.3f} "
            f"opp_margin_loss={val_loss.get('opportunity_margin', 0):.3f}/"
            f"{val_loss.get('opportunity_margin_rows', 0):.0f}/"
            f"{val_loss.get('opportunity_margin_violation', 0):.3f} "
            f"rank_margin_loss={val_loss.get('current_rank_margin', 0):.3f}/"
            f"{val_loss.get('current_rank_margin_rows', 0):.0f}/"
            f"{val_loss.get('current_rank_margin_violation', 0):.3f} "
            f"known_rate={val_acc.get('known_opp_rate', 0):.3f} "
            f"known_slots={val_acc.get('known_opp_slots_mean', 0):.2f} "
            f"known_action_top1={val_acc.get('known_action_top1', 0):.3f} "
            f"turn_acc={val_acc.get('turn_continue_acc', 0):.3f} "
            f"turn_f1={val_acc.get('turn_continue_f1', 0):.3f} "
            f"turn_rate={val_acc.get('turn_continue_target_rate', 0):.3f}->{val_acc.get('turn_continue_pred_rate', 0):.3f} "
            f"turn_rem_mae={val_acc.get('turn_remaining_mae', 0):.3f} "
            f"turn_type_f1={val_acc.get('turn_future_type_f1', 0):.3f} "
            f"turn_next={val_acc.get('turn_next_exists_rate', 0):.3f}/"
            f"{val_acc.get('turn_next_type_acc', 0):.3f}/"
            f"{val_acc.get('turn_next_type_pos_acc', 0):.3f}/"
            f"none{val_acc.get('turn_next_none_acc', 0):.3f}/"
            f"card{val_acc.get('turn_next_card_acc', 0):.3f}/"
            f"atk{val_acc.get('turn_next_attack_acc', 0):.3f}/"
            f"ctx{val_acc.get('turn_next_context_acc', 0):.3f} "
            f"cur_turn={val_acc.get('cur_turn_continue_f1', 0):.3f}/"
            f"{val_acc.get('cur_turn_continue_target_rate', 0):.3f}->{val_acc.get('cur_turn_continue_pred_rate', 0):.3f}/"
            f"miss{val_acc.get('cur_turn_continue_miss', 0):.3f}/"
            f"term{val_acc.get('cur_terminal_when_continue', 0):.3f}/"
            f"pterm{val_acc.get('cur_terminal_prob_when_continue', 0):.3f}/"
            f"stopT{val_acc.get('cur_terminal_prob_when_stop', 0):.3f}/"
            f"over{val_acc.get('cur_nonterminal_when_stop', 0):.3f} "
            f"cur_turn_next={val_acc.get('cur_turn_next_exists_rate', 0):.3f}/"
            f"{val_acc.get('cur_turn_next_type_acc', 0):.3f}/"
            f"{val_acc.get('cur_turn_next_type_pos_acc', 0):.3f}/"
            f"card{val_acc.get('cur_turn_next_card_acc', 0):.3f}/"
            f"atk{val_acc.get('cur_turn_next_attack_acc', 0):.3f} "
            f"cond=h{val_loss.get('hist_query_ratio', 0):.3f}/p{val_loss.get('plan_query_ratio', 0):.3f}/"
            f"n{val_loss.get('next_query_ratio', 0):.3f}/d{val_loss.get('dca_query_ratio', 0):.3f}/"
            f"k{val_loss.get('known_query_ratio', 0):.3f}/u{val_loss.get('turn_query_ratio', 0):.3f}/"
            f"tn{val_loss.get('turn_next_query_ratio', 0):.3f}/"
            f"q{val_loss.get('conditioned_query_ratio', 0):.3f}/"
            f"t{val_loss.get('type_prior_abs', 0):.3f} "
            f"warnings={','.join(warnings) if warnings else 'none'}",
            flush=True,
        )
        save_checkpoint(last_path, model=model, args=args, corpus=corpus, epoch=epoch, val_loss=val)
        _run_random_smoke(args, last_path, epoch)
        if val < best:
            best = val
            save_checkpoint(best_path, model=model, args=args, corpus=corpus, epoch=epoch, val_loss=val)
            print(f"  saved best {best:.4f} -> {best_path}", flush=True)
    print(f"Training complete best={best:.4f} checkpoint={best_path}", flush=True)


if __name__ == "__main__":
    main()
