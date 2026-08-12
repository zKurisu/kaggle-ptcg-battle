#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

if os.environ.get("PTCG_DISABLE_CUDNN"):
    torch.backends.cudnn.enabled = False

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.bc2 import BCCorpus, discover_npz_paths, sequence_loss_parts, sequence_nll
from ptcg_rl.deck_plans import CARD_NAMES
from ptcg_rl.history_features import BOARD_HISTORY_FEAT_DIM
from ptcg_rl.model import (
    build_policy_model,
    checkpoint_board_history_dims,
    checkpoint_feature_dims,
    checkpoint_history_k,
    checkpoint_log_history_k,
    checkpoint_opp_history_k,
)
from ptcg_rl.plan_labels import PLAN_LABELS


CONTEXT_IDS = {
    "MAIN": 0,
    "SETUP_ACTIVE": 1,
    "SETUP_BENCH": 2,
    "SWITCH": 3,
    "TO_ACTIVE": 4,
    "TO_BENCH": 5,
    "TO_FIELD": 6,
    "TO_HAND": 7,
    "DISCARD": 8,
    "TO_DECK": 9,
    "TO_DECK_BOTTOM": 10,
    "TO_PRIZE": 11,
    "NOT_MOVE": 12,
    "DAMAGE_COUNTER": 13,
    "DAMAGE_COUNTER_ANY": 14,
    "DAMAGE": 15,
    "REMOVE_DAMAGE_COUNTER": 16,
    "HEAL": 17,
    "EVOLVES_FROM": 18,
    "EVOLVES_TO": 19,
    "DEVOLVE": 20,
    "ATTACH_FROM": 21,
    "ATTACH_TO": 22,
    "DETACH_FROM": 23,
    "LOOK": 24,
    "EFFECT_TARGET": 25,
    "DISCARD_ENERGY_CARD": 26,
    "DISCARD_TOOL_CARD": 27,
    "SWITCH_ENERGY_CARD": 28,
    "DISCARD_CARD_OR_ATTACHED_CARD": 29,
    "DISCARD_ENERGY": 30,
    "TO_HAND_ENERGY": 31,
    "TO_DECK_ENERGY": 32,
    "SWITCH_ENERGY": 33,
    "SKILL_ORDER": 34,
    "ATTACK": 35,
    "DISABLE_ATTACK": 36,
    "EVOLVE": 37,
    "DRAW_COUNT": 38,
    "DAMAGE_COUNTER_COUNT": 39,
    "REMOVE_DAMAGE_COUNTER_COUNT": 40,
    "IS_FIRST": 41,
    "MULLIGAN": 42,
    "ACTIVATE": 43,
}

TYPE_IDS = {
    "NUMBER": 0,
    "YES": 1,
    "NO": 2,
    "CARD": 3,
    "TOOL_CARD": 4,
    "ENERGY_CARD": 5,
    "ENERGY": 6,
    "PLAY": 7,
    "ATTACH": 8,
    "EVOLVE": 9,
    "ABILITY": 10,
    "DISCARD": 11,
    "RETREAT": 12,
    "ATTACK": 13,
    "END": 14,
    "SKILL": 15,
    "SPECIAL_CONDITION": 16,
}

CARD_IDS = {str(name).upper(): int(card_id) for card_id, name in CARD_NAMES.items()}


def _parse_weight_specs(specs: list[str], names: dict[str, int], label: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"{label} weight must be NAME=WEIGHT or ID=WEIGHT, got {spec!r}")
        key, value = spec.split("=", 1)
        key = key.strip().upper()
        if key.lstrip("-").isdigit():
            idx = int(key)
        elif key in names:
            idx = names[key]
        else:
            known = ", ".join(sorted(names)[:12])
            raise ValueError(f"unknown {label} {key!r}; use numeric id or one of: {known}, ...")
        out[idx] = float(value)
    return out


def _parse_text_weight_specs(specs: list[str], label: str, *, lower: bool = False) -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"{label} weight must be NAME=WEIGHT, got {spec!r}")
        key, value = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{label} weight has an empty key in {spec!r}")
        out[key.lower() if lower else key] = float(value)
    return out


def _to_float(value: object) -> tuple[bool, float]:
    try:
        return True, float(value)
    except Exception:
        return False, 0.0


def _trajectory_key(row: dict[str, str]) -> str:
    key = str(row.get("game_key", "")).strip()
    if key:
        return key
    episode_id = str(row.get("episode_id", "")).strip()
    player_index = str(row.get("player_index", "")).strip()
    if episode_id and player_index:
        return f"{episode_id}:{player_index}"
    raise ValueError("trajectory CSV row must contain game_key or episode_id/player_index")


def _trajectory_condition(row: dict[str, str], expr: str) -> bool:
    expr = expr.strip()
    if not expr:
        return False
    if "|" in expr:
        return any(_trajectory_condition(row, part) for part in expr.split("|"))
    if "&" in expr:
        return all(_trajectory_condition(row, part) for part in expr.split("&"))
    if expr.startswith("!"):
        return not _trajectory_condition(row, expr[1:])
    for op in (">=", "<=", "==", "!=", ">", "<"):
        if op not in expr:
            continue
        col, rhs = expr.split(op, 1)
        col = col.strip()
        rhs = rhs.strip()
        lhs_raw = str(row.get(col, "")).strip()
        lhs_ok, lhs = _to_float(lhs_raw)
        rhs_ok, rhs_value = _to_float(rhs)
        if lhs_ok and rhs_ok:
            if op == ">=":
                return lhs >= rhs_value
            if op == "<=":
                return lhs <= rhs_value
            if op == "==":
                return lhs == rhs_value
            if op == "!=":
                return lhs != rhs_value
            if op == ">":
                return lhs > rhs_value
            if op == "<":
                return lhs < rhs_value
        if op == "==":
            return lhs_raw == rhs
        if op == "!=":
            return lhs_raw != rhs
        raise ValueError(f"non-numeric trajectory condition cannot use {op!r}: {expr!r}")
    ok, value = _to_float(row.get(expr, 0.0))
    return ok and value > 0.0


def _is_trajectory_condition(expr: str) -> bool:
    return any(op in expr for op in (">=", "<=", "==", "!=", ">", "<", "&", "|", "!"))


def _parse_trajectory_weight_specs(specs: list[str]) -> list[tuple[str, float]]:
    parsed: list[tuple[str, float]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                "trajectory weight must be CONDITION=WEIGHT, e.g. attack_by_4=1.4 "
                "or attack_count>=5=1.2"
            )
        expr, weight = spec.rsplit("=", 1)
        expr = expr.strip()
        if not expr:
            raise ValueError(f"trajectory weight has an empty condition in {spec!r}")
        parsed.append((expr, float(weight)))
    return parsed


def _load_trajectory_weights(
    paths: list[str],
    specs: list[str],
    *,
    base_weight: float,
    cap: float,
) -> tuple[dict[str, float], dict[str, float]]:
    if not paths:
        return {}, {}
    parsed = _parse_trajectory_weight_specs(specs)
    weights: dict[str, float] = {}
    stats = {
        "files": float(len(paths)),
        "rows": 0.0,
        "keys": 0.0,
        "matched_conditions": 0.0,
        "duplicates": 0.0,
        "min_weight": 0.0,
        "max_weight": 0.0,
        "mean_weight": 0.0,
    }
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["rows"] += 1.0
                key = _trajectory_key(row)
                weight = float(base_weight)
                for expr, mult in parsed:
                    if _trajectory_condition(row, expr):
                        weight *= float(mult)
                        stats["matched_conditions"] += 1.0
                if cap > 0:
                    weight = min(weight, float(cap))
                if key in weights:
                    stats["duplicates"] += 1.0
                    weights[key] = max(weights[key], weight)
                else:
                    weights[key] = weight
    stats["keys"] = float(len(weights))
    if weights:
        arr = np.asarray(list(weights.values()), dtype=np.float32)
        stats["min_weight"] = float(arr.min())
        stats["max_weight"] = float(arr.max())
        stats["mean_weight"] = float(arr.mean())
    return weights, stats


def _load_trajectory_targets(
    paths: list[str],
    columns: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if not paths or not columns:
        return {}, {}
    clean_columns = [c.strip() for c in columns if c.strip()]
    if not clean_columns:
        return {}, {}
    targets: dict[str, np.ndarray] = {}
    stats = {
        "files": float(len(paths)),
        "rows": 0.0,
        "keys": 0.0,
        "duplicates": 0.0,
        "target_dim": float(len(clean_columns)),
    }
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["rows"] += 1.0
                key = _trajectory_key(row)
                values = []
                for col in clean_columns:
                    if _is_trajectory_condition(col):
                        values.append(1.0 if _trajectory_condition(row, col) else 0.0)
                    else:
                        ok, value = _to_float(row.get(col, 0.0))
                        values.append(float(value) if ok else 0.0)
                arr = np.asarray(values, dtype=np.float32)
                if key in targets:
                    stats["duplicates"] += 1.0
                    targets[key] = np.maximum(targets[key], arr)
                else:
                    targets[key] = arr
    stats["keys"] = float(len(targets))
    return targets, stats


def _load_trajectory_filter(
    paths: list[str],
    keep_specs: list[str],
    drop_specs: list[str],
) -> tuple[set[str] | None, dict[str, float]]:
    keep_specs = [str(x).strip() for x in keep_specs if str(x).strip()]
    drop_specs = [str(x).strip() for x in drop_specs if str(x).strip()]
    if not keep_specs and not drop_specs:
        return None, {}
    if not paths:
        raise ValueError("--trajectory-keep/--trajectory-drop requires at least one --trajectory-csv")
    allowed: set[str] = set()
    seen: set[str] = set()
    stats = {
        "files": float(len(paths)),
        "rows": 0.0,
        "seen_keys": 0.0,
        "allowed_keys": 0.0,
        "kept_rows": 0.0,
        "dropped_rows": 0.0,
        "duplicates": 0.0,
        "keep_conditions": float(len(keep_specs)),
        "drop_conditions": float(len(drop_specs)),
    }
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["rows"] += 1.0
                key = _trajectory_key(row)
                if key in seen:
                    stats["duplicates"] += 1.0
                seen.add(key)
                keep_ok = all(_trajectory_condition(row, expr) for expr in keep_specs)
                drop_ok = any(_trajectory_condition(row, expr) for expr in drop_specs)
                if keep_ok and not drop_ok:
                    allowed.add(key)
                    stats["kept_rows"] += 1.0
                else:
                    stats["dropped_rows"] += 1.0
    stats["seen_keys"] = float(len(seen))
    stats["allowed_keys"] = float(len(allowed))
    if not allowed:
        raise RuntimeError(
            "trajectory filter kept no games; relax --trajectory-keep/--trajectory-drop conditions"
        )
    return allowed, stats


def _save_npz(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})


def _load_npz_init(
    model: torch.nn.Module,
    path: str,
    device: torch.device,
    *,
    partial: bool = False,
    skip_prefixes: tuple[str, ...] = (),
) -> tuple[int, list[str]]:
    with np.load(path) as z:
        checkpoint = {
            k: torch.as_tensor(z[k], device=device)
            for k in z.files
        }
    skip_prefixes = tuple(x for x in skip_prefixes if x)
    if not partial:
        if skip_prefixes:
            raise ValueError("init skip prefixes require partial init")
        current = model.state_dict()
        extra = [k for k in checkpoint if k not in current]
        missing = [k for k in current if k not in checkpoint]
        shape_mismatch = [
            k for k, v in checkpoint.items()
            if k in current and tuple(v.shape) != tuple(current[k].shape)
        ]
        if extra and all(k.startswith("plan_") for k in extra) and not missing and not shape_mismatch:
            filtered = {k: v for k, v in checkpoint.items() if k in current}
            model.load_state_dict(filtered, strict=True)
            return len(filtered), [f"ignored auxiliary plan tensors: {len(extra)}"]
        model.load_state_dict(checkpoint, strict=True)
        return len(checkpoint), []

    current = model.state_dict()
    loaded = {}
    skipped: list[str] = []
    for key, tensor in checkpoint.items():
        if skip_prefixes and any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped.append(f"{key}: skipped by prefix")
            continue
        if key not in current:
            skipped.append(f"{key}: unexpected")
            continue
        if tuple(tensor.shape) != tuple(current[key].shape):
            skipped.append(f"{key}: {tuple(tensor.shape)} != {tuple(current[key].shape)}")
            continue
        loaded[key] = tensor.to(dtype=current[key].dtype)
    if not loaded:
        raise RuntimeError(f"No compatible tensors found in --init checkpoint: {path}")
    current.update(loaded)
    model.load_state_dict(current, strict=True)
    return len(loaded), skipped


def _checkpoint_feature_dims(path: str) -> tuple[int, int]:
    with np.load(path) as z:
        state_feat_dim, opt_feat_dim, _, _ = checkpoint_feature_dims(z)
    return int(state_feat_dim), int(opt_feat_dim)


def _checkpoint_history_k(path: str) -> int:
    with np.load(path) as z:
        return checkpoint_history_k(z)


def _checkpoint_history_dims(path: str) -> tuple[int, int, int, int, int]:
    with np.load(path) as z:
        board_k, board_feat_dim = checkpoint_board_history_dims(z)
        return (
            checkpoint_history_k(z),
            checkpoint_opp_history_k(z),
            checkpoint_log_history_k(z),
            board_k,
            board_feat_dim,
        )


def _configure_cuda_memory_limit(device: torch.device, *, gb: float = 0.0, fraction: float = 0.0) -> str:
    if device.type != "cuda":
        return ""
    if gb <= 0 and fraction <= 0:
        return ""
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / (1024 ** 3)
    if gb > 0:
        fraction = float(gb) / max(total_gb, 1e-9)
    fraction = min(max(float(fraction), 0.01), 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    limit_gb = total_gb * fraction
    return f"cuda_memory_limit={limit_gb:.2f}GB/{total_gb:.1f}GB fraction={fraction:.3f}"


def _metric_value(metrics: dict[str, float], name: str) -> float:
    value = metrics.get(name, float("nan"))
    return float(value)


def _format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    keys = [
        "loss",
        "policy",
        "policy_raw",
        "first_action_raw",
        "first_action_acc",
        "set",
        "trajectory",
        "step_plan",
        "value",
    ]
    parts = [prefix]
    for key in keys:
        value = metrics.get(key)
        if value is None or not np.isfinite(value):
            continue
        if key.endswith("_acc"):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value:.4f}")
    return " ".join(parts)


def _empty_metric_sums() -> dict[str, float]:
    return {
        "loss_sum": 0.0,
        "batch_weight": 0.0,
        "policy_sum": 0.0,
        "policy_weight": 0.0,
        "policy_raw_sum": 0.0,
        "policy_raw_count": 0.0,
        "first_action_sum": 0.0,
        "first_action_weight": 0.0,
        "first_action_raw_sum": 0.0,
        "first_count": 0.0,
        "first_action_correct": 0.0,
        "set_sum": 0.0,
        "set_weight": 0.0,
        "value_sum": 0.0,
        "value_weight": 0.0,
        "trajectory_sum": 0.0,
        "trajectory_count": 0.0,
        "step_plan_sum": 0.0,
        "step_plan_count": 0.0,
        "rows": 0.0,
    }


def _update_metric_sums(sums: dict[str, float], parts: dict[str, torch.Tensor]) -> None:
    rows = float(parts["rows"].detach().cpu())
    policy_weight = float(parts["policy_weight"].detach().cpu())
    policy_raw_count = float(parts["policy_raw_count"].detach().cpu())
    first_weight = float(parts["first_weight"].detach().cpu())
    first_count = float(parts["first_count"].detach().cpu())
    trajectory_count = float(parts["trajectory_count"].detach().cpu())
    step_plan_count = float(parts["step_plan_count"].detach().cpu())
    loss = float(parts["loss"].detach().cpu())
    policy = float(parts["policy"].detach().cpu())
    policy_raw = float(parts["policy_raw"].detach().cpu())
    first_action = float(parts["first_action"].detach().cpu())
    first_action_raw = float(parts["first_action_raw"].detach().cpu())
    first_action_acc = float(parts["first_action_acc"].detach().cpu())
    set_loss = float(parts["set"].detach().cpu())
    value_loss = float(parts["value"].detach().cpu())
    trajectory_loss = float(parts["trajectory"].detach().cpu())
    step_plan_loss = float(parts["step_plan"].detach().cpu())

    sums["loss_sum"] += loss * rows
    sums["batch_weight"] += rows
    sums["policy_sum"] += policy * policy_weight
    sums["policy_weight"] += policy_weight
    sums["policy_raw_sum"] += policy_raw * policy_raw_count
    sums["policy_raw_count"] += policy_raw_count
    sums["first_action_sum"] += first_action * first_weight
    sums["first_action_weight"] += first_weight
    sums["first_action_raw_sum"] += first_action_raw * first_count
    sums["first_count"] += first_count
    sums["first_action_correct"] += first_action_acc * first_count
    sums["set_sum"] += set_loss * rows
    sums["set_weight"] += rows
    sums["value_sum"] += value_loss * rows
    sums["value_weight"] += rows
    sums["trajectory_sum"] += trajectory_loss * trajectory_count
    sums["trajectory_count"] += trajectory_count
    sums["step_plan_sum"] += step_plan_loss * step_plan_count
    sums["step_plan_count"] += step_plan_count
    sums["rows"] += rows


def _finalize_metric_sums(sums: dict[str, float]) -> dict[str, float]:
    def div(num: str, den: str) -> float:
        d = sums.get(den, 0.0)
        return sums.get(num, 0.0) / d if d > 0 else float("nan")

    return {
        "loss": div("loss_sum", "batch_weight"),
        "policy": div("policy_sum", "policy_weight"),
        "policy_raw": div("policy_raw_sum", "policy_raw_count"),
        "first_action": div("first_action_sum", "first_action_weight"),
        "first_action_raw": div("first_action_raw_sum", "first_count"),
        "first_action_acc": div("first_action_correct", "first_count"),
        "set": div("set_sum", "set_weight"),
        "value": div("value_sum", "value_weight"),
        "trajectory": div("trajectory_sum", "trajectory_count"),
        "step_plan": div("step_plan_sum", "step_plan_count"),
        "rows": sums.get("rows", 0.0),
    }


def _run_epoch(model, corpus, indices, batch_size, device, optimizer=None,
               first_action_weight=1.0, raw_policy_loss_weight=0.0,
               value_weight=0.0,
               plan_weight=0.0, step_plan_weight=0.0,
               plan_teacher_forcing=0.0,
               set_loss_weight=0.0, set_loss_min_count=2,
               set_loss_negative_weight=0.25):
    training = optimizer is not None
    model.train(training)
    sums = _empty_metric_sums()
    steps = 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        if len(batch_idx) < 2:
            continue
        batch = corpus.collate(batch_idx, device)
        parts = sequence_loss_parts(
            model,
            batch,
            first_action_weight=first_action_weight,
            raw_policy_loss_weight=raw_policy_loss_weight,
            value_weight=value_weight,
            plan_weight=plan_weight,
            step_plan_weight=step_plan_weight,
            plan_teacher_forcing=plan_teacher_forcing,
            set_loss_weight=set_loss_weight,
            set_loss_min_count=set_loss_min_count,
            set_loss_negative_weight=set_loss_negative_weight,
        )
        loss = parts["loss"]
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        _update_metric_sums(sums, parts)
        steps += 1
    metrics = _finalize_metric_sums(sums)
    return metrics, steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/bc_corpus_banded_v4")
    parser.add_argument("--aux-corpus", action="append", default=[],
                        help="extra corpus root mixed into training, e.g. generated rollout BC; repeatable")
    parser.add_argument("--aux-score-bands", nargs="+", default=["generated"],
                        help="score bands to read from each --aux-corpus")
    parser.add_argument("--aux-archetype", default="",
                        help="archetype name for --aux-corpus; defaults to --archetype")
    parser.add_argument("--aux-repeat", type=int, default=1,
                        help="repeat aux corpus paths N times as a coarse sample multiplier")
    parser.add_argument("--archetype", default="Marnie Grimmsnarl")
    parser.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])
    parser.add_argument("--date-from", default="",
                        help="keep only corpus npz files whose filename date is >= YYYY-MM-DD")
    parser.add_argument("--date-to", default="",
                        help="keep only corpus npz files whose filename date is <= YYYY-MM-DD")
    parser.add_argument("--deck-sig", action="append", default=[],
                        help="filter to one or more deck signatures; repeatable. Requires freshly extracted corpus metadata.")
    parser.add_argument("--team-name", action="append", default=[],
                        help="filter to one or more exact team names; repeatable. Use with --deck-sig for trajectory specialists.")
    parser.add_argument("--opponent-deck-sig", action="append", default=[],
                        help="filter to decisions from games against one or more opponent deck signatures")
    parser.add_argument("--opponent-archetype", action="append", default=[],
                        help="filter to decisions from games against one or more opponent archetypes")
    parser.add_argument("--opponent-team-name", action="append", default=[],
                        help="filter to decisions from games against one or more exact opponent team names")
    parser.add_argument("--opponent-deck-sig-weight", action="append", default=[],
                        help="repeatable opponent deck signature multiplier without filtering, e.g. 697a82e582d5=2.0")
    parser.add_argument("--opponent-archetype-weight", action="append", default=[],
                        help="repeatable opponent archetype multiplier without filtering, e.g. 'Crustle Wall=2.0'")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--arch", choices=["pointer", "cross_attn"], default="pointer",
                        help="policy architecture; pointer is the legacy MLP, cross_attn tokenizes board/hand and cross-attends options")
    parser.add_argument("--state-layers", type=int, default=2,
                        help="number of state self-attention layers for --arch cross_attn")
    parser.add_argument("--hierarchical-plan", action="store_true",
                        help=(
                            "condition action logits on predicted plan signals. Use with "
                            "--trajectory-target or --step-plan."
                        ))
    parser.add_argument("--step-plan", action="store_true",
                        help=(
                            "derive per-decision sequence plan labels from game order, board state, "
                            "selected card/action/context, and deck-plan card tags"
                        ))
    parser.add_argument("--step-plan-loss-weight", type=float, default=0.0,
                        help="BCE loss multiplier for --step-plan labels")
    parser.add_argument("--step-plan-teacher-forcing", type=float, default=0.0,
                        help=(
                            "probability of feeding available trajectory/step-plan labels to the "
                            "hierarchical scorer during training; inference still uses predicted plans"
                        ))
    parser.add_argument("--history-k", type=int, default=0,
                        help="condition the policy on this many previous own decisions from the same game; 0 disables")
    parser.add_argument("--opp-history-k", type=int, default=0,
                        help=(
                            "condition on previous opponent labeled decisions saved by v12 extraction. "
                            "Use mainly for offline diagnostics; public Kaggle inference cannot exactly reproduce it."
                        ))
    parser.add_argument("--log-history-k", type=int, default=0,
                        help="condition on this many recent public observation log events saved by v12 extraction")
    parser.add_argument("--board-history-k", type=int, default=0,
                        help="condition on this many previous board snapshots saved by v12 extraction")
    parser.add_argument("--board-history-feat-dim", type=int, default=BOARD_HISTORY_FEAT_DIM,
                        help="feature width for board history snapshots")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-memory-gb", type=float, default=0.0,
                        help="cap this process' CUDA allocator to approximately N GiB; 0 disables")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.0,
                        help="cap this process' CUDA allocator to a fraction of the visible GPU; 0 disables")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init", default="",
                        help="optional .npz checkpoint used to initialize the model before training")
    parser.add_argument("--init-partial", action="store_true",
                        help="load only matching tensors from --init; default requires an exact architecture match")
    parser.add_argument("--init-skip-prefix", action="append", default=[],
                        help="with partial init, skip checkpoint tensors whose name starts with this prefix; repeatable")
    parser.add_argument("--reset-scorer", action="store_true",
                        help="when using --init, keep encoders but reinitialize action scorer tensors: score_fc* and stop_vec")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--load-progress-every", type=int, default=200000,
                        help="print corpus indexing progress every N raw decisions; 0 disables")
    parser.add_argument("--winner-only", action="store_true",
                        help="train only on decisions from games this player won; requires outcome metadata")
    parser.add_argument("--win-weight", type=float, default=1.0,
                        help="sample weight multiplier for winning-game decisions when outcome metadata exists")
    parser.add_argument("--loss-weight", type=float, default=1.0,
                        help="sample weight multiplier for losing-game decisions when outcome metadata exists")
    parser.add_argument("--draw-weight", type=float, default=1.0,
                        help="sample weight multiplier for drawn-game decisions when outcome metadata exists")
    parser.add_argument("--legacy-state-pool", action="store_true",
                        help="use old pooled board encoder instead of slot-aware active/bench encoder")
    parser.add_argument("--state-feat-dim", type=int, default=0,
                        help="override state feature width; 0 uses current encoder default")
    parser.add_argument("--opt-feat-dim", type=int, default=0,
                        help="override per-option feature width; 0 uses current encoder default")
    parser.add_argument("--first-action-weight", type=float, default=1.5)
    parser.add_argument("--raw-policy-loss-weight", type=float, default=0.0,
                        help=(
                            "add unweighted action-sequence NLL to the training objective. "
                            "Useful when sample reweighting/plan losses hide ordinary imitation quality."
                        ))
    parser.add_argument("--best-metric",
                        choices=[
                            "loss",
                            "policy",
                            "policy_raw",
                            "first_action",
                            "first_action_raw",
                            "set",
                            "trajectory",
                            "step_plan",
                        ],
                        default="loss",
                        help=(
                            "validation metric used to save the best checkpoint. "
                            "Use policy_raw for action-first checkpoint selection."
                        ))
    parser.add_argument("--value-weight", type=float, default=0.0,
                        help="optional auxiliary outcome-value MSE weight; requires outcome metadata")
    parser.add_argument("--set-loss-weight", type=float, default=0.0,
                        help="optional order-free multi-select auxiliary loss weight; 0 disables")
    parser.add_argument("--set-loss-min-count", type=int, default=2,
                        help="minimum labeled action length for set auxiliary loss")
    parser.add_argument("--set-loss-negative-weight", type=float, default=0.25,
                        help="relative weight for unselected options inside the set auxiliary loss")
    parser.add_argument("--option-weight", type=float, default=0.15)
    parser.add_argument("--context-weight", action="append", default=[],
                        help="repeatable context multiplier, e.g. MAIN=2.0 or 21=2.5")
    parser.add_argument("--type-weight", action="append", default=[],
                        help="repeatable true first option type multiplier, e.g. ATTACK=2.5")
    parser.add_argument("--card-weight", action="append", default=[],
                        help="repeatable true first option card multiplier, e.g. 647=2.5")
    parser.add_argument("--multi-select-weight", type=float, default=1.0,
                        help="sample multiplier when the labeled action selects more than one option")
    parser.add_argument("--trajectory-csv", action="append", default=[],
                        help="repeatable trajectory-level CSV from tools/mine_strategy_trajectories.py, usually games.csv")
    parser.add_argument("--trajectory-weight", action="append", default=[],
                        help=(
                            "repeatable whole-game multiplier CONDITION=WEIGHT. CONDITION can be a truthy "
                            "numeric column, e.g. attack_by_4=1.4, or a comparison, e.g. attack_count>=5=1.2"
                        ))
    parser.add_argument("--trajectory-target", action="append", default=[],
                        help=(
                            "repeatable binary/numeric trajectory column or condition predicted by an auxiliary "
                            "plan head, e.g. attack_by_4, primary_board_by_4, or outcome==win. "
                            "Requires --trajectory-csv."
                        ))
    parser.add_argument("--trajectory-target-loss-weight", type=float, default=0.0,
                        help="BCE loss multiplier for --trajectory-target auxiliary heads; 0 disables")
    parser.add_argument("--trajectory-keep", action="append", default=[],
                        help=(
                            "repeatable whole-game filter condition. Games must satisfy all keep "
                            "conditions from --trajectory-csv to be used, e.g. outcome_win or "
                            "strategy_success>=1. Conditions support simple A&B, A|B, and !A."
                        ))
    parser.add_argument("--trajectory-drop", action="append", default=[],
                        help=(
                            "repeatable whole-game drop condition. Games satisfying any drop "
                            "condition from --trajectory-csv are removed, e.g. outcome_loss or "
                            "outcome_loss&setup_success==0."
                        ))
    parser.add_argument("--trajectory-filter-missing-policy", choices=["keep", "drop"], default="drop",
                        help=(
                            "for --trajectory-keep/drop, whether corpus games not present in "
                            "--trajectory-csv are kept or dropped"
                        ))
    parser.add_argument("--trajectory-base-weight", type=float, default=1.0,
                        help="base multiplier for each trajectory CSV game before --trajectory-weight rules")
    parser.add_argument("--trajectory-weight-cap", type=float, default=8.0,
                        help="cap trajectory multiplier; 0 disables capping")
    parser.add_argument("--trajectory-missing-weight", type=float, default=1.0,
                        help="multiplier for corpus games not present in --trajectory-csv when missing policy is default")
    parser.add_argument("--trajectory-missing-policy", choices=["default", "drop"], default="default",
                        help="default keeps non-CSV games with --trajectory-missing-weight; drop trains only CSV games")
    parser.add_argument("--split-by-game", action="store_true",
                        help="split train/validation by episode_id:player_index groups instead of whole npz files")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--save", default="checkpoints/bc2_marnie_w2.npz")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    memory_limit_msg = _configure_cuda_memory_limit(
        device,
        gb=args.cuda_memory_gb,
        fraction=args.cuda_memory_fraction,
    )
    paths = discover_npz_paths(
        args.corpus,
        args.archetype,
        args.score_bands,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    base_path_count = len(paths)
    aux_path_count = 0
    aux_details: list[tuple[str, int]] = []
    aux_repeat = max(1, int(args.aux_repeat))
    for aux_root in args.aux_corpus:
        aux_paths = discover_npz_paths(
            aux_root,
            args.aux_archetype or args.archetype,
            args.aux_score_bands,
            date_from=args.date_from,
            date_to=args.date_to,
        )
        aux_path_count += len(aux_paths)
        aux_details.append((aux_root, len(aux_paths)))
        paths.extend(aux_paths * aux_repeat)
    context_weights = _parse_weight_specs(args.context_weight, CONTEXT_IDS, "context")
    type_weights = _parse_weight_specs(args.type_weight, TYPE_IDS, "type")
    card_weights = _parse_weight_specs(args.card_weight, CARD_IDS, "card")
    opponent_deck_sig_weights = _parse_text_weight_specs(
        args.opponent_deck_sig_weight,
        "opponent deck sig",
    )
    opponent_archetype_weights = _parse_text_weight_specs(
        args.opponent_archetype_weight,
        "opponent archetype",
        lower=True,
    )
    trajectory_weights, trajectory_stats = _load_trajectory_weights(
        args.trajectory_csv,
        args.trajectory_weight,
        base_weight=args.trajectory_base_weight,
        cap=args.trajectory_weight_cap,
    )
    trajectory_filter_keys, trajectory_filter_stats = _load_trajectory_filter(
        args.trajectory_csv,
        args.trajectory_keep,
        args.trajectory_drop,
    )
    trajectory_targets, trajectory_target_stats = _load_trajectory_targets(
        args.trajectory_csv,
        args.trajectory_target,
    )
    trajectory_target_dim = len([x for x in args.trajectory_target if str(x).strip()])
    plan_target_dim = trajectory_target_dim
    if trajectory_target_dim and not args.trajectory_csv:
        raise ValueError("--trajectory-target requires at least one --trajectory-csv")
    if args.step_plan:
        plan_target_dim += len(PLAN_LABELS)
    if args.trajectory_target_loss_weight > 0 and not args.trajectory_target:
        raise ValueError("--trajectory-target-loss-weight > 0 requires at least one --trajectory-target")
    if args.step_plan_loss_weight > 0 and not args.step_plan:
        raise ValueError("--step-plan-loss-weight > 0 requires --step-plan")
    if args.hierarchical_plan and plan_target_dim <= 0:
        raise ValueError("--hierarchical-plan requires --trajectory-target or --step-plan")
    history_k = max(0, int(args.history_k))
    opp_history_k = max(0, int(args.opp_history_k))
    log_history_k = max(0, int(args.log_history_k))
    board_history_k = max(0, int(args.board_history_k))
    board_history_feat_dim = max(0, int(args.board_history_feat_dim))
    if args.init:
        init_hist, init_opp_hist, init_log_hist, init_board_hist, init_board_feat = _checkpoint_history_dims(args.init)
        if history_k <= 0:
            history_k = init_hist
        if opp_history_k <= 0:
            opp_history_k = init_opp_hist
        if log_history_k <= 0:
            log_history_k = init_log_hist
        if board_history_k <= 0:
            board_history_k = init_board_hist
        if board_history_k > 0 and (not args.board_history_feat_dim or args.board_history_feat_dim == BOARD_HISTORY_FEAT_DIM):
            board_history_feat_dim = init_board_feat
    inferred_from_init = False
    state_feat_dim = int(args.state_feat_dim) if args.state_feat_dim else None
    opt_feat_dim = int(args.opt_feat_dim) if args.opt_feat_dim else None
    if args.init and not args.init_partial and (state_feat_dim is None or opt_feat_dim is None):
        init_state_dim, init_opt_dim = _checkpoint_feature_dims(args.init)
        if state_feat_dim is None:
            state_feat_dim = init_state_dim
            inferred_from_init = True
        if opt_feat_dim is None:
            opt_feat_dim = init_opt_dim
            inferred_from_init = True
    corpus = BCCorpus(
        paths,
        include_empty=args.include_empty,
        option_weight=args.option_weight,
        **({"state_feat_dim": state_feat_dim} if state_feat_dim is not None else {}),
        **({"opt_feat_dim": opt_feat_dim} if opt_feat_dim is not None else {}),
        deck_sigs=args.deck_sig,
        team_names=args.team_name,
        opponent_deck_sigs=args.opponent_deck_sig,
        opponent_archetypes=args.opponent_archetype,
        opponent_team_names=args.opponent_team_name,
        opponent_deck_sig_weights=opponent_deck_sig_weights,
        opponent_archetype_weights=opponent_archetype_weights,
        winner_only=args.winner_only,
        win_weight=args.win_weight,
        loss_weight=args.loss_weight,
        draw_weight=args.draw_weight,
        context_weights=context_weights,
        type_weights=type_weights,
        card_weights=card_weights,
        multi_select_weight=args.multi_select_weight,
        trajectory_filter_keys=trajectory_filter_keys,
        trajectory_filter_missing=args.trajectory_filter_missing_policy,
        trajectory_weights=trajectory_weights,
        trajectory_default_weight=args.trajectory_missing_weight,
        trajectory_missing=args.trajectory_missing_policy,
        trajectory_targets=trajectory_targets,
        trajectory_target_dim=trajectory_target_dim,
        archetype=args.archetype,
        step_plan=args.step_plan,
        history_k=history_k,
        opp_history_k=opp_history_k,
        log_history_k=log_history_k,
        board_history_k=board_history_k,
        board_history_feat_dim=board_history_feat_dim,
        split_by_game=(
            args.split_by_game
            or bool(trajectory_weights)
            or bool(trajectory_filter_keys)
            or bool(trajectory_targets)
            or args.step_plan
            or history_k > 0
            or opp_history_k > 0
            or log_history_k > 0
            or board_history_k > 0
        ),
        load_progress_every=args.load_progress_every,
    )
    if corpus.stats["kept"] <= 0:
        raise RuntimeError(
            "No training samples kept after filters. Check --score-bands, --deck-sig, "
            "--opponent-*, --winner-only, and corpus path before training."
        )
    train_idx, val_idx = corpus.split_indices(args.val_fraction, args.seed)
    if len(train_idx) < 2 or len(val_idx) < 1:
        raise RuntimeError(
            f"Not enough samples after split: train={len(train_idx)} val={len(val_idx)}. "
            "Relax filters or lower --val-fraction."
        )

    model_kwargs = {}
    if state_feat_dim is not None:
        model_kwargs["state_feat_dim"] = state_feat_dim
    if opt_feat_dim is not None:
        model_kwargs["opt_feat_dim"] = opt_feat_dim
    if plan_target_dim:
        model_kwargs["plan_dim"] = plan_target_dim
    if args.hierarchical_plan:
        model_kwargs["hierarchical_plan"] = True
    if history_k > 0:
        model_kwargs["history_k"] = history_k
    if opp_history_k > 0:
        model_kwargs["opp_history_k"] = opp_history_k
    if log_history_k > 0:
        model_kwargs["log_history_k"] = log_history_k
    if board_history_k > 0:
        model_kwargs["board_history_k"] = board_history_k
        model_kwargs["board_history_feat_dim"] = board_history_feat_dim
    model = build_policy_model(
        args.arch,
        width=args.width,
        slot_state=not args.legacy_state_pool,
        state_layers=args.state_layers,
        **model_kwargs,
    ).to(device)
    if args.init:
        skip_prefixes = list(args.init_skip_prefix)
        if args.reset_scorer:
            skip_prefixes.extend(["score_fc", "stop_vec"])
        partial_init = (
            args.init_partial
            or bool(plan_target_dim)
            or bool(skip_prefixes)
            or history_k > 0
            or opp_history_k > 0
            or log_history_k > 0
            or board_history_k > 0
        )
        loaded, skipped = _load_npz_init(
            model,
            args.init,
            device,
            partial=partial_init,
            skip_prefixes=tuple(skip_prefixes),
        )
        msg = f"Init: loaded={loaded} path={args.init}"
        if partial_init and not args.init_partial:
            msg += " partial_init_for_new_heads=True"
        if skip_prefixes:
            msg += f" skip_prefixes={skip_prefixes}"
        if skipped:
            msg += f" skipped={len(skipped)}"
        print(msg, flush=True)
        if skipped:
            print(f"Init skipped examples: {skipped[:8]}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_batches = (len(train_idx) + args.batch_size - 1) // args.batch_size
    total_steps = max(args.epochs * train_batches, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.03)
    params = sum(p.numel() for p in model.parameters())

    print(
        f"BC2: {args.archetype} {args.score_bands} device={device} "
        f"date_from={args.date_from or 'all'} date_to={args.date_to or 'all'} "
        f"{memory_limit_msg + ' ' if memory_limit_msg else ''}"
        f"arch={args.arch} width={args.width} state_layers={args.state_layers} "
        f"hierarchical_plan={args.hierarchical_plan} "
        f"history_k={history_k} opp_history_k={opp_history_k} "
        f"log_history_k={log_history_k} board_history_k={board_history_k} "
        f"board_history_feat_dim={board_history_feat_dim} "
        f"slot_state={not args.legacy_state_pool} "
        f"state_feat_dim={state_feat_dim or 'default'} opt_feat_dim={opt_feat_dim or 'default'} "
        f"{'feature_dims_from_init ' if inferred_from_init else ''}"
        f"deck_sigs={args.deck_sig or 'all'} team_names={args.team_name or 'all'} "
        f"opponent_deck_sigs={args.opponent_deck_sig or 'all'} "
        f"opponent_archetypes={args.opponent_archetype or 'all'} "
        f"opponent_team_names={args.opponent_team_name or 'all'} "
        f"opponent_deck_sig_weights={opponent_deck_sig_weights or '{}'} "
        f"opponent_archetype_weights={opponent_archetype_weights or '{}'} "
        f"winner_only={args.winner_only} "
        f"win/loss/draw_weight={args.win_weight}/{args.loss_weight}/{args.draw_weight} "
        f"first_action_weight={args.first_action_weight} raw_policy_loss_weight={args.raw_policy_loss_weight} "
        f"best_metric={args.best_metric} "
        f"context_weights={context_weights or '{}'} type_weights={type_weights or '{}'} "
        f"card_weights={card_weights or '{}'} "
        f"multi_select_weight={args.multi_select_weight} "
        f"trajectory_csv={args.trajectory_csv or 'none'} "
        f"trajectory_weight={args.trajectory_weight or 'none'} "
        f"trajectory_keep={args.trajectory_keep or 'none'} "
        f"trajectory_drop={args.trajectory_drop or 'none'} "
        f"trajectory_filter_missing={args.trajectory_filter_missing_policy} "
        f"trajectory_target={args.trajectory_target or 'none'} "
        f"trajectory_target_loss_weight={args.trajectory_target_loss_weight} "
        f"trajectory_target_dim={trajectory_target_dim} plan_target_dim={plan_target_dim} "
        f"step_plan={args.step_plan} step_plan_labels={PLAN_LABELS if args.step_plan else 'none'} "
        f"step_plan_loss/teacher_forcing={args.step_plan_loss_weight}/{args.step_plan_teacher_forcing} "
        f"trajectory_base/cap/missing={args.trajectory_base_weight}/{args.trajectory_weight_cap}/"
        f"{args.trajectory_missing_weight}:{args.trajectory_missing_policy} "
        f"reset_scorer={args.reset_scorer} init_skip_prefix={args.init_skip_prefix or 'none'} "
        f"split_by_game={corpus.split_by_game} "
        f"set_loss={args.set_loss_weight}/{args.set_loss_min_count}/{args.set_loss_negative_weight} "
        f"aux_corpus={aux_details or 'none'} aux_bands={args.aux_score_bands} aux_repeat={aux_repeat} "
        f"params={params/1e6:.1f}M",
        flush=True,
    )
    if trajectory_stats:
        print(f"Trajectory weights: {trajectory_stats}", flush=True)
    if trajectory_filter_stats:
        print(f"Trajectory filter: {trajectory_filter_stats}", flush=True)
    if trajectory_target_stats:
        print(f"Trajectory targets: {trajectory_target_stats}", flush=True)
    if args.step_plan:
        print(f"Step-plan counts: {dict(corpus.step_plan_counts)}", flush=True)
    print(
        f"Corpus: files={len(paths)} base_files={base_path_count} "
        f"aux_files={aux_path_count} stats={corpus.stats}",
        flush=True,
    )
    print(f"Split: train={len(train_idx)} val={len(val_idx)} batch={args.batch_size}", flush=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(train_idx)
        start = time.time()
        model.train()
        train_sums = _empty_metric_sums()
        steps = 0
        for batch_start in range(0, len(train_idx), args.batch_size):
            batch_idx = train_idx[batch_start : batch_start + args.batch_size]
            if len(batch_idx) < 2:
                continue
            parts = sequence_loss_parts(
                model,
                corpus.collate(batch_idx, device),
                first_action_weight=args.first_action_weight,
                raw_policy_loss_weight=args.raw_policy_loss_weight,
                value_weight=args.value_weight,
                plan_weight=args.trajectory_target_loss_weight,
                step_plan_weight=args.step_plan_loss_weight,
                plan_teacher_forcing=args.step_plan_teacher_forcing,
                set_loss_weight=args.set_loss_weight,
                set_loss_min_count=args.set_loss_min_count,
                set_loss_negative_weight=args.set_loss_negative_weight,
            )
            loss = parts["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            _update_metric_sums(train_sums, parts)
            steps += 1
            if steps == 1 or steps % 25 == 0 or steps == train_batches:
                train_so_far = _finalize_metric_sums(train_sums)
                print(
                    f"  epoch {epoch:02d} {steps:4d}/{train_batches} "
                    f"loss={train_so_far['loss']:.4f} "
                    f"policy_raw={train_so_far['policy_raw']:.4f} "
                    f"first_acc={train_so_far['first_action_acc']:.3f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
        train_metrics = _finalize_metric_sums(train_sums)
        with torch.no_grad():
            val_metrics, val_steps = _run_epoch(
                model,
                corpus,
                val_idx,
                args.batch_size,
                device,
                optimizer=None,
                first_action_weight=args.first_action_weight,
                raw_policy_loss_weight=args.raw_policy_loss_weight,
                value_weight=args.value_weight,
                plan_weight=args.trajectory_target_loss_weight,
                step_plan_weight=args.step_plan_loss_weight,
                plan_teacher_forcing=0.0,
                set_loss_weight=args.set_loss_weight,
                set_loss_min_count=args.set_loss_min_count,
                set_loss_negative_weight=args.set_loss_negative_weight,
            )
        elapsed = time.time() - start
        val_loss = _metric_value(val_metrics, args.best_metric)
        print(
            f"  done epoch {epoch}/{args.epochs} "
            f"{_format_metrics('train', train_metrics)} "
            f"{_format_metrics('val', val_metrics)} "
            f"best_metric={args.best_metric}:{val_loss:.4f} {elapsed:.0f}s",
            flush=True,
        )
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            _save_npz(model, args.save)
            print(f"  saved best {args.best_metric}={best_val:.4f} -> {args.save}", flush=True)
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            ckpt = args.save.replace(".npz", f"_ep{epoch:03d}.npz")
            _save_npz(model, ckpt)
            print(f"  checkpoint {ckpt}", flush=True)

    print(f"Best {args.best_metric}={best_val:.4f} -> {args.save}")


if __name__ == "__main__":
    main()
