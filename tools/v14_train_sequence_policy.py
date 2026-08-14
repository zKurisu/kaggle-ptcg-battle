#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import math
import json
import os
import random
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
    return {
        f"{name}_top1": float(ab_acc.get("cur_top1", 0.0)),
        f"{name}_delta": float(ab_acc.get("cur_top1", 0.0) - full_acc.get("cur_top1", 0.0)),
        f"{name}_agree": float((full_pred == ab_pred).float().mean().item()),
        f"{name}_kl": float(kl),
    }


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
    no_action = _no_action_history_batch(batch)
    out.update(_current_logit_compare(
        name="noact",
        batch=batch,
        full_outputs=full_outputs,
        ab_outputs=model(no_action),
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
    if val_acc.get("cur1_agree", 0.0) > 0.90 and abs(val_acc.get("cur1_delta", 0.0)) < 0.02:
        warnings.append("history_not_affecting_current")
    if val_acc.get("noact_agree", 0.0) > 0.92 and abs(val_acc.get("noact_delta", 0.0)) < 0.02:
        warnings.append("action_ledger_not_affecting_current")
    if val_acc.get("revhist_agree", 0.0) > 0.92 and abs(val_acc.get("revhist_delta", 0.0)) < 0.02:
        warnings.append("history_order_not_affecting_current")
    if val_acc.get("plan_pos_rate", 0.0) > 0.10 and val_acc.get("plan_f1", 0.0) < 0.20:
        warnings.append("future_plan_weak")
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
                    if not np.isfinite(grad_norm):
                        print(
                            f"FATAL nonfinite_grad mode=train epoch={epoch} batch={bi}/{total_batches} "
                            f"grad_norm={grad_norm} parts={parts}",
                            flush=True,
                        )
                        raise RuntimeError("nonfinite sequence policy gradient")
                scaler.step(optimizer)
                scaler.update()
                if progress_every and (bi == 1 or bi % progress_every == 0 or bi == total_batches):
                    parts = dict(parts)
                    parts["grad_norm"] = grad_norm
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
                f"seq={ma.get('seq_len_mean', 0):.1f}/{ma.get('seq_full_rate', 0):.2f}/{ma.get('history_present_rate', 0):.2f} "
                f"setF1={ma.get('set_f1', 0):.3f} ordAcc={ma.get('order_acc', 0):.3f} "
                f"atype={ma.get('action_type_acc', 0):.3f} "
                f"pln={ma.get('plan_f1', 0):.3f}/{ma.get('plan_mae', 0):.3f}/"
                f"{ma.get('plan_pos_rate', 0):.2f}->{ma.get('plan_pred_pos_rate', 0):.2f} "
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
                f"boost={mp.get('weight_boost', 0):.2f} dcaW={mp.get('dca_weight_share', 0):.2f} "
                f"m2W={mp.get('multi_weight_share', 0):.2f} "
                f"curA={mp.get('current_action', 0):.3f}/{mp.get('prefix_action', 0):.3f} "
                f"curHead={mp.get('action_current_head', 0):.3f}/{mp.get('action_prefix_head', 0):.3f} "
                f"dcaA={mp.get('dca_action', 0):.3f}/{mp.get('non_dca_action', 0):.3f} "
                f"out={ma.get('outcome_acc', 0):.3f}/{ma.get('outcome_brier', 0):.3f}/"
                f"{ma.get('outcome_pos_rate', 0):.2f}->{ma.get('outcome_pred_pos_rate', 0):.2f} "
                f"types=p{ma.get('play_top1', 0):.2f},at{ma.get('attach_top1', 0):.2f},"
                f"ev{ma.get('evolve_top1', 0):.2f},ab{ma.get('ability_top1', 0):.2f},"
                f"ak{ma.get('attack_top1', 0):.2f},en{ma.get('end_top1', 0):.2f} "
                f"grad={mp.get('grad_norm', 0):.2f} "
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
    p.add_argument("--multi-target-weight", type=float, default=1.0,
                   help="boost decision losses on target_k>1 rows; default keeps historical weighting")
    p.add_argument("--damage-counter-weight", type=float, default=1.0,
                   help="boost DamageCounterAny resolution rows such as Dragapult Phantom Dive")
    p.add_argument("--out", required=True)
    args = p.parse_args()

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
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M device={device}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_cfg = SequenceLossConfig(
        action_weight=args.action_weight,
        current_action_weight=args.current_action_weight,
        prefix_action_weight=args.prefix_action_weight,
        order_weight=args.order_weight,
        multi_weight=args.multi_weight,
        count_weight=args.count_weight,
        plan_weight=args.plan_weight,
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
            f"val_seq={val_acc.get('seq_len_mean', 0):.1f}/{val_acc.get('seq_full_rate', 0):.2f}/"
            f"{val_acc.get('history_present_rate', 0):.2f} "
            f"val_cur1={val_acc.get('cur1_top1', 0):.3f}/{val_acc.get('cur1_delta', 0):+.3f}/"
            f"{val_acc.get('cur1_agree', 0):.3f}/{val_acc.get('cur1_kl', 0):.3f} "
            f"val_noact={val_acc.get('noact_top1', 0):.3f}/{val_acc.get('noact_delta', 0):+.3f}/"
            f"{val_acc.get('noact_agree', 0):.3f}/{val_acc.get('noact_kl', 0):.3f} "
            f"val_revhist={val_acc.get('revhist_top1', 0):.3f}/{val_acc.get('revhist_delta', 0):+.3f}/"
            f"{val_acc.get('revhist_agree', 0):.3f}/{val_acc.get('revhist_kl', 0):.3f} "
            f"val_plan={val_loss.get('plan', 0):.4f} val_type={val_acc.get('type_acc', 0):.3f} "
            f"val_planSig={val_acc.get('plan_f1', 0):.3f}/{val_acc.get('plan_mae', 0):.3f}/"
            f"{val_acc.get('plan_pos_rate', 0):.2f}->{val_acc.get('plan_pred_pos_rate', 0):.2f} "
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
            f"val_boost={val_loss.get('weight_boost', 0):.2f} "
            f"val_dcaW={val_loss.get('dca_weight_share', 0):.2f} val_m2W={val_loss.get('multi_weight_share', 0):.2f} "
            f"val_curA={val_loss.get('current_action', 0):.3f}/{val_loss.get('prefix_action', 0):.3f} "
            f"val_curHead={val_loss.get('action_current_head', 0):.3f}/{val_loss.get('action_prefix_head', 0):.3f} "
            f"val_dcaA={val_loss.get('dca_action', 0):.3f}/{val_loss.get('non_dca_action', 0):.3f} "
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
            f"noact_delta={val_acc.get('noact_delta', 0):+.3f} "
            f"noact_agree={val_acc.get('noact_agree', 0):.3f} "
            f"revhist_delta={val_acc.get('revhist_delta', 0):+.3f} "
            f"revhist_agree={val_acc.get('revhist_agree', 0):.3f} "
            f"plan_f1={val_acc.get('plan_f1', 0):.3f} "
            f"ambig_top1={val_acc.get('ambig_type_top1', 0):.3f} "
            f"train_grad={train_loss.get('grad_norm', 0):.2f} "
            f"dca_weight_share={val_loss.get('dca_weight_share', 0):.3f} "
            f"multi_weight_share={val_loss.get('multi_weight_share', 0):.3f} "
            f"current_weight_share={val_loss.get('current_weight_share', 0):.3f} "
            f"current_row_share={val_loss.get('current_row_share', 0):.3f} "
            f"warnings={','.join(warnings) if warnings else 'none'}",
            flush=True,
        )
        save_checkpoint(last_path, model=model, args=args, corpus=corpus, epoch=epoch, val_loss=val)
        if val < best:
            best = val
            save_checkpoint(best_path, model=model, args=args, corpus=corpus, epoch=epoch, val_loss=val)
            print(f"  saved best {best:.4f} -> {best_path}", flush=True)
    print(f"Training complete best={best:.4f} checkpoint={best_path}", flush=True)


if __name__ == "__main__":
    main()
