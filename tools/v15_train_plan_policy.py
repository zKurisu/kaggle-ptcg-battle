#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import glob
import math
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

from ptcg_rl.v15.constants import DEFAULT_HISTORY_K, DEFAULT_MAX_OPTIONS, DEFAULT_PLAN_STEPS, FEATURE_VERSION
from ptcg_rl.v15.data import V15RowCorpus, discover_v15_npz, signal_stats_line, signal_warnings, split_indices_by_game
from ptcg_rl.v15.model import V15LossConfig, V15PlanPolicyNet, compare_logits, v15_accuracy, v15_policy_loss


def _split_csv(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


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


def _mean_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({k for d in items for k in d})
    return {k: float(np.mean([d.get(k, 0.0) for d in items])) for k in keys}


def _fmt(stats: dict[str, float], key: str, digits: int = 3) -> str:
    return f"{float(stats.get(key, 0.0)):.{digits}f}"


def _mem(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return f" mem={torch.cuda.memory_allocated(idx)/1024**3:.1f}/{torch.cuda.memory_reserved(idx)/1024**3:.1f}G"


def _signal_health(val: dict[str, float], stats: dict[str, float | int]) -> list[str]:
    warnings = signal_warnings(stats)
    if val.get("reverse_history_agree", 1.0) > 0.94 and val.get("reverse_history_kl", 0.0) < 0.01:
        warnings.append("history_order_not_affecting_action")
    if val.get("no_history_agree", 1.0) > 0.90 and val.get("no_history_kl", 0.0) < 0.02:
        warnings.append("history_not_used_by_action")
    if val.get("no_plan_agree", 1.0) > 0.92 and val.get("no_plan_kl", 0.0) < 0.02:
        warnings.append("plan_latent_not_used_by_action")
    if float(stats.get("known_rate", 0.0)) > 0 and val.get("no_known_agree", 1.0) > 0.96:
        warnings.append("known_info_not_used_by_action")
    if val.get("plan_type_acc", 0.0) < 0.25 and float(stats.get("plan_slots_mean", 0.0)) > 0.5:
        warnings.append("plan_type_weak")
    if val.get("history_type_acc", 0.0) < 0.18 and float(stats.get("event_slots_mean", 0.0)) > 2.0:
        warnings.append("history_type_weak")
    if float(stats.get("known_rate", 0.0)) > 0 and val.get("known_type_acc", 0.0) < 0.15:
        warnings.append("known_type_weak")
    if val.get("attach_n", 0.0) > 20 and val.get("attach_top1", 0.0) < 0.45:
        warnings.append("attach_decision_weak")
    if val.get("attach_n", 0.0) > 20 and val.get("attach_within_top1", 0.0) < 0.55:
        warnings.append("attach_within_type_weak")
    if val.get("evolve_n", 0.0) > 20 and val.get("evolve_top1", 0.0) < 0.45:
        warnings.append("evolve_decision_weak")
    if val.get("evolve_n", 0.0) > 20 and val.get("evolve_within_top1", 0.0) < 0.55:
        warnings.append("evolve_within_type_weak")
    if val.get("attack_n", 0.0) > 20 and val.get("attack_top1", 0.0) < 0.55:
        warnings.append("attack_decision_weak")
    if val.get("attack_n", 0.0) > 20 and val.get("attack_within_top1", 0.0) < 0.65:
        warnings.append("attack_within_type_weak")
    if val.get("route_rate", 0.0) > 0.02 and val.get("route_top1", 0.0) < 0.75:
        warnings.append("route_top1_weak")
    if val.get("route_rate", 0.0) > 0.02 and val.get("route_label_hit", 1.0) < 0.55:
        warnings.append("route_labels_conflict_with_mainline")
    if val.get("optional_count_acc", 1.0) < 0.65:
        warnings.append("optional_count_weak")
    return sorted(set(warnings))


@torch.no_grad()
def evaluate(
    model: V15PlanPolicyNet,
    corpus: V15RowCorpus,
    ids: list[int],
    cfg: V15LossConfig,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: int = 0,
    seed: int = 0,
    ablations: bool = True,
) -> dict[str, float]:
    model.eval()
    losses: list[dict[str, float]] = []
    accs: list[dict[str, float]] = []
    for batch_ids in _iter_batches(ids, batch_size, shuffle=False, seed=seed, max_batches=max_batches):
        batch = corpus.make_batch(batch_ids).to(device)
        outputs = model(batch)
        _, parts = v15_policy_loss(outputs, batch, cfg)
        acc = v15_accuracy(outputs, batch)
        if ablations:
            for name, mode in (
                ("no_history", "no_history"),
                ("reverse_history", "reverse_history"),
                ("no_known", "no_known"),
                ("no_plan", "no_plan"),
            ):
                ab = model(batch, ablate=mode)
                acc.update(compare_logits(name, outputs, ab, batch))
        losses.append(parts)
        accs.append(acc)
    out = _mean_dict(losses)
    out.update(_mean_dict(accs))
    return out


def _paths_from_args(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    for pattern in args.npz:
        paths.extend(sorted(glob.glob(pattern)))
    if args.corpus and args.archetype:
        paths.extend(
            discover_v15_npz(
                args.corpus,
                args.archetype,
                args.score_band,
                date_from=args.date_from,
                date_to=args.date_to,
            )
        )
    return sorted(set(paths))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="")
    p.add_argument("--npz", action="append", default=[])
    p.add_argument("--archetype", default="")
    p.add_argument("--score-band", action="append", default=["600-699", "700-799", "800-899", "900-999", "1000-1099", "1100-1199", "1200+"])
    p.add_argument("--date-from", default="")
    p.add_argument("--date-to", default="")
    p.add_argument("--deck-sig", action="append", default=[])
    p.add_argument("--team-name", action="append", default=[])
    p.add_argument("--opponent-archetype", action="append", default=[])
    p.add_argument("--opponent-deck-sig", action="append", default=[])
    p.add_argument("--winner-only", action="store_true")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--max-score", type=float, default=0.0)
    p.add_argument("--win-weight", type=float, default=1.0)
    p.add_argument("--loss-weight", type=float, default=1.0)
    p.add_argument("--draw-weight", type=float, default=1.0)
    p.add_argument("--history-k", type=int, default=DEFAULT_HISTORY_K)
    p.add_argument("--plan-steps", type=int, default=DEFAULT_PLAN_STEPS)
    p.add_argument("--max-options", type=int, default=DEFAULT_MAX_OPTIONS)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--type-prior-scale", type=float, default=1.50)
    p.add_argument("--history-type-prior-scale", type=float, default=0.35)
    p.add_argument("--known-type-prior-scale", type=float, default=0.25)
    p.add_argument("--route-prior-scale", type=float, default=1.50,
                   help="live inference prior added to explicit mainline route options")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--val-frac", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--out", required=True)
    for field in dataclasses.fields(V15LossConfig):
        p.add_argument(f"--{field.name.replace('_', '-')}", type=float, default=getattr(V15LossConfig(), field.name))
    args = p.parse_args()

    paths = _paths_from_args(args)
    if not paths:
        raise FileNotFoundError("No v15 npz files matched")
    print(f"v15 train feature={FEATURE_VERSION} files={len(paths)}", flush=True)
    corpus = V15RowCorpus(
        paths,
        max_options=args.max_options,
        history_k=args.history_k,
        plan_steps=args.plan_steps,
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
    print(signal_stats_line(corpus.stats), flush=True)
    sw = signal_warnings(corpus.stats)
    if sw:
        print("initial_signal_warnings=" + ",".join(sw), flush=True)
    train_ids, val_ids = split_indices_by_game(corpus.game_keys, val_frac=args.val_frac, seed=args.seed)
    print(f"split rows train={len(train_ids)} val={len(val_ids)} seed={args.seed}", flush=True)
    if len(val_ids) < max(32, args.batch_size // 2):
        print("WARNING: validation split is small; signal probes may be noisy", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = V15PlanPolicyNet(
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        history_k=args.history_k,
        plan_steps=args.plan_steps,
        type_prior_scale=args.type_prior_scale,
        history_type_prior_scale=args.history_type_prior_scale,
        known_type_prior_scale=args.known_type_prior_scale,
    ).to(device)
    cfg = V15LossConfig(**{field.name: float(getattr(args, field.name)) for field in dataclasses.fields(V15LossConfig)})
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        parts_log: list[dict[str, float]] = []
        acc_log: list[dict[str, float]] = []
        total_batches = _total_batches(len(train_ids), args.batch_size, args.max_train_batches)
        for bi, batch_ids in enumerate(_iter_batches(train_ids, args.batch_size, shuffle=True, seed=args.seed + epoch, max_batches=args.max_train_batches), 1):
            batch = corpus.make_batch(batch_ids).to(device)
            outputs = model(batch)
            loss, parts = v15_policy_loss(outputs, batch, cfg)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at epoch={epoch} batch={bi}: {parts}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad):
                raise RuntimeError(f"nonfinite gradient at epoch={epoch} batch={bi}")
            opt.step()
            parts_log.append(parts)
            acc_log.append(v15_accuracy(outputs, batch))
            if args.progress_every and (bi == 1 or bi % args.progress_every == 0 or bi == total_batches):
                mp = _mean_dict(parts_log[-args.progress_every:])
                ma = _mean_dict(acc_log[-args.progress_every:])
                elapsed = time.time() - t0
                rate = bi / max(elapsed, 1e-9)
                eta = (total_batches - bi) / max(rate, 1e-9)
                print(
                    f"epoch {epoch}/{args.epochs} batch {bi}/{total_batches} "
                    f"loss={_fmt(mp,'loss')} act={_fmt(mp,'action')} planT={_fmt(mp,'plan_type')} "
                    f"top1={_fmt(ma,'top1')} top3={_fmt(ma,'top3')} within={_fmt(ma,'within_type_top1')} "
                    f"route={_fmt(ma,'route_top1')} count={_fmt(ma,'count_acc')} optCount={_fmt(ma,'optional_count_acc')} "
                    f"type={_fmt(ma,'type_acc')} "
                    f"histType={_fmt(ma,'history_type_acc')} knownType={_fmt(ma,'known_type_acc')} "
                    f"contF1={_fmt(ma,'continue_f1')} planType={_fmt(ma,'plan_type_acc')} "
                    f"multiF1={_fmt(ma,'multi_f1')} grad={float(grad):.2f}{_mem(device)} eta={eta:.0f}s",
                    flush=True,
                )
        train = _mean_dict(parts_log)
        train.update(_mean_dict(acc_log))
        val = evaluate(
            model,
            corpus,
            val_ids,
            cfg,
            batch_size=args.batch_size,
            device=device,
            max_batches=args.max_val_batches,
            seed=args.seed + epoch,
            ablations=True,
        )
        warnings = _signal_health(val, corpus.stats)
        print(
            f"done epoch {epoch}/{args.epochs} "
            f"train_loss={_fmt(train,'loss')} train_top1={_fmt(train,'top1')} "
            f"val_loss={_fmt(val,'loss')} val_top1={_fmt(val,'top1')} val_top3={_fmt(val,'top3')} "
            f"val_within={_fmt(val,'within_type_top1')} within_rate={_fmt(val,'within_type_rate')} "
            f"val_route={_fmt(val,'route_top1')} route_label={_fmt(val,'route_label_hit')} route_rate={_fmt(val,'route_rate')} "
            f"val_count={_fmt(val,'count_acc')} val_optCount={_fmt(val,'optional_count_acc')} "
            f"val_type={_fmt(val,'type_acc')} val_histType={_fmt(val,'history_type_acc')} "
            f"val_knownType={_fmt(val,'known_type_acc')} val_contF1={_fmt(val,'continue_f1')} "
            f"val_planType={_fmt(val,'plan_type_acc')} val_planCard={_fmt(val,'plan_card_acc')} "
            f"val_planAtk={_fmt(val,'plan_attack_acc')} val_multiF1={_fmt(val,'multi_f1')}",
            flush=True,
        )
        print(
            "probe "
            f"noHist={_fmt(val,'no_history_delta')}/{_fmt(val,'no_history_agree')}/{_fmt(val,'no_history_kl',5)} "
            f"revHist={_fmt(val,'reverse_history_delta')}/{_fmt(val,'reverse_history_agree')}/{_fmt(val,'reverse_history_kl',5)} "
            f"noKnown={_fmt(val,'no_known_delta')}/{_fmt(val,'no_known_agree')}/{_fmt(val,'no_known_kl',5)} "
            f"noPlan={_fmt(val,'no_plan_delta')}/{_fmt(val,'no_plan_agree')}/{_fmt(val,'no_plan_kl',5)}",
            flush=True,
        )
        print(
            "per_type "
            f"play={_fmt(val,'play_top1')} attach={_fmt(val,'attach_top1')} evolve={_fmt(val,'evolve_top1')} "
            f"ability={_fmt(val,'ability_top1')} retreat={_fmt(val,'retreat_top1')} "
            f"attack={_fmt(val,'attack_top1')} end={_fmt(val,'end_top1')}",
            flush=True,
        )
        print(
            "within_type "
            f"play={_fmt(val,'play_within_top1')} attach={_fmt(val,'attach_within_top1')} "
            f"evolve={_fmt(val,'evolve_within_top1')} ability={_fmt(val,'ability_within_top1')} "
            f"retreat={_fmt(val,'retreat_within_top1')} attack={_fmt(val,'attack_within_top1')} "
            f"end={_fmt(val,'end_within_top1')}",
            flush=True,
        )
        if warnings:
            print("signal_health warnings=" + ",".join(warnings), flush=True)
        if val.get("loss", float("inf")) < best:
            best = float(val["loss"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "width": args.width,
                        "layers": args.layers,
                        "heads": args.heads,
                        "dropout": args.dropout,
                        "history_k": args.history_k,
                        "plan_steps": args.plan_steps,
                        "max_options": args.max_options,
                        "type_prior_scale": args.type_prior_scale,
                        "history_type_prior_scale": args.history_type_prior_scale,
                        "known_type_prior_scale": args.known_type_prior_scale,
                        "archetype": args.archetype,
                        "route_prior_scale": args.route_prior_scale,
                        "feature_version": FEATURE_VERSION,
                    },
                    "loss_config": dataclasses.asdict(cfg),
                    "stats": corpus.stats,
                    "val": val,
                },
                out_path,
            )
            print(f"saved best {best:.4f} -> {out_path}", flush=True)
    print(f"complete best_val={best:.4f} out={out_path}", flush=True)


if __name__ == "__main__":
    main()
