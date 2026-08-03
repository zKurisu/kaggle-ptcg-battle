#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.bc2 import BCCorpus, discover_npz_paths, sequence_nll
from ptcg_rl.model import PolicyValueNet


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


def _save_npz(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})


def _run_epoch(model, corpus, indices, batch_size, device, optimizer=None,
               first_action_weight=1.0, value_weight=0.0):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    steps = 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        if len(batch_idx) < 2:
            continue
        batch = corpus.collate(batch_idx, device)
        loss = sequence_nll(
            model,
            batch,
            first_action_weight=first_action_weight,
            value_weight=value_weight,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        total += float(loss.detach().cpu())
        steps += 1
    return total / max(steps, 1), steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/bc_corpus_banded_v4")
    parser.add_argument("--archetype", default="Marnie Grimmsnarl")
    parser.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])
    parser.add_argument("--deck-sig", action="append", default=[],
                        help="filter to one or more deck signatures; repeatable. Requires freshly extracted corpus metadata.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
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
    parser.add_argument("--first-action-weight", type=float, default=1.5)
    parser.add_argument("--value-weight", type=float, default=0.0,
                        help="optional auxiliary outcome-value MSE weight; requires outcome metadata")
    parser.add_argument("--option-weight", type=float, default=0.15)
    parser.add_argument("--context-weight", action="append", default=[],
                        help="repeatable context multiplier, e.g. MAIN=2.0 or 21=2.5")
    parser.add_argument("--type-weight", action="append", default=[],
                        help="repeatable true first option type multiplier, e.g. ATTACK=2.5")
    parser.add_argument("--multi-select-weight", type=float, default=1.0,
                        help="sample multiplier when the labeled action selects more than one option")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--save", default="checkpoints/bc2_marnie_w2.npz")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    paths = discover_npz_paths(args.corpus, args.archetype, args.score_bands)
    context_weights = _parse_weight_specs(args.context_weight, CONTEXT_IDS, "context")
    type_weights = _parse_weight_specs(args.type_weight, TYPE_IDS, "type")
    corpus = BCCorpus(
        paths,
        include_empty=args.include_empty,
        option_weight=args.option_weight,
        deck_sigs=args.deck_sig,
        winner_only=args.winner_only,
        win_weight=args.win_weight,
        loss_weight=args.loss_weight,
        draw_weight=args.draw_weight,
        context_weights=context_weights,
        type_weights=type_weights,
        multi_select_weight=args.multi_select_weight,
        load_progress_every=args.load_progress_every,
    )
    if corpus.stats["kept"] <= 0:
        raise RuntimeError(
            "No training samples kept after filters. Check --score-bands, --deck-sig, "
            "--winner-only, and corpus path before training."
        )
    train_idx, val_idx = corpus.split_indices(args.val_fraction, args.seed)
    if len(train_idx) < 2 or len(val_idx) < 1:
        raise RuntimeError(
            f"Not enough samples after split: train={len(train_idx)} val={len(val_idx)}. "
            "Relax filters or lower --val-fraction."
        )

    model = PolicyValueNet(width=args.width, slot_state=not args.legacy_state_pool).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_batches = (len(train_idx) + args.batch_size - 1) // args.batch_size
    total_steps = max(args.epochs * train_batches, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.03)
    params = sum(p.numel() for p in model.parameters())

    print(
        f"BC2: {args.archetype} {args.score_bands} device={device} "
        f"width={args.width} slot_state={not args.legacy_state_pool} "
        f"deck_sigs={args.deck_sig or 'all'} winner_only={args.winner_only} "
        f"win/loss/draw_weight={args.win_weight}/{args.loss_weight}/{args.draw_weight} "
        f"context_weights={context_weights or '{}'} type_weights={type_weights or '{}'} "
        f"multi_select_weight={args.multi_select_weight} "
        f"params={params/1e6:.1f}M",
        flush=True,
    )
    print(f"Corpus: files={len(paths)} stats={corpus.stats}", flush=True)
    print(f"Split: train={len(train_idx)} val={len(val_idx)} batch={args.batch_size}", flush=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(train_idx)
        start = time.time()
        model.train()
        total = 0.0
        steps = 0
        for batch_start in range(0, len(train_idx), args.batch_size):
            batch_idx = train_idx[batch_start : batch_start + args.batch_size]
            if len(batch_idx) < 2:
                continue
            loss = sequence_nll(
                model,
                corpus.collate(batch_idx, device),
                first_action_weight=args.first_action_weight,
                value_weight=args.value_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach().cpu())
            steps += 1
            if steps == 1 or steps % 25 == 0 or steps == train_batches:
                print(
                    f"  epoch {epoch:02d} {steps:4d}/{train_batches} "
                    f"loss={total/max(steps,1):.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
        train_loss = total / max(steps, 1)
        with torch.no_grad():
            val_loss, val_steps = _run_epoch(
                model,
                corpus,
                val_idx,
                args.batch_size,
                device,
                optimizer=None,
                first_action_weight=args.first_action_weight,
                value_weight=args.value_weight,
            )
        elapsed = time.time() - start
        print(f"  done epoch {epoch}/{args.epochs} train={train_loss:.4f} val={val_loss:.4f} {elapsed:.0f}s", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            _save_npz(model, args.save)
            print(f"  saved best {best_val:.4f} -> {args.save}", flush=True)
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            ckpt = args.save.replace(".npz", f"_ep{epoch:03d}.npz")
            _save_npz(model, ckpt)
            print(f"  checkpoint {ckpt}", flush=True)

    print(f"Best val={best_val:.4f} -> {args.save}")


if __name__ == "__main__":
    main()
