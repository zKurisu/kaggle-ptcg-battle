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


def _save_npz(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})


def _run_epoch(model, corpus, indices, batch_size, device, optimizer=None, first_action_weight=1.0):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    steps = 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        if len(batch_idx) < 2:
            continue
        batch = corpus.collate(batch_idx, device)
        loss = sequence_nll(model, batch, first_action_weight=first_action_weight)
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
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--first-action-weight", type=float, default=1.5)
    parser.add_argument("--option-weight", type=float, default=0.15)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--save", default="checkpoints/bc2_marnie_w2.npz")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    paths = discover_npz_paths(args.corpus, args.archetype, args.score_bands)
    corpus = BCCorpus(paths, include_empty=args.include_empty, option_weight=args.option_weight)
    train_idx, val_idx = corpus.split_indices(args.val_fraction, args.seed)

    model = PolicyValueNet(width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_batches = (len(train_idx) + args.batch_size - 1) // args.batch_size
    total_steps = max(args.epochs * train_batches, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.03)
    params = sum(p.numel() for p in model.parameters())

    print(
        f"BC2: {args.archetype} {args.score_bands} device={device} "
        f"width={args.width} params={params/1e6:.1f}M",
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
