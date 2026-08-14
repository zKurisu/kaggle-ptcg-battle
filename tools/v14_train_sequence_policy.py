#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from ptcg_rl.seq.data import SequenceCorpus, discover_sequence_npz
from ptcg_rl.seq.model import SequenceLossConfig, SequencePolicyNet, sequence_accuracy, sequence_policy_loss


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
) -> tuple[dict[str, float], dict[str, float]]:
    train = optimizer is not None
    model.train(train)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and train)
    loss_parts: list[dict[str, float]] = []
    acc_parts: list[dict[str, float]] = []
    t0 = time.time()
    for bi, sample_ids in enumerate(
        _iter_batches(ids, batch_size, shuffle=train, seed=epoch * 100003 + 17, max_batches=max_batches),
        1,
    ):
        batch = corpus.collate(sample_ids).to(device)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(batch)
                loss, parts = sequence_policy_loss(outputs, batch, loss_cfg)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        loss_parts.append(parts)
        acc_parts.append(sequence_accuracy(outputs, batch))
        if progress_every and (bi == 1 or bi % progress_every == 0):
            elapsed = time.time() - t0
            rate = bi * batch_size / max(elapsed, 1e-9)
            mp = _mean_parts(loss_parts[-max(progress_every, 1):])
            ma = _mean_parts(acc_parts[-max(progress_every, 1):])
            mode = "train" if train else "val"
            print(
                f"  {mode} epoch={epoch} batch={bi} loss={mp.get('loss', 0):.4f} "
                f"act={mp.get('action', 0):.4f} plan={mp.get('plan', 0):.4f} "
                f"top1={ma.get('top1', 0):.3f} type={ma.get('type_acc', 0):.3f} "
                f"{rate:.0f} samples/s",
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
    p.add_argument("--action-weight", type=float, default=1.0)
    p.add_argument("--multi-weight", type=float, default=0.15)
    p.add_argument("--plan-weight", type=float, default=0.35)
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
        multi_weight=args.multi_weight,
        plan_weight=args.plan_weight,
        outcome_weight=0.10,
        type_weight=0.10,
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
            )
        val = val_loss.get("loss", float("inf"))
        print(
            f"done epoch {epoch}/{args.epochs} "
            f"train={train_loss.get('loss', 0):.4f} val={val:.4f} "
            f"train_top1={train_acc.get('top1', 0):.3f} val_top1={val_acc.get('top1', 0):.3f} "
            f"val_plan={val_loss.get('plan', 0):.4f} val_type={val_acc.get('type_acc', 0):.3f}",
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
