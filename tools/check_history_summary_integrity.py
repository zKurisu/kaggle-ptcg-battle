#!/usr/bin/env python3
"""Smoke-test BC history summary batches for Kaggle-live compatibility."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.bc2.data import BCCorpus, discover_npz_paths  # noqa: E402
from ptcg_rl.history_features import HISTORY_SUMMARY_DIM  # noqa: E402


def _summary_stats(summary: np.ndarray) -> dict[str, float]:
    arr = np.asarray(summary, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return {
            "abs_mean": 0.0,
            "nonzero_rate": 0.0,
            "opp_abs_mean": 0.0,
            "own_abs_mean": 0.0,
            "log_abs_mean": 0.0,
            "board_abs_mean": 0.0,
        }
    return {
        "abs_mean": float(np.abs(arr).mean()),
        "nonzero_rate": float((np.abs(arr) > 1e-7).mean()),
        "opp_abs_mean": float(np.abs(arr[:, 14:24]).mean()) if arr.shape[1] >= 24 else 0.0,
        "own_abs_mean": float(np.abs(arr[:, 4:14]).mean()) if arr.shape[1] >= 14 else 0.0,
        "log_abs_mean": float(np.abs(arr[:, 24:30]).mean()) if arr.shape[1] >= 30 else 0.0,
        "board_abs_mean": float(np.abs(arr[:, 30:39]).mean()) if arr.shape[1] >= 39 else 0.0,
    }


def run_case(
    paths: list[str],
    *,
    archetype: str,
    deck_sig: list[str],
    batch_size: int,
    history_k: int,
    log_history_k: int,
    board_history_k: int,
    history_augment: bool,
) -> dict[str, float]:
    corpus = BCCorpus(
        paths,
        archetype=archetype,
        deck_sigs=deck_sig,
        win_weight=1.0,
        loss_weight=1.0,
        draw_weight=1.0,
        history_k=history_k,
        log_history_k=log_history_k,
        board_history_k=board_history_k,
        history_summary_dim=HISTORY_SUMMARY_DIM,
        split_by_game=True,
        load_progress_every=0,
    )
    indices = corpus.all_indices()[: max(1, int(batch_size))]
    if not indices:
        raise RuntimeError("no kept indices for requested filters")
    batch = corpus.collate(
        indices,
        torch.device("cpu"),
        history_augment=history_augment,
        history_stream_drop_prob=0.35,
        history_event_drop_prob=0.15,
        history_tail_drop_prob=0.25,
    )
    stats = _summary_stats(batch.history["summary"].cpu().numpy())
    stats["rows"] = float(len(indices))
    stats["kept"] = float(corpus.stats.get("kept", 0))
    stats["raw"] = float(corpus.stats.get("raw", 0))
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--archetype", required=True)
    p.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099", "900-999"])
    p.add_argument("--date-from", default="")
    p.add_argument("--date-to", default="")
    p.add_argument("--deck-sig", action="append", default=[])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-files", type=int, default=0,
                   help="limit npz files after discovery for fast preflight checks")
    p.add_argument("--min-abs-mean", type=float, default=0.002)
    p.add_argument("--max-opp-abs-mean", type=float, default=1e-6)
    args = p.parse_args()

    paths = discover_npz_paths(
        args.corpus,
        args.archetype,
        args.score_bands,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    if not paths:
        raise FileNotFoundError("no npz paths found for requested corpus/archetype/bands")
    if args.max_files > 0:
        paths = paths[: args.max_files]

    cases = [
        ("summary_only_noaug", 0, 0, 0, False),
        ("summary_only_aug", 0, 0, 0, True),
        ("raw_plus_summary_aug", 16, 64, 8, True),
    ]
    failed = False
    print(f"paths={len(paths)} archetype={args.archetype} deck_sig={args.deck_sig or 'all'}")
    for name, hk, lk, bk, aug in cases:
        stats = run_case(
            paths,
            archetype=args.archetype,
            deck_sig=args.deck_sig,
            batch_size=args.batch_size,
            history_k=hk,
            log_history_k=lk,
            board_history_k=bk,
            history_augment=aug,
        )
        line = (
            f"{name}: rows={stats['rows']:.0f} kept={stats['kept']:.0f}/{stats['raw']:.0f} "
            f"abs_mean={stats['abs_mean']:.5f} nonzero={stats['nonzero_rate']:.3f} "
            f"own={stats['own_abs_mean']:.5f} opp={stats['opp_abs_mean']:.8f} "
            f"log={stats['log_abs_mean']:.5f} board={stats['board_abs_mean']:.5f}"
        )
        print(line, flush=True)
        if stats["abs_mean"] < args.min_abs_mean:
            print(f"ERROR: {name} summary looks erased", flush=True)
            failed = True
        if stats["opp_abs_mean"] > args.max_opp_abs_mean:
            print(f"ERROR: {name} contains offline opponent-label summary", flush=True)
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
