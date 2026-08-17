#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _arch_band(path: str, root: str) -> tuple[str, str]:
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if len(parts) >= 3:
        return parts[0], parts[1]
    return "unknown", "unknown"


def _iter_paths(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "*", "*", "*.npz")))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("corpus")
    p.add_argument("--top-cards", type=int, default=12)
    p.add_argument("--min-rows", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=50)
    args = p.parse_args()

    root = str(Path(args.corpus))
    paths = _iter_paths(root)
    if not paths:
        raise FileNotFoundError(f"no .npz files found under {root}")

    rows: Counter[tuple[str, str]] = Counter()
    known_rows: Counter[tuple[str, str]] = Counter()
    known_slots: Counter[tuple[str, str]] = Counter()
    known_count_sum: Counter[tuple[str, str]] = Counter()
    card_counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    missing_known = 0

    for i, path in enumerate(paths, 1):
        arch, band = _arch_band(path, root)
        key = (arch, band)
        z = np.load(path, allow_pickle=True)
        n = len(z["board"])
        rows[key] += n
        if "known_opp_mask" not in z.files:
            missing_known += 1
            continue
        mask = np.asarray(z["known_opp_mask"], dtype=np.float32)
        cards = np.asarray(z.get("known_opp_cards", np.zeros_like(mask)), dtype=np.int64)
        counts = np.asarray(z.get("known_opp_counts", np.zeros_like(mask)), dtype=np.float32)
        slots = mask.sum(axis=-1)
        has = slots > 0
        known_rows[key] += int(has.sum())
        known_slots[key] += float(slots.sum())
        known_count_sum[key] += float((counts * mask).sum())
        if args.top_cards > 0 and bool(has.any()):
            visible = cards[(cards > 0) & (mask > 0)]
            card_counts[key].update(int(x) for x in visible.tolist())
        if args.progress_every and (i == 1 or i % args.progress_every == 0 or i == len(paths)):
            print(f"processed {i}/{len(paths)} files", flush=True)

    total_rows = sum(rows.values())
    total_known = sum(known_rows.values())
    print(
        f"TOTAL files={len(paths)} missing_known_files={missing_known} rows={total_rows} "
        f"known_rows={total_known} known_rate={total_known / max(total_rows, 1):.4f}",
        flush=True,
    )
    print("arch,band,rows,known_rows,known_rate,known_slots_mean,known_count_mean,top_known_cards")
    for key in sorted(rows, key=lambda k: (-known_rows[k] / max(rows[k], 1), -rows[k], k)):
        n = rows[key]
        if n < args.min_rows:
            continue
        kr = known_rows[key]
        slot_mean = known_slots[key] / max(kr, 1)
        count_mean = known_count_sum[key] / max(kr, 1)
        top = " ".join(f"{card}:{cnt}" for card, cnt in card_counts[key].most_common(args.top_cards))
        print(
            f"{key[0]},{key[1]},{n},{kr},{kr / max(n, 1):.4f},"
            f"{slot_mean:.2f},{count_mean:.2f},{top}",
            flush=True,
        )


if __name__ == "__main__":
    main()
