#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.seq.constants import TYPE_ABILITY, TYPE_ATTACH, TYPE_ATTACK, TYPE_EVOLVE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=5000)
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.corpus, "*", "*", "*.npz")))
    if args.max_files:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"no v14 npz found under {args.corpus}")
    rng = random.Random(args.seed)

    stats = Counter()
    examples: list[str] = []
    per_file = max(1, args.samples // max(len(paths), 1))
    print(
        f"integrity_start files={len(paths)} per_file={per_file} target_rows~={per_file * len(paths)}",
        flush=True,
    )
    for file_i, path in enumerate(paths, 1):
        print(f"  file {file_i}/{len(paths)} start {path}", flush=True)
        with np.load(path, allow_pickle=True) as z:
            n = len(z["action"])
            ids = list(range(n))
            rng.shuffle(ids)
            for i in ids[:per_file]:
                stats["rows"] += 1
                action = np.asarray(z["action"][i], dtype=np.int64).reshape(-1)
                ot = np.asarray(z["ot"][i], dtype=np.int64).reshape(-1)
                mn = int(z["min_c"][i])
                mx = int(z["max_c"][i])
                act_type = int(z["act_type"][i]) if "act_type" in z else -999
                plan = np.asarray(z["future_plan"][i], dtype=np.float32).reshape(-1) if "future_plan" in z else np.zeros(0)

                if len(action) < mn or len(action) > mx:
                    _bad(stats, examples, path, i, "count_outside_minmax")
                if len(set(action.tolist())) != len(action):
                    _bad(stats, examples, path, i, "duplicate_action_index")
                if np.any(action < 0) or np.any(action >= len(ot)):
                    _bad(stats, examples, path, i, "action_index_out_of_range")
                    continue
                if len(action):
                    if act_type != int(ot[int(action[0])]):
                        _bad(stats, examples, path, i, f"act_type_mismatch saved={act_type} real={int(ot[int(action[0])])}")
                if plan.size:
                    if np.any(~np.isfinite(plan)):
                        _bad(stats, examples, path, i, "future_plan_nonfinite")
                    if np.min(plan) < -1.001 or np.max(plan) > 1.001:
                        _bad(stats, examples, path, i, "future_plan_out_of_range")
                    for idx in (0, 1, 2, 3, 5, 9, 10, 11, 12):
                        if idx < plan.size and plan[idx] < -1e-6:
                            _bad(stats, examples, path, i, f"future_plan_negative_{idx}")
                stats[f"type_{act_type}"] += 1
                if len(action) > 1:
                    stats["multi_rows"] += 1
                if len(action) == 0:
                    stats["empty_rows"] += 1
                if args.progress_every and stats["rows"] % args.progress_every == 0:
                    print(
                        f"  checked rows={stats['rows']} files={file_i}/{len(paths)} "
                        f"bad={stats.get('bad', 0)} multi={stats.get('multi_rows', 0)}",
                        flush=True,
                    )
        print(
            f"  file {file_i}/{len(paths)} done rows={stats['rows']} "
            f"bad={stats.get('bad', 0)}",
            flush=True,
        )

    print("integrity_stats", flush=True)
    for key, value in stats.most_common():
        print(f"{key}: {value}", flush=True)
    if examples:
        print("bad_examples", flush=True)
        for ex in examples[:30]:
            print(ex, flush=True)
        raise SystemExit(1)
    print("OK", flush=True)


def _bad(stats: Counter, examples: list[str], path: str, i: int, reason: str) -> None:
    stats["bad"] += 1
    stats[f"bad_{reason.split()[0]}"] += 1
    if len(examples) < 50:
        examples.append(f"{reason} file={path} row={i}")


if __name__ == "__main__":
    main()
