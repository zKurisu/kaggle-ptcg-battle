#!/usr/bin/env python3
"""Summarize BC corpus composition by deck signature and decision shape."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.bc2.data import discover_npz_paths

CONTEXT_NAMES = {
    0: "MAIN", 1: "SETUP_ACTIVE", 2: "SETUP_BENCH", 3: "SWITCH", 4: "TO_ACTIVE",
    5: "TO_BENCH", 6: "TO_FIELD", 7: "TO_HAND", 8: "DISCARD", 13: "DAMAGE_COUNTER",
    15: "DAMAGE", 16: "REMOVE_DAMAGE_COUNTER", 21: "ATTACH_FROM", 22: "ATTACH_TO",
    30: "DISCARD_ENERGY", 34: "SKILL_ORDER", 37: "EVOLVE", 40: "REMOVE_DAMAGE_COUNTER_COUNT",
    41: "IS_FIRST", 43: "ACTIVATE",
}


def label_status(data: dict[str, np.ndarray], i: int, include_empty: bool) -> str:
    action = np.asarray(data["action"][i], dtype=np.int64)
    n_opt = len(data["ot"][i])
    mn = int(data["min_c"][i])
    mx = int(data["max_c"][i])
    if len(action) == 0:
        return "keep" if include_empty and mn == 0 else "empty"
    if len(action) < mn or len(action) > mx:
        return "bad"
    if len(set(action.tolist())) != len(action):
        return "bad"
    if not ((action >= 0) & (action < n_opt)).all():
        return "bad"
    return "keep"


def context_id(feats) -> int:
    arr = np.asarray(feats, dtype=np.float32)
    if arr.shape[0] <= 17:
        return -1
    return int(round(float(arr[17]) * 64.0))


def opt_bucket(n: int) -> str:
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    return "11+"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/bc_corpus_banded_v6wide")
    p.add_argument("--archetype", required=True)
    p.add_argument("--score-bands", nargs="+", default=["1200+", "1100-1199", "1000-1099"])
    p.add_argument("--include-empty", action="store_true")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out-csv", default="")
    args = p.parse_args()

    paths = discover_npz_paths(args.corpus, args.archetype, args.score_bands)
    if not paths:
        raise FileNotFoundError(f"No files for {args.archetype} {args.score_bands} in {args.corpus}")

    by_deck = defaultdict(lambda: {
        "raw": 0, "kept": 0, "empty": 0, "bad": 0, "scores": [], "teams": Counter(),
        "contexts": Counter(), "option_counts": Counter(), "files": Counter(), "episodes": set(),
    })
    global_counts = Counter()
    has_sig = True

    for path in paths:
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        if "deck_sig" not in data:
            has_sig = False
        n = len(data["board"])
        for i in range(n):
            sig = str(data["deck_sig"][i]) if has_sig else "(missing-deck-sig)"
            row = by_deck[sig]
            status = label_status(data, i, args.include_empty)
            field = "kept" if status == "keep" else status
            global_counts["raw"] += 1
            global_counts[field] += 1
            row["raw"] += 1
            row[field] += 1
            row["files"][Path(path).name] += 1
            if "score" in data:
                row["scores"].append(float(data["score"][i]))
            if "team_name" in data:
                team = str(data["team_name"][i])
                if team:
                    row["teams"][team] += 1
            if "episode_id" in data:
                row["episodes"].add(str(data["episode_id"][i]))
            if status == "keep":
                row["contexts"][context_id(data["feats"][i])] += 1
                row["option_counts"][opt_bucket(len(data["ot"][i]))] += 1

    print(f"Corpus: {args.corpus}")
    print(f"Archetype: {args.archetype} bands={args.score_bands}")
    print(f"Files: {len(paths)}")
    print(f"Global: {dict(global_counts)}")
    if not has_sig:
        print("WARNING: corpus has no deck_sig metadata. Re-extract with updated tools/bc_extract_v2.py for deck-specific stats.")

    rows = []
    for sig, row in by_deck.items():
        kept = int(row["kept"])
        raw = int(row["raw"])
        scores = row["scores"]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        top_team = row["teams"].most_common(1)[0][0] if row["teams"] else ""
        top_ctx = row["contexts"].most_common(1)[0] if row["contexts"] else (-1, 0)
        top_opt = row["option_counts"].most_common(1)[0] if row["option_counts"] else ("", 0)
        rows.append({
            "deck_sig": sig,
            "raw": raw,
            "kept": kept,
            "empty": int(row["empty"]),
            "bad": int(row["bad"]),
            "episodes": len(row["episodes"]) or "",
            "avg_score": avg_score,
            "teams": len(row["teams"]),
            "top_team": top_team,
            "top_context": CONTEXT_NAMES.get(top_ctx[0], str(top_ctx[0])),
            "top_context_n": top_ctx[1],
            "top_option_count": top_opt[0],
            "top_option_count_n": top_opt[1],
        })
    rows.sort(key=lambda r: (r["kept"], r["raw"], r["avg_score"]), reverse=True)

    print("\nTop deck signatures:")
    for r in rows[: args.top]:
        print(
            f"  {r['deck_sig']:12s} kept={r['kept']:8d} raw={r['raw']:8d} "
            f"episodes={str(r['episodes']):>5s} avg_score={r['avg_score']:.1f} "
            f"teams={r['teams']:4d} ctx={r['top_context']} team={r['top_team'][:30]}"
        )

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["deck_sig"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
