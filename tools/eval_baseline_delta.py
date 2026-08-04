#!/usr/bin/env python3
"""Paired baseline-vs-candidate validation against a shared opponent pool."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from tools.eval_round_robin import (
    Entry,
    clean_entry_name,
    has_cg_engine,
    load_entries,
    play_matchup,
)


FIELDS = [
    "candidate",
    "baseline",
    "opponent",
    "games",
    "baseline_wins",
    "baseline_losses",
    "baseline_draws",
    "baseline_wr",
    "candidate_wins",
    "candidate_losses",
    "candidate_draws",
    "candidate_wr",
    "delta",
    "weight",
    "opponent_policy",
    "opponent_deck",
]


def read_manifest_entries(path: str, *, limit: int, random_from_deck: bool) -> list[tuple[str, str, float]]:
    entries: list[tuple[str, str, float]] = []
    seen_entries: set[str] = set()
    seen_names: dict[str, int] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if limit and len(entries) >= limit:
                break
            weight = float(row.get("weight") or row.get("trajectory_score") or 1.0)
            name = row.get("shadow_name") or row.get("name") or row.get("team_name") or row.get("deck_sig") or ""
            name = clean_entry_name(name)
            deck = (row.get("deck_path") or row.get("deck") or "").strip()
            policy = (row.get("policy_path") or row.get("checkpoint_path") or row.get("policy") or "").strip()
            if random_from_deck:
                if not deck:
                    continue
                entry = f"{name}=random:{deck}"
            else:
                entry = (row.get("eval_entry") or row.get("entry") or "").strip()
                if not entry:
                    if policy and deck:
                        entry = f"{name}={policy}:{deck}"
                    elif policy:
                        entry = f"{name}={policy}"
            if entry:
                if entry in seen_entries:
                    continue
                seen_entries.add(entry)
                raw_name = entry.split("=", 1)[0] if "=" in entry else entry
                base_name = clean_entry_name(raw_name)
                n = seen_names.get(base_name, 0) + 1
                seen_names[base_name] = n
                unique_name = base_name if n == 1 else f"{base_name}_{n}"
                if unique_name != base_name:
                    if "=" in entry:
                        entry = f"{unique_name}={entry.split('=', 1)[1]}"
                    else:
                        entry = f"{unique_name}={entry}"
                entries.append((unique_name, entry, weight))
    return entries


def opponent_specs(args: argparse.Namespace) -> tuple[list[str], dict[str, float]]:
    specs: list[str] = []
    weights: dict[str, float] = {}
    for spec in args.opponent:
        name = clean_entry_name(spec.split("=", 1)[0] if "=" in spec else spec)
        specs.append(spec)
        weights[name] = 1.0
    for manifest in args.opponent_manifest:
        for name, spec, weight in read_manifest_entries(
            manifest, limit=args.manifest_limit, random_from_deck=args.manifest_random
        ):
            specs.append(spec)
            weights[name] = weight
    if not specs:
        raise ValueError("provide --opponent or --opponent-manifest")
    return specs, weights


def run_baselines(baseline: Entry, opponents: list[Entry], args: argparse.Namespace) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i, opp in enumerate(opponents, 1):
        seed = args.seed + i * 100000
        print(f"\nBaseline {i}/{len(opponents)}: {baseline.name} vs {opp.name}", flush=True)
        out[opp.name] = play_matchup(
            baseline, opp, args.games, args.mcts, args.mcts_sims,
            args.time_budget, args.max_turns, args.progress_every,
            workers=args.workers, seed=seed,
        )
    return out


def run_one(candidate: Entry, baseline: Entry, opponents: list[Entry], weights: dict[str, float],
            baseline_results: dict[str, dict], args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    t0 = time.time()
    for i, opp in enumerate(opponents, 1):
        seed = args.seed + i * 100000
        print(f"\nOpponent {i}/{len(opponents)}: {opp.name}", flush=True)
        b = baseline_results[opp.name]
        print(f"  candidate {candidate.name} vs {opp.name}", flush=True)
        c = play_matchup(
            candidate, opp, args.games, args.mcts, args.mcts_sims,
            args.time_budget, args.max_turns, args.progress_every,
            workers=args.workers, seed=seed,
        )
        b_wr = b["wins_a"] / max(args.games, 1)
        c_wr = c["wins_a"] / max(args.games, 1)
        row = {
            "candidate": candidate.name,
            "baseline": baseline.name,
            "opponent": opp.name,
            "games": args.games,
            "baseline_wins": b["wins_a"],
            "baseline_losses": b["wins_b"],
            "baseline_draws": b["draws"],
            "baseline_wr": b_wr,
            "candidate_wins": c["wins_a"],
            "candidate_losses": c["wins_b"],
            "candidate_draws": c["draws"],
            "candidate_wr": c_wr,
            "delta": c_wr - b_wr,
            "weight": weights.get(opp.name, 1.0),
            "opponent_policy": opp.policy_path,
            "opponent_deck": opp.deck_path,
        }
        rows.append(row)
        print(
            f"  delta={row['delta']:+.3f} candidate={c_wr:.3f} baseline={b_wr:.3f} "
            f"elapsed={time.time()-t0:.0f}s",
            flush=True,
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    avg_delta = sum(float(r["delta"]) for r in rows) / len(rows)
    avg_candidate = sum(float(r["candidate_wr"]) for r in rows) / len(rows)
    avg_baseline = sum(float(r["baseline_wr"]) for r in rows) / len(rows)
    wsum = sum(float(r["weight"]) for r in rows)
    weighted_delta = sum(float(r["delta"]) * float(r["weight"]) for r in rows) / max(wsum, 1e-9)
    worst = min(rows, key=lambda r: float(r["delta"]))
    lost = sum(1 for r in rows if float(r["delta"]) < 0.0)
    return {
        "avg_delta": avg_delta,
        "weighted_delta": weighted_delta,
        "avg_candidate": avg_candidate,
        "avg_baseline": avg_baseline,
        "worst_opponent": worst["opponent"],
        "worst_delta": float(worst["delta"]),
        "lost_matchups": lost,
        "matchups": len(rows),
    }


def write_rows(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ("baseline_wr", "candidate_wr", "delta", "weight"):
                formatted[key] = f"{float(formatted[key]):.6f}"
            w.writerow(formatted)
    print(f"\nWrote {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="NAME=POLICY[:DECK]")
    p.add_argument("--candidate", action="append", required=True, help="NAME=POLICY[:DECK]; repeatable")
    p.add_argument("--opponent", action="append", default=[], help="NAME=POLICY[:DECK], or random:deck")
    p.add_argument("--opponent-manifest", action="append", default=[],
                   help="CSV with eval_entry, or policy/checkpoint + deck_path columns")
    p.add_argument("--manifest-limit", type=int, default=0)
    p.add_argument("--manifest-random", action="store_true",
                   help="use random:deck from manifest deck_path when no eval_entry is present")
    p.add_argument("--registry", default="")
    p.add_argument("--deck", default="deck.csv")
    p.add_argument("--skip-bad-entries", action="store_true")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mcts", action="store_true")
    p.add_argument("--mcts-sims", type=int, default=48)
    p.add_argument("--time-budget", type=float, default=4.0)
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--out-csv", default="logs/baseline_delta.csv")
    args = p.parse_args()

    if not has_cg_engine():
        p.error("cg.game not found; run in the remote/Kaggle workspace with cg available")

    opp_specs, weights = opponent_specs(args)
    specs = [args.baseline, *args.candidate, *opp_specs]
    entries = load_entries(
        specs,
        args.deck,
        include_random=False,
        registry=args.registry,
        skip_bad_entries=args.skip_bad_entries,
    )
    baseline = entries[0]
    candidates = entries[1:1 + len(args.candidate)]
    opponents = entries[1 + len(args.candidate):]
    if not opponents:
        raise ValueError("no valid opponents loaded")

    all_rows: list[dict] = []
    print(
        f"\nRunning shared baseline once: baseline={baseline.name} "
        f"opponents={len(opponents)} games={args.games}",
        flush=True,
    )
    baseline_results = run_baselines(baseline, opponents, args)

    for candidate in candidates:
        print(
            f"\nPaired delta: candidate={candidate.name} baseline={baseline.name} "
            f"opponents={len(opponents)} games={args.games}",
            flush=True,
        )
        rows = run_one(candidate, baseline, opponents, weights, baseline_results, args)
        all_rows.extend(rows)
        s = summarize(rows)
        print(
            "\nSummary "
            f"{candidate.name}: avg_delta={s['avg_delta']:+.3f} "
            f"weighted_delta={s['weighted_delta']:+.3f} "
            f"candidate={s['avg_candidate']:.3f} baseline={s['avg_baseline']:.3f} "
            f"worst={s['worst_opponent']}:{s['worst_delta']:+.3f} "
            f"lost={s['lost_matchups']}/{s['matchups']}",
            flush=True,
        )
    write_rows(args.out_csv, all_rows)


if __name__ == "__main__":
    main()
