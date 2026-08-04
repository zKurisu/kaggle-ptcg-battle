#!/usr/bin/env python3
"""Evaluate manifest policy entries against same-deck legal random."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

import tools.eval_bc as eval_bc
from ptcg_rl.numpy_policy import NumpyPolicy
from tools.eval_baseline_delta import read_manifest_entries
from tools.eval_round_robin import clean_entry_name, parse_entry


FIELDS = [
    "rank",
    "name",
    "archetype",
    "team_name",
    "deck_sig",
    "weight",
    "games",
    "wins",
    "win_rate",
    "timeouts",
    "seconds",
    "policy_path",
    "deck_path",
]


def read_manifest_meta(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ckpt = (row.get("checkpoint_path") or row.get("policy_path") or row.get("policy") or "").strip()
            if ckpt and ckpt not in out:
                out[ckpt] = row
    return out


def load_done_names(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    with p.open(newline="") as f:
        return {clean_entry_name(r.get("name", "")) for r in csv.DictReader(f) if r.get("name")}


def write_header_if_needed(path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return
    with out.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def append_row(path: str, row: dict) -> None:
    with Path(path).open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(row)
        f.flush()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="CSV with eval_entry/checkpoint_path/deck_path columns")
    p.add_argument("--limit", type=int, default=0, help="max entries after filters; 0 means all")
    p.add_argument("--offset", type=int, default=0, help="skip this many entries after filters")
    p.add_argument("--archetype", action="append", default=[], help="optional exact archetype filter; repeatable")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--progress-every", type=int, default=50, help="per-policy game progress; 0 disables")
    p.add_argument("--skip-bad-entries", action="store_true")
    p.add_argument("--resume", action="store_true", help="skip names already present in --out-csv")
    p.add_argument("--out-csv", default="logs/eval_manifest_random.csv")
    args = p.parse_args()

    meta_by_policy = read_manifest_meta(args.manifest)
    wanted_arch = {x.lower() for x in args.archetype}
    done_names = load_done_names(args.out_csv) if args.resume else set()
    entries = read_manifest_entries(args.manifest, limit=0, random_from_deck=False)

    selected = []
    for name, spec, weight in entries:
        _entry_name, policy_path, deck_path = parse_entry(spec, default_deck="")
        meta = meta_by_policy.get(policy_path, {})
        arch = str(meta.get("archetype") or "")
        if wanted_arch and arch.lower() not in wanted_arch:
            continue
        selected.append((name, spec, weight, policy_path, deck_path, meta))

    if args.offset:
        selected = selected[args.offset:]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("no manifest entries selected")

    write_header_if_needed(args.out_csv)
    print(
        f"Manifest random eval: entries={len(selected)} games={args.games} "
        f"workers={args.workers} out={args.out_csv}",
        flush=True,
    )

    t_all = time.time()
    for i, (name, _spec, weight, policy_path, deck_path, meta) in enumerate(selected, 1):
        safe_name = clean_entry_name(name)
        if args.resume and safe_name in done_names:
            print(f"\nSkip completed {i}/{len(selected)} {safe_name}", flush=True)
            continue
        print(
            f"\nEntry {i}/{len(selected)} {safe_name} "
            f"arch={meta.get('archetype', '')} team={meta.get('team_name', '')}",
            flush=True,
        )
        t0 = time.time()
        try:
            policy = NumpyPolicy.load(policy_path)
            deck = eval_bc.load_deck(deck_path)
            wr = eval_bc.eval_vs_random(
                policy,
                deck,
                policy_path,
                games=args.games,
                workers=args.workers,
                seed=args.seed + i * 100000,
                max_turns=args.max_turns,
                progress_every=args.progress_every,
            )
        except Exception as exc:
            if not args.skip_bad_entries:
                raise
            print(f"Skipping bad entry {safe_name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        seconds = time.time() - t0
        wins = int(round(wr * args.games))
        row = {
            "rank": meta.get("rank", ""),
            "name": safe_name,
            "archetype": meta.get("archetype", ""),
            "team_name": meta.get("team_name", ""),
            "deck_sig": meta.get("deck_sig", ""),
            "weight": f"{float(weight):.6f}",
            "games": args.games,
            "wins": wins,
            "win_rate": f"{wr:.6f}",
            "timeouts": eval_bc._LAST_TIMEOUTS,
            "seconds": f"{seconds:.1f}",
            "policy_path": policy_path,
            "deck_path": deck_path,
        }
        append_row(args.out_csv, row)
        done_names.add(safe_name)
        print(
            f"  result {safe_name}: wr={wr:.3f} wins={wins}/{args.games} "
            f"timeouts={eval_bc._LAST_TIMEOUTS} seconds={seconds:.1f}",
            flush=True,
        )

    print(f"\nDone in {(time.time() - t_all) / 60.0:.1f}m. Wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
