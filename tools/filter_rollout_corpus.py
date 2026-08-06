#!/usr/bin/env python3
"""Filter generated rollout BC corpora by actor/opponent/status metadata."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


def compile_any(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p) for p in patterns if p]


def match_any(value: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(value) for p in patterns)


def discover_files(root: str, score_bands: list[str]) -> list[Path]:
    paths: list[Path] = []
    root_path = Path(root)
    for band in score_bands:
        paths.extend(root_path.glob(f"*/*{band}*/*.npz"))
        paths.extend(root_path.glob(f"*/{band}/*.npz"))
    return sorted(set(paths))


def relative_out_path(src: Path, in_root: Path, out_root: Path) -> Path:
    try:
        rel = src.relative_to(in_root)
    except ValueError:
        rel = Path(src.name)
    return out_root / rel


def as_str_arr(z, key: str, n: int) -> np.ndarray:
    if key not in z.files:
        return np.asarray([""] * n, dtype=object)
    arr = np.asarray(z[key], dtype=object)
    if arr.shape[:1] != (n,):
        return np.asarray([""] * n, dtype=object)
    return arr


def filter_file(src: Path, dst: Path, args: argparse.Namespace,
                include_actor: list[re.Pattern], exclude_actor: list[re.Pattern],
                include_opp: list[re.Pattern], exclude_opp: list[re.Pattern]) -> tuple[int, int]:
    with np.load(src, allow_pickle=True) as z:
        if "board" not in z.files:
            return 0, 0
        n = len(z["board"])
        mask = np.ones(n, dtype=bool)
        if args.won_only and "won" in z.files:
            mask &= np.asarray(z["won"]).astype(np.int8) == 1
        if args.final_status:
            status = as_str_arr(z, "final_status", n)
            keep = np.zeros(n, dtype=bool)
            wanted = set(args.final_status)
            for i, value in enumerate(status):
                keep[i] = str(value) in wanted
            mask &= keep

        actor = as_str_arr(z, "actor_mode", n)
        if include_actor:
            mask &= np.asarray([match_any(str(v), include_actor) for v in actor], dtype=bool)
        if exclude_actor:
            mask &= ~np.asarray([match_any(str(v), exclude_actor) for v in actor], dtype=bool)

        opp = as_str_arr(z, "opponent_name", n)
        if include_opp:
            mask &= np.asarray([match_any(str(v), include_opp) for v in opp], dtype=bool)
        if exclude_opp:
            mask &= ~np.asarray([match_any(str(v), exclude_opp) for v in opp], dtype=bool)

        kept = int(mask.sum())
        if kept < args.min_rows:
            return n, 0

        out = {}
        for key in z.files:
            arr = z[key]
            if getattr(arr, "shape", ())[:1] == (n,):
                out[key] = arr[mask]
            else:
                out[key] = arr
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **out)
        return n, kept


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--score-bands", nargs="+", default=["weak_win_search", "win_search", "generated"])
    p.add_argument("--actor-include-regex", action="append", default=[])
    p.add_argument("--actor-exclude-regex", action="append", default=["fallback_random", "epsilon_random"])
    p.add_argument("--opponent-include-regex", action="append", default=[])
    p.add_argument("--opponent-exclude-regex", action="append", default=[])
    p.add_argument("--final-status", action="append", default=[])
    p.add_argument("--won-only", action="store_true", default=True)
    p.add_argument("--min-rows", type=int, default=1)
    args = p.parse_args()

    include_actor = compile_any(args.actor_include_regex)
    exclude_actor = compile_any(args.actor_exclude_regex)
    include_opp = compile_any(args.opponent_include_regex)
    exclude_opp = compile_any(args.opponent_exclude_regex)
    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    total = 0
    kept = 0
    written = 0
    for src in discover_files(args.in_root, args.score_bands):
        dst = relative_out_path(src, in_root, out_root)
        n, k = filter_file(src, dst, args, include_actor, exclude_actor, include_opp, exclude_opp)
        total += n
        kept += k
        written += int(k > 0)
    print(f"Filtered rollout corpus: files={written} rows={kept}/{total} out={args.out_root}", flush=True)


if __name__ == "__main__":
    main()
