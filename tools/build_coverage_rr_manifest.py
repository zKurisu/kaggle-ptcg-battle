#!/usr/bin/env python3
"""Build an RR manifest that preserves long-tail archetype coverage.

The high-quality RR pool should not silently drop low-frequency archetypes.
This tool merges policy manifests, optional random audits, and live ladder deck
manifests into one eval-compatible CSV.  If an archetype has no usable policy,
its ladder deck can be emitted as a legal-random opponent so failures against
that deck still appear in coverage RR and climb-weighted scoring.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


SUPPORTED_ARCHETYPES = [
    "Marnie Grimmsnarl",
    "Alakazam",
    "Crustle Wall",
    "Dragapult",
    "Mega Lucario",
    "Archaludon",
    "Cynthia Garchomp",
    "Mega Lopunny",
    "Teal Mask Ogerpon",
    "Team Rocket Mewtwo",
    "Festival Lead",
    "Mega Starmie",
    "Iono Bellibolt",
    "Mega Abomasnow",
    "N's Zoroark",
    "Hop Trevenant",
    "Raging Bolt",
]

FIELDS = [
    "name",
    "source_kind",
    "coverage_reason",
    "archetype",
    "team_name",
    "deck_sig",
    "score",
    "score_band",
    "games",
    "wins",
    "losses",
    "weight",
    "random_win_rate",
    "random_games",
    "random_wins",
    "random_timeouts",
    "quality_score",
    "checkpoint_path",
    "policy_path",
    "deck_path",
    "eval_entry",
]


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: str | None, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def rate_str(value: str | None) -> str:
    if value in ("", None):
        return ""
    try:
        x = float(value)
    except Exception:
        return str(value)
    if x > 1.0 and x <= 100.0:
        x /= 100.0
    return f"{x:.6f}"


def clean_name(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "entry"


def row_name(row: dict[str, str]) -> str:
    return (
        row.get("name")
        or row.get("shadow_name")
        or row.get("team_name")
        or row.get("deck_sig")
        or ""
    ).strip()


def policy_path(row: dict[str, str]) -> str:
    return (row.get("policy_path") or row.get("checkpoint_path") or row.get("policy") or "").strip()


def deck_path(row: dict[str, str]) -> str:
    return (row.get("deck_path") or row.get("deck") or "").strip()


def eval_entry(row: dict[str, str]) -> str:
    entry = (row.get("eval_entry") or row.get("entry") or "").strip()
    if entry:
        return entry
    name = clean_name(row_name(row))
    policy = policy_path(row)
    deck = deck_path(row)
    if name and policy and deck:
        return f"{name}={policy}:{deck}"
    if name and deck and row.get("source_kind") == "random_tail":
        return f"{name}=random:{deck}"
    return ""


def key_candidates(row: dict[str, str]) -> list[str]:
    keys = [
        clean_name(row_name(row)),
        policy_path(row),
        Path(policy_path(row)).name if policy_path(row) else "",
        deck_path(row),
        row.get("deck_sig", "").strip(),
        (row.get("eval_entry") or "").split("=", 1)[0].strip(),
    ]
    out: list[str] = []
    for key in keys:
        if key and key not in out:
            out.append(key)
    return out


def build_random_lookup(paths: list[str]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in read_csv(path):
            for key in key_candidates(row):
                lookup.setdefault(key, row)
    return lookup


def enrich_policy_rows(rows: list[dict[str, str]], random_lookup: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        policy = policy_path(row)
        deck = deck_path(row)
        if not policy or not deck:
            continue
        base = dict(row)
        base["source_kind"] = base.get("source_kind") or "policy"
        base["checkpoint_path"] = base.get("checkpoint_path") or policy
        base["policy_path"] = policy
        base["deck_path"] = deck
        base["name"] = clean_name(row_name(base))
        base["eval_entry"] = eval_entry(base)

        rr = None
        for key in key_candidates(base):
            if key in random_lookup:
                rr = random_lookup[key]
                break
        if rr:
            base["random_win_rate"] = rate_str(rr.get("win_rate", rr.get("wr", rr.get("random_win_rate", ""))))
            base["random_games"] = rr.get("games", rr.get("random_games", ""))
            base["random_wins"] = rr.get("wins", rr.get("random_wins", ""))
            base["random_timeouts"] = rr.get("timeouts", rr.get("random_timeouts", ""))
        else:
            base.setdefault("random_win_rate", "")
            base.setdefault("random_games", "")
            base.setdefault("random_wins", "")
            base.setdefault("random_timeouts", "")
        base.setdefault("coverage_reason", "policy_candidate")
        out.append(base)
    return out


def read_policy_rows(paths: list[str], random_paths: list[str]) -> list[dict[str, str]]:
    random_lookup = build_random_lookup(random_paths)
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(enrich_policy_rows(read_csv(path), random_lookup))
    return rows


def read_ladder_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        for row in read_csv(path):
            if not deck_path(row) or not row.get("archetype"):
                continue
            base = dict(row)
            base["source_kind"] = "ladder_deck"
            base["policy_path"] = ""
            base["checkpoint_path"] = ""
            base["deck_path"] = deck_path(base)
            base["name"] = clean_name(row_name(base))
            rows.append(base)
    return rows


def policy_quality(row: dict[str, str]) -> tuple[float, float, float, float, str]:
    random_wr = fnum(row.get("random_win_rate"), -1.0)
    score = fnum(row.get("score"), 0.0)
    weight = fnum(row.get("weight"), 0.0)
    games = fnum(row.get("games"), 0.0)
    q = (
        0.55 * max(random_wr, 0.0)
        + 0.20 * min(max((score - 600.0) / 600.0, 0.0), 1.0)
        + 0.15 * min(weight / 1000.0, 1.0)
        + 0.10 * min(games / 500.0, 1.0)
    )
    return q, random_wr, score, weight, row.get("name", "")


def ladder_quality(row: dict[str, str]) -> tuple[float, float, float, str]:
    return (
        fnum(row.get("score"), 0.0),
        fnum(row.get("weight"), 0.0),
        fnum(row.get("games"), 0.0),
        row.get("name", ""),
    )


def unique_name(base: str, seen: Counter[str]) -> str:
    clean = clean_name(base)
    seen[clean] += 1
    if seen[clean] == 1:
        return clean
    return f"{clean}_{seen[clean]}"


def make_random_tail(row: dict[str, str], *, name_prefix: str, reason: str, seen_names: Counter[str]) -> dict[str, str]:
    base = dict(row)
    sig = base.get("deck_sig", "")
    arch = clean_name(base.get("archetype", "unknown"))
    base["source_kind"] = "random_tail"
    base["coverage_reason"] = reason
    base["policy_path"] = "random"
    base["checkpoint_path"] = "random"
    base["name"] = unique_name(f"{name_prefix}_{arch}_{sig[:8]}_{row_name(base)}", seen_names)
    base["eval_entry"] = f"{base['name']}=random:{deck_path(base)}"
    base.setdefault("random_win_rate", "")
    base.setdefault("random_games", "")
    base.setdefault("random_wins", "")
    base.setdefault("random_timeouts", "")
    return base


def select_rows(args: argparse.Namespace, policy_rows: list[dict[str, str]], ladder_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    wanted = args.archetype or SUPPORTED_ARCHETYPES
    wanted_set = {a.lower() for a in wanted}
    by_policy: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in policy_rows:
        arch = row.get("archetype", "")
        if not arch or arch.lower() not in wanted_set:
            continue
        if fnum(row.get("random_timeouts"), 0.0) > args.max_policy_timeouts:
            continue
        random_wr = fnum(row.get("random_win_rate"), -1.0)
        if random_wr >= 0 and random_wr < args.min_policy_random_wr:
            continue
        by_policy[arch].append(row)

    by_ladder: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ladder_rows:
        arch = row.get("archetype", "")
        if arch and arch.lower() in wanted_set:
            by_ladder[arch].append(row)

    selected: list[dict[str, str]] = []
    seen_key: set[tuple[str, str, str]] = set()
    seen_names: Counter[str] = Counter()

    for arch in wanted:
        policies = sorted(by_policy.get(arch, []), key=policy_quality, reverse=True)
        sig_count: Counter[str] = Counter()
        kept = 0
        for row in policies:
            sig = row.get("deck_sig", "")
            if sig_count[sig] >= args.max_policy_per_deck_sig:
                continue
            out = dict(row)
            out["name"] = unique_name(out.get("name", row_name(out)), seen_names)
            out["eval_entry"] = f"{out['name']}={policy_path(out)}:{deck_path(out)}"
            out["coverage_reason"] = out.get("coverage_reason") or "policy_quality"
            out["quality_score"] = f"{policy_quality(out)[0]:.6f}"
            key = ("policy", policy_path(out), deck_path(out))
            if key not in seen_key:
                selected.append(out)
                seen_key.add(key)
                sig_count[sig] += 1
                kept += 1
            if kept >= args.policy_per_archetype:
                break

        ladders = sorted(by_ladder.get(arch, []), key=ladder_quality, reverse=True)
        need_tail = args.random_tail_per_archetype
        if args.tail_only_missing_policy and kept > 0:
            need_tail = 0
        tail_kept = 0
        tail_sig_count: Counter[str] = Counter()
        for row in ladders:
            sig = row.get("deck_sig", "")
            if tail_sig_count[sig] >= args.max_tail_per_deck_sig:
                continue
            if fnum(row.get("score"), 0.0) < args.min_tail_score and fnum(row.get("games"), 0.0) < args.min_tail_games:
                continue
            tail = make_random_tail(
                row,
                name_prefix=args.random_tail_prefix,
                reason=("forced_missing_policy" if kept == 0 else "live_ladder_tail"),
                seen_names=seen_names,
            )
            key = ("random", "", deck_path(tail))
            if key not in seen_key:
                selected.append(tail)
                seen_key.add(key)
                tail_sig_count[sig] += 1
                tail_kept += 1
            if tail_kept >= need_tail:
                break

    if args.include_other:
        other_rows = sorted(by_ladder.get("Other", []), key=ladder_quality, reverse=True)
        for row in other_rows[: args.random_tail_per_archetype]:
            tail = make_random_tail(
                row,
                name_prefix=args.random_tail_prefix,
                reason="other_live_ladder_tail",
                seen_names=seen_names,
            )
            key = ("random", "", deck_path(tail))
            if key not in seen_key:
                selected.append(tail)
                seen_key.add(key)

    return selected


def output_fields(rows: list[dict[str, str]]) -> list[str]:
    seen = set(FIELDS)
    extras: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                extras.append(key)
                seen.add(key)
    return FIELDS + extras


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = output_fields(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    by_arch = defaultdict(list)
    for row in rows:
        by_arch[row.get("archetype", "")].append(row)
    print(f"rows={len(rows)} archetypes={len(by_arch)}", flush=True)
    for arch in sorted(by_arch):
        kinds = Counter(r.get("source_kind", "") for r in by_arch[arch])
        details = " ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        print(f"  {arch:<22} n={len(by_arch[arch]):2d} {details}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-manifest", action="append", default=[],
                   help="manifest with policy/checkpoint_path + deck_path/eval_entry; repeatable")
    p.add_argument("--random-audit", action="append", default=[],
                   help="optional eval_manifest_random.py CSV for policy quality; repeatable")
    p.add_argument("--ladder-manifest", action="append", default=[],
                   help="pool_manifest.csv with deck_path/score/games/weight; repeatable")
    p.add_argument("--out", required=True)
    p.add_argument("--archetype", action="append", default=[],
                   help="wanted archetype; default is all supported extractor archetypes")
    p.add_argument("--policy-per-archetype", type=int, default=2)
    p.add_argument("--max-policy-per-deck-sig", type=int, default=1)
    p.add_argument("--min-policy-random-wr", type=float, default=0.95)
    p.add_argument("--max-policy-timeouts", type=int, default=0)
    p.add_argument("--random-tail-per-archetype", type=int, default=1)
    p.add_argument("--max-tail-per-deck-sig", type=int, default=1)
    p.add_argument("--min-tail-score", type=float, default=0.0)
    p.add_argument("--min-tail-games", type=float, default=1.0)
    p.add_argument("--tail-only-missing-policy", action="store_true",
                   help="only add random tail rows when an archetype has no kept policy")
    p.add_argument("--include-other", action="store_true")
    p.add_argument("--random-tail-prefix", default="tail")
    args = p.parse_args()

    policy_rows = read_policy_rows(args.policy_manifest, args.random_audit)
    ladder_rows = read_ladder_rows(args.ladder_manifest)
    if not policy_rows and not ladder_rows:
        raise ValueError("provide at least one --policy-manifest or --ladder-manifest")
    rows = select_rows(args, policy_rows, ladder_rows)
    write_csv(Path(args.out), rows)
    print_summary(rows)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
