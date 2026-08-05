#!/usr/bin/env python3
"""Split audited shadow policies into evaluation pools.

The output CSVs remain manifest-compatible: every row keeps eval_entry,
checkpoint_path, and deck_path so the files can be passed directly to the
evaluation tools.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except Exception:
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "") or default))
    except Exception:
        return default


def entry_name(row: dict[str, str]) -> str:
    return (row.get("name") or row.get("shadow_name") or "").strip()


def policy_path(row: dict[str, str]) -> str:
    return (row.get("policy_path") or row.get("checkpoint_path") or "").strip()


def deck_path(row: dict[str, str]) -> str:
    return (row.get("deck_path") or row.get("deck") or "").strip()


def eval_entry(row: dict[str, str]) -> str:
    entry = (row.get("eval_entry") or "").strip()
    if entry:
        return entry
    name = entry_name(row)
    policy = policy_path(row)
    deck = deck_path(row)
    if name and policy and deck:
        return f"{name}={policy}:{deck}"
    return ""


def row_weight(row: dict[str, str]) -> float:
    return as_float(row, "trajectory_score", as_float(row, "weight", 0.0))


def random_wr(row: dict[str, str]) -> float:
    return as_float(row, "random_win_rate", as_float(row, "win_rate", 0.0))


def random_timeouts(row: dict[str, str]) -> int:
    return as_int(row, "random_timeouts", as_int(row, "timeouts", 0))


def quality_tier(row: dict[str, str]) -> str:
    wr = random_wr(row)
    timeouts = random_timeouts(row)
    if timeouts > 0:
        return "debug_timeout"
    if wr >= 0.99:
        return "trusted_ge099"
    if wr >= 0.97:
        return "trusted_ge097"
    if wr >= 0.95:
        return "usable_ge095"
    if wr >= 0.90:
        return "watch_ge090"
    return "debug_low_random"


def row_sort_key(row: dict[str, str]) -> tuple[float, float, float, str]:
    return (random_wr(row), row_weight(row), as_float(row, "decisions"), entry_name(row))


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (policy_path(row), deck_path(row))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def with_reason(row: dict[str, str], reason: str) -> dict[str, str]:
    out = dict(row)
    out["pool_reason"] = reason
    return out


def best_by_archetype(
    rows: list[dict[str, str]],
    *,
    cap: int,
    min_wr: float,
    per_deck_sig: int,
    fallback: bool,
) -> list[dict[str, str]]:
    by_arch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_arch[row["archetype"]].append(row)

    selected = []
    for arch in sorted(by_arch, key=lambda a: -sum(row_weight(r) for r in by_arch[a])):
        candidates = sorted(by_arch[arch], key=row_sort_key, reverse=True)
        good = [r for r in candidates if random_timeouts(r) == 0 and random_wr(r) >= min_wr]
        deck_counts: Counter[str] = Counter()
        arch_rows = []
        for row in good:
            sig = row.get("deck_sig", "")
            if deck_counts[sig] >= per_deck_sig:
                continue
            arch_rows.append(with_reason(row, f"wr>={min_wr:.2f}"))
            deck_counts[sig] += 1
            if len(arch_rows) >= cap:
                break
        if fallback and not arch_rows and candidates:
            no_timeout = [r for r in candidates if random_timeouts(r) == 0]
            row = (no_timeout or candidates)[0]
            arch_rows.append(with_reason(row, f"forced_arch_fallback_best_wr={random_wr(row):.3f}"))
        selected.extend(arch_rows)
    return dedupe(selected)


def environment_pool(rows: list[dict[str, str]], cap: int, per_deck_sig: int) -> list[dict[str, str]]:
    return best_by_archetype(
        rows,
        cap=cap,
        min_wr=0.90,
        per_deck_sig=per_deck_sig,
        fallback=True,
    )


def stress_pool(rows: list[dict[str, str]], cap: int, per_deck_sig: int) -> list[dict[str, str]]:
    by_arch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_arch[row["archetype"]].append(row)

    selected = []
    high_cap = max(1, cap // 2)
    low_cap = max(1, cap - high_cap)
    for arch in sorted(by_arch, key=lambda a: -sum(row_weight(r) for r in by_arch[a])):
        candidates = sorted(by_arch[arch], key=lambda r: (row_weight(r), random_wr(r)), reverse=True)
        high = [r for r in candidates if random_timeouts(r) == 0 and random_wr(r) >= 0.95]
        low = sorted(
            [r for r in candidates if random_timeouts(r) > 0 or random_wr(r) < 0.95],
            key=lambda r: (-random_timeouts(r), random_wr(r), -row_weight(r), entry_name(r)),
        )

        arch_rows = []
        deck_counts: Counter[str] = Counter()
        for reason, source, limit in (("high_trajectory_quality", high, high_cap), ("low_random_or_timeout", low, low_cap)):
            for row in source:
                sig = row.get("deck_sig", "")
                if deck_counts[sig] >= per_deck_sig:
                    continue
                if any(policy_path(row) == policy_path(x) and deck_path(row) == deck_path(x) for x in arch_rows):
                    continue
                arch_rows.append(with_reason(row, reason))
                deck_counts[sig] += 1
                if sum(1 for x in arch_rows if x["pool_reason"] == reason) >= limit:
                    break
        if not arch_rows and candidates:
            arch_rows.append(with_reason(candidates[0], "forced_arch_fallback"))
        selected.extend(arch_rows[:cap])
    return dedupe(selected)


def debug_pool(rows: list[dict[str, str]], cap: int) -> list[dict[str, str]]:
    by_arch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if random_timeouts(row) > 0 or random_wr(row) < 0.95:
            by_arch[row["archetype"]].append(row)

    selected = []
    for arch in sorted(by_arch, key=lambda a: -len(by_arch[a])):
        ranked = sorted(
            by_arch[arch],
            key=lambda r: (-random_timeouts(r), random_wr(r), -row_weight(r), entry_name(r)),
        )
        selected.extend(with_reason(row, "debug_random_quality") for row in ranked[:cap])
    return dedupe(selected)


def enrich_rows(
    manifest_rows: list[dict[str, str]],
    random_rows: list[dict[str, str]],
    *,
    source_kind: str,
) -> list[dict[str, str]]:
    manifest_by_name = {entry_name(r): r for r in manifest_rows if entry_name(r)}
    manifest_by_policy = {policy_path(r): r for r in manifest_rows if policy_path(r)}

    out = []
    for rr in random_rows:
        name = entry_name(rr)
        policy = policy_path(rr)
        base = dict(manifest_by_name.get(name) or manifest_by_policy.get(policy) or {})
        base.update({
            "source_kind": source_kind,
            "name": name,
            "shadow_name": base.get("shadow_name") or name,
            "archetype": rr.get("archetype") or base.get("archetype", ""),
            "team_name": rr.get("team_name") or base.get("team_name", ""),
            "deck_sig": rr.get("deck_sig") or base.get("deck_sig", ""),
            "weight": base.get("weight") or rr.get("weight", ""),
            "deck_path": rr.get("deck_path") or base.get("deck_path", ""),
            "checkpoint_path": rr.get("policy_path") or base.get("checkpoint_path", ""),
            "policy_path": rr.get("policy_path") or base.get("checkpoint_path", ""),
            "random_rank": rr.get("rank", ""),
            "random_games": rr.get("games", ""),
            "random_wins": rr.get("wins", ""),
            "random_win_rate": rr.get("win_rate", ""),
            "random_timeouts": rr.get("timeouts", ""),
            "random_seconds": rr.get("seconds", ""),
            "quality_tier": "",
            "pool_reason": "",
        })
        base["eval_entry"] = eval_entry(base)
        base["quality_tier"] = quality_tier(base)
        if base["eval_entry"]:
            out.append(base)
    return dedupe(out)


def add_missing_fallbacks(
    rows: list[dict[str, str]],
    fallback_rows: list[dict[str, str]],
    *,
    per_missing_archetype: int,
) -> list[dict[str, str]]:
    if not fallback_rows:
        return rows
    present = {row["archetype"] for row in rows if row.get("archetype")}
    all_arches = {row["archetype"] for row in fallback_rows if row.get("archetype")}
    selected = list(rows)
    for arch in sorted(all_arches - present):
        candidates = sorted(
            [row for row in fallback_rows if row.get("archetype") == arch],
            key=lambda row: (random_timeouts(row) == 0, row_weight(row), random_wr(row), entry_name(row)),
            reverse=True,
        )
        for row in candidates[:per_missing_archetype]:
            selected.append(with_reason(row, "pop_fallback_missing_shadow_archetype"))
    return dedupe(selected)


def output_fields(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
        "name",
        "shadow_name",
        "source_kind",
        "archetype",
        "team_name",
        "deck_sig",
        "trajectory_score",
        "weight",
        "avg_score",
        "max_score",
        "episodes",
        "decisions",
        "dates",
        "first_date",
        "last_date",
        "decision_win_rate",
        "random_games",
        "random_wins",
        "random_win_rate",
        "random_timeouts",
        "quality_tier",
        "pool_reason",
        "checkpoint_path",
        "policy_path",
        "deck_path",
        "eval_entry",
    ]
    seen = set(preferred)
    extras = []
    for row in rows:
        for key in row:
            if key not in seen:
                extras.append(key)
                seen.add(key)
    return preferred + extras


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pool_stats(name: str, rows: list[dict[str, str]]) -> list[str]:
    by_arch = Counter(row["archetype"] for row in rows)
    wrs = [random_wr(r) for r in rows]
    weights = [row_weight(r) for r in rows]
    weighted = sum(w * r for w, r in zip(weights, wrs)) / max(sum(weights), 1e-9)
    lines = [
        f"{name}: n={len(rows)} archetypes={len(by_arch)} "
        f"mean_wr={sum(wrs) / max(len(wrs), 1):.3f} weighted_wr={weighted:.3f} "
        f"min_wr={min(wrs) if wrs else 0.0:.3f} timeouts={sum(random_timeouts(r) for r in rows)}",
    ]
    for arch, count in sorted(by_arch.items(), key=lambda kv: (-kv[1], kv[0])):
        vals = [random_wr(r) for r in rows if r["archetype"] == arch]
        lines.append(f"  {arch:<22} n={count:2d} min={min(vals):.3f} mean={sum(vals)/len(vals):.3f} max={max(vals):.3f}")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--random", required=True, help="eval_manifest_random.py output CSV")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--env-per-archetype", type=int, default=6)
    p.add_argument("--quality-per-archetype", type=int, default=5)
    p.add_argument("--stress-per-archetype", type=int, default=4)
    p.add_argument("--debug-per-archetype", type=int, default=5)
    p.add_argument("--per-deck-sig", type=int, default=2)
    p.add_argument("--fallback-random", action="append", default=[],
                   help="optional non-shadow random audit CSV used only to cover missing archetypes")
    p.add_argument("--fallback-per-missing-archetype", type=int, default=2)
    args = p.parse_args()

    manifest_rows = read_csv(args.manifest)
    random_rows = read_csv(args.random)
    rows = enrich_rows(manifest_rows, random_rows, source_kind="shadow")
    fallback_rows: list[dict[str, str]] = []
    for fallback in args.fallback_random:
        fallback_rows.extend(enrich_rows([], read_csv(fallback), source_kind="fallback"))
    fields = output_fields(rows)
    out_dir = Path(args.out_dir)

    strict = [with_reason(r, "strict_wr>=0.97_timeout0") for r in rows if random_timeouts(r) == 0 and random_wr(r) >= 0.97]
    quality = best_by_archetype(
        rows,
        cap=args.quality_per_archetype,
        min_wr=0.97,
        per_deck_sig=args.per_deck_sig,
        fallback=True,
    )
    env = environment_pool(rows, cap=args.env_per_archetype, per_deck_sig=args.per_deck_sig)
    stress = stress_pool(rows, cap=args.stress_per_archetype, per_deck_sig=args.per_deck_sig)
    debug = debug_pool(rows, cap=args.debug_per_archetype)

    outputs = {
        "shadow_all_enriched.csv": rows,
        "shadow_pool_quality_strict_ge097.csv": strict,
        "shadow_pool_quality_balanced_inclusive.csv": quality,
        "shadow_pool_environment_balanced.csv": env,
        "shadow_pool_stress_balanced.csv": stress,
        "shadow_pool_debug_low_random.csv": debug,
    }
    if fallback_rows:
        mixed_outputs = {
            "mixed_shadow_popfallback_quality_balanced_inclusive.csv": add_missing_fallbacks(
                quality, fallback_rows, per_missing_archetype=args.fallback_per_missing_archetype
            ),
            "mixed_shadow_popfallback_environment_balanced.csv": add_missing_fallbacks(
                env, fallback_rows, per_missing_archetype=args.fallback_per_missing_archetype
            ),
            "mixed_shadow_popfallback_stress_balanced.csv": add_missing_fallbacks(
                stress, fallback_rows, per_missing_archetype=args.fallback_per_missing_archetype
            ),
        }
        outputs.update(mixed_outputs)
        fields = output_fields(rows + fallback_rows)

    for filename, pool_rows in outputs.items():
        write_csv(out_dir / filename, pool_rows, fields)

    lines = [
        "Shadow pool split summary",
        f"manifest={args.manifest}",
        f"random={args.random}",
        f"audited_rows={len(rows)} manifest_rows={len(manifest_rows)} random_rows={len(random_rows)} "
        f"fallback_rows={len(fallback_rows)}",
        "",
    ]
    for filename, pool_rows in outputs.items():
        lines.extend(pool_stats(filename, pool_rows))
        lines.append("")
    summary = out_dir / "shadow_pool_split_summary.txt"
    summary.write_text("\n".join(lines))
    print(summary.read_text())
    for filename in outputs:
        print(f"Wrote {out_dir / filename}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
