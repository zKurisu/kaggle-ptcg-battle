#!/usr/bin/env python3
"""Build filtered RR opponent pools from audited policy manifests.

This tool is intentionally stricter than split_shadow_pools.py. A policy that
beats random can still be a weak local opponent and inflate every candidate's
headline RR score, so the primary pool also uses RR self-strength and live
ladder support when available.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in ("", None) else default
    except Exception:
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        return int(float(value)) if value not in ("", None) else default
    except Exception:
        return default


def clean_name(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def entry_name(row: dict[str, str]) -> str:
    return (
        row.get("name")
        or row.get("shadow_name")
        or row.get("eval_entry", "").split("=", 1)[0]
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
    name = clean_name(entry_name(row))
    policy = policy_path(row)
    deck = deck_path(row)
    if name and policy and deck:
        return f"{name}={policy}:{deck}"
    return ""


def key_candidates(row: dict[str, str]) -> list[str]:
    out = []
    for value in (
        clean_name(entry_name(row)),
        policy_path(row),
        deck_path(row),
        row.get("deck_sig", "").strip(),
    ):
        if value and value not in out:
            out.append(value)
    return out


def build_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        for key in key_candidates(row):
            lookup.setdefault(key, row)
    return lookup


def aggregate_rr(path: str) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    wins: Counter[str] = Counter()
    games: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    for row in read_csv(path):
        name = clean_name(row.get("row", ""))
        if not name:
            continue
        g = as_int(row, "games", 0)
        if g <= 0:
            continue
        wins[name] += as_float(row, "row_wins", as_float(row, "row_win_rate", 0.0) * g)
        games[name] += g
        pair_counts[name] += 1
    stats = {}
    for name, g in games.items():
        stats[name] = {
            "rr_games": float(g),
            "rr_pairs": float(pair_counts[name]),
            "rr_mean_wr": float(wins[name]) / max(float(g), 1.0),
        }
    return stats


def aggregate_env(paths: list[str]) -> dict[str, dict[str, float | str]]:
    by_sig: dict[str, dict[str, float | str]] = {}
    for path in paths:
        for row in read_csv(path):
            sig = row.get("deck_sig", "").strip()
            if not sig:
                continue
            score = as_float(row, "score", 0.0)
            weight = as_float(row, "weight", 0.0)
            games = as_float(row, "games", 0.0)
            current = by_sig.setdefault(
                sig,
                {
                    "env_score": 0.0,
                    "env_weight": 0.0,
                    "env_games": 0.0,
                    "env_sources": "",
                    "env_score_band": "",
                    "env_team_name": "",
                },
            )
            current["env_weight"] = float(current["env_weight"]) + weight
            current["env_games"] = float(current["env_games"]) + games
            sources = str(current["env_sources"])
            source_token = Path(path).parent.name
            if source_token not in sources.split("|"):
                current["env_sources"] = f"{sources}|{source_token}".strip("|")
            if score >= float(current["env_score"]):
                current["env_score"] = score
                current["env_score_band"] = row.get("score_band", "")
                current["env_team_name"] = row.get("team_name", "")
    return by_sig


def enrich_rows(
    manifest_rows: list[dict[str, str]],
    random_rows: list[dict[str, str]],
    rr_stats: dict[str, dict[str, float]],
    env_stats: dict[str, dict[str, float | str]],
) -> list[dict[str, str]]:
    random_lookup = build_lookup(random_rows)
    out = []
    seen: set[tuple[str, str]] = set()
    for row in manifest_rows:
        base = dict(row)
        random_row = None
        for key in key_candidates(row):
            if key in random_lookup:
                random_row = random_lookup[key]
                break
        if random_row:
            base["random_games"] = random_row.get("games", random_row.get("random_games", ""))
            base["random_wins"] = random_row.get("wins", random_row.get("random_wins", ""))
            base["random_win_rate"] = random_row.get("win_rate", random_row.get("random_win_rate", ""))
            base["random_timeouts"] = random_row.get("timeouts", random_row.get("random_timeouts", ""))
        else:
            base.setdefault("random_games", "")
            base.setdefault("random_wins", "")
            base.setdefault("random_win_rate", "")
            base.setdefault("random_timeouts", "")

        name = clean_name(entry_name(base))
        base["name"] = name
        base["policy_path"] = policy_path(base)
        base["checkpoint_path"] = base.get("checkpoint_path") or base["policy_path"]
        base["deck_path"] = deck_path(base)
        base["eval_entry"] = eval_entry(base)

        rr = rr_stats.get(name, {})
        base["rr_mean_wr"] = f"{rr.get('rr_mean_wr', 0.0):.6f}" if rr else ""
        base["rr_games"] = f"{rr.get('rr_games', 0.0):.0f}" if rr else ""
        base["rr_pairs"] = f"{rr.get('rr_pairs', 0.0):.0f}" if rr else ""

        env = env_stats.get(base.get("deck_sig", "").strip(), {})
        base["env_score"] = f"{float(env.get('env_score', 0.0)):.1f}" if env else ""
        base["env_weight"] = f"{float(env.get('env_weight', 0.0)):.4f}" if env else ""
        base["env_games"] = f"{float(env.get('env_games', 0.0)):.0f}" if env else ""
        base["env_score_band"] = str(env.get("env_score_band", "")) if env else ""
        base["env_team_name"] = str(env.get("env_team_name", "")) if env else ""
        base["env_sources"] = str(env.get("env_sources", "")) if env else ""

        key = (base["policy_path"], base["deck_path"])
        if key in seen:
            continue
        seen.add(key)
        out.append(base)
    return out


def row_random_wr(row: dict[str, str]) -> float:
    return as_float(row, "random_win_rate", -1.0)


def row_timeouts(row: dict[str, str]) -> int:
    return as_int(row, "random_timeouts", 0)


def row_rr(row: dict[str, str]) -> float:
    return as_float(row, "rr_mean_wr", -1.0)


def row_env_score(row: dict[str, str]) -> float:
    return as_float(row, "env_score", 0.0)


def row_env_weight(row: dict[str, str]) -> float:
    return as_float(row, "env_weight", 0.0)


def usable_eval(row: dict[str, str]) -> bool:
    return bool(row.get("eval_entry") and row.get("policy_path") and row.get("deck_path"))


def compute_quality_scores(rows: list[dict[str, str]]) -> None:
    max_weight = max((row_env_weight(r) for r in rows), default=0.0)
    max_weight_log = math.log1p(max_weight) if max_weight > 0 else 1.0
    for row in rows:
        random_wr = max(row_random_wr(row), 0.0)
        rr_wr = row_rr(row)
        rr_component = rr_wr if rr_wr >= 0 else 0.0
        env_component = max(0.0, min(1.0, (row_env_score(row) - 600.0) / 600.0))
        weight_component = math.log1p(row_env_weight(row)) / max_weight_log if max_weight > 0 else 0.0
        penalty = 0.0
        if row_random_wr(row) < 0:
            penalty += 0.20
        penalty += min(row_timeouts(row), 5) * 0.03
        if not usable_eval(row):
            penalty += 0.50
        score = 0.45 * random_wr + 0.25 * rr_component + 0.20 * env_component + 0.10 * weight_component - penalty
        row["quality_score"] = f"{score:.6f}"


def gate_reason(
    row: dict[str, str],
    *,
    min_random_wr: float,
    max_timeouts: int,
    min_rr_mean: float,
    score_floor: float,
    env_weight_floor: float,
) -> tuple[bool, str]:
    reasons = []
    if not usable_eval(row):
        reasons.append("missing_eval_entry")
    if row_random_wr(row) < 0:
        reasons.append("missing_random_audit")
    elif row_random_wr(row) < min_random_wr:
        reasons.append(f"random<{min_random_wr:.2f}")
    if row_random_wr(row) >= 0 and row_timeouts(row) > max_timeouts:
        reasons.append(f"timeouts>{max_timeouts}")

    rr_ok = row_rr(row) >= min_rr_mean
    env_ok = row_env_score(row) >= score_floor or row_env_weight(row) >= env_weight_floor
    if not (rr_ok or env_ok):
        reasons.append(f"rr<{min_rr_mean:.2f}_and_env_below_floor")

    if reasons:
        return False, ";".join(reasons)
    why = [f"random>={min_random_wr:.2f}"]
    why.append(f"rr>={min_rr_mean:.2f}" if rr_ok else "live_env_supported")
    return True, "+".join(why)


def select_pool(
    rows: list[dict[str, str]],
    *,
    per_archetype: int,
    per_deck_sig: int,
    reason_key: str,
    min_random_wr: float,
    max_timeouts: int,
    min_rr_mean: float,
    score_floor: float,
    env_weight_floor: float,
    fallback_per_missing_archetype: int = 0,
) -> list[dict[str, str]]:
    selected = []
    rejected_by_arch: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_arch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_arch[row.get("archetype", "")].append(row)

    for arch, arch_rows in sorted(by_arch.items(), key=lambda kv: (-sum(row_env_weight(r) for r in kv[1]), kv[0])):
        ranked = sorted(arch_rows, key=lambda r: as_float(r, "quality_score", 0.0), reverse=True)
        sig_count: Counter[str] = Counter()
        kept_for_arch = 0
        for row in ranked:
            ok, reason = gate_reason(
                row,
                min_random_wr=min_random_wr,
                max_timeouts=max_timeouts,
                min_rr_mean=min_rr_mean,
                score_floor=score_floor,
                env_weight_floor=env_weight_floor,
            )
            if not ok:
                rejected_by_arch[arch].append(row)
                continue
            sig = row.get("deck_sig", "")
            if sig_count[sig] >= per_deck_sig:
                continue
            out = dict(row)
            out["pool_reason"] = f"{reason_key}:{reason}"
            selected.append(out)
            sig_count[sig] += 1
            kept_for_arch += 1
            if kept_for_arch >= per_archetype:
                break

        if kept_for_arch == 0 and fallback_per_missing_archetype > 0:
            fallback_rows = [r for r in ranked if usable_eval(r) and row_timeouts(r) <= max_timeouts]
            for row in fallback_rows[:fallback_per_missing_archetype]:
                out = dict(row)
                out["pool_reason"] = f"{reason_key}:forced_arch_fallback"
                selected.append(out)

    return selected


def select_environment_pool(
    rows: list[dict[str, str]],
    *,
    per_archetype: int,
    per_deck_sig: int,
    min_random_wr: float,
    min_env_score: float,
) -> list[dict[str, str]]:
    selected = []
    by_arch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if usable_eval(row) and row_random_wr(row) >= min_random_wr and row_env_score(row) >= min_env_score:
            by_arch[row.get("archetype", "")].append(row)
    for arch, arch_rows in sorted(by_arch.items(), key=lambda kv: (-sum(row_env_weight(r) for r in kv[1]), kv[0])):
        ranked = sorted(
            arch_rows,
            key=lambda r: (row_env_score(r), row_env_weight(r), as_float(r, "quality_score", 0.0)),
            reverse=True,
        )
        sig_count: Counter[str] = Counter()
        kept = 0
        for row in ranked:
            sig = row.get("deck_sig", "")
            if sig_count[sig] >= per_deck_sig:
                continue
            out = dict(row)
            out["pool_reason"] = f"environment:score>={min_env_score:.0f}+random>={min_random_wr:.2f}"
            selected.append(out)
            sig_count[sig] += 1
            kept += 1
            if kept >= per_archetype:
                break
    return selected


def rejected_rows(
    rows: list[dict[str, str]],
    kept_rows: list[dict[str, str]],
    *,
    min_random_wr: float,
    max_timeouts: int,
    min_rr_mean: float,
    score_floor: float,
    env_weight_floor: float,
) -> list[dict[str, str]]:
    kept = {(r.get("policy_path", ""), r.get("deck_path", "")) for r in kept_rows}
    out = []
    for row in rows:
        key = (row.get("policy_path", ""), row.get("deck_path", ""))
        if key in kept:
            continue
        ok, reason = gate_reason(
            row,
            min_random_wr=min_random_wr,
            max_timeouts=max_timeouts,
            min_rr_mean=min_rr_mean,
            score_floor=score_floor,
            env_weight_floor=env_weight_floor,
        )
        if ok:
            reason = "passed_gate_but_removed_by_cap"
        new = dict(row)
        new["pool_reason"] = f"rejected:{reason}"
        out.append(new)
    return sorted(out, key=lambda r: as_float(r, "quality_score", 0.0), reverse=True)


def output_fields(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
        "name",
        "archetype",
        "variant",
        "deck_rank",
        "team_name",
        "deck_sig",
        "random_games",
        "random_wins",
        "random_win_rate",
        "random_timeouts",
        "rr_mean_wr",
        "rr_games",
        "rr_pairs",
        "env_score",
        "env_score_band",
        "env_weight",
        "env_games",
        "env_team_name",
        "env_sources",
        "quality_score",
        "pool_reason",
        "checkpoint_path",
        "policy_path",
        "deck_path",
        "eval_entry",
        "weight",
        "kept",
        "avg_score",
        "source_plan",
        "status",
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


def stats_lines(name: str, rows: list[dict[str, str]]) -> list[str]:
    by_arch = Counter(row.get("archetype", "") for row in rows)
    random_wrs = [row_random_wr(r) for r in rows if row_random_wr(r) >= 0]
    rr_wrs = [row_rr(r) for r in rows if row_rr(r) >= 0]
    lines = [
        f"{name}: n={len(rows)} archetypes={len(by_arch)} "
        f"random_mean={sum(random_wrs) / max(len(random_wrs), 1):.3f} "
        f"random_min={min(random_wrs) if random_wrs else 0.0:.3f} "
        f"rr_mean={sum(rr_wrs) / max(len(rr_wrs), 1):.3f} "
        f"rr_min={min(rr_wrs) if rr_wrs else 0.0:.3f} "
        f"timeouts={sum(row_timeouts(r) for r in rows if row_timeouts(r) < 999)}",
    ]
    for arch, count in sorted(by_arch.items(), key=lambda kv: (-kv[1], kv[0])):
        vals = [row_random_wr(r) for r in rows if r.get("archetype", "") == arch and row_random_wr(r) >= 0]
        rr_vals = [row_rr(r) for r in rows if r.get("archetype", "") == arch and row_rr(r) >= 0]
        lines.append(
            f"  {arch:<24} n={count:2d} "
            f"rand={sum(vals) / max(len(vals), 1):.3f} "
            f"rr={sum(rr_vals) / max(len(rr_vals), 1):.3f} "
            f"env_max={max((row_env_score(r) for r in rows if r.get('archetype', '') == arch), default=0.0):.1f}"
        )
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="candidate manifest with eval_entry/checkpoint_path/deck_path")
    p.add_argument("--random", required=True, help="random audit CSV")
    p.add_argument("--rr", default="", help="optional RR CSV used to remove weak local opponents")
    p.add_argument("--env-manifest", action="append", default=[], help="live ladder pool manifest; repeatable")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-random-wr", type=float, default=0.97)
    p.add_argument("--balanced-min-random-wr", type=float, default=0.95)
    p.add_argument("--env-min-random-wr", type=float, default=0.90)
    p.add_argument("--max-timeouts", type=int, default=0)
    p.add_argument("--min-rr-mean", type=float, default=0.38)
    p.add_argument("--balanced-min-rr-mean", type=float, default=0.34)
    p.add_argument("--score-floor", type=float, default=900.0)
    p.add_argument("--balanced-score-floor", type=float, default=850.0)
    p.add_argument("--env-min-score", type=float, default=850.0)
    p.add_argument("--env-weight-floor", type=float, default=500.0)
    p.add_argument("--primary-per-archetype", type=int, default=3)
    p.add_argument("--balanced-per-archetype", type=int, default=3)
    p.add_argument("--environment-per-archetype", type=int, default=3)
    p.add_argument("--per-deck-sig", type=int, default=1)
    p.add_argument(
        "--balanced-fallback-per-missing-archetype",
        type=int,
        default=0,
        help=(
            "force weak best-of-archetype rows into balanced coverage; default 0 "
            "keeps balanced suitable for headline scoring"
        ),
    )
    args = p.parse_args()

    manifest_rows = read_csv(args.manifest)
    random_rows = read_csv(args.random)
    rr_stats = aggregate_rr(args.rr)
    env_stats = aggregate_env(args.env_manifest)
    rows = enrich_rows(manifest_rows, random_rows, rr_stats, env_stats)
    compute_quality_scores(rows)
    rows = sorted(rows, key=lambda r: as_float(r, "quality_score", 0.0), reverse=True)

    primary = select_pool(
        rows,
        per_archetype=args.primary_per_archetype,
        per_deck_sig=args.per_deck_sig,
        reason_key="primary",
        min_random_wr=args.min_random_wr,
        max_timeouts=args.max_timeouts,
        min_rr_mean=args.min_rr_mean,
        score_floor=args.score_floor,
        env_weight_floor=args.env_weight_floor,
    )
    balanced = select_pool(
        rows,
        per_archetype=args.balanced_per_archetype,
        per_deck_sig=args.per_deck_sig,
        reason_key="balanced",
        min_random_wr=args.balanced_min_random_wr,
        max_timeouts=args.max_timeouts,
        min_rr_mean=args.balanced_min_rr_mean,
        score_floor=args.balanced_score_floor,
        env_weight_floor=args.env_weight_floor,
        fallback_per_missing_archetype=args.balanced_fallback_per_missing_archetype,
    )
    environment = select_environment_pool(
        rows,
        per_archetype=args.environment_per_archetype,
        per_deck_sig=args.per_deck_sig,
        min_random_wr=args.env_min_random_wr,
        min_env_score=args.env_min_score,
    )
    rejected = rejected_rows(
        rows,
        primary,
        min_random_wr=args.min_random_wr,
        max_timeouts=args.max_timeouts,
        min_rr_mean=args.min_rr_mean,
        score_floor=args.score_floor,
        env_weight_floor=args.env_weight_floor,
    )
    diagnostic = [dict(r, pool_reason=f"diagnostic:{r.get('pool_reason', '')}") for r in rejected]

    out_dir = Path(args.out_dir)
    all_rows = rows + primary + balanced + environment + rejected
    fields = output_fields(all_rows)
    outputs = {
        "rr_pool_all_enriched.csv": rows,
        "rr_pool_primary.csv": primary,
        "rr_pool_balanced.csv": balanced,
        "rr_pool_environment.csv": environment,
        "rr_pool_diagnostic_low_quality.csv": diagnostic,
        "rr_pool_rejected.csv": rejected,
    }
    for filename, pool_rows in outputs.items():
        write_csv(out_dir / filename, pool_rows, fields)

    lines = [
        "RR pool filter summary",
        f"manifest={args.manifest}",
        f"random={args.random}",
        f"rr={args.rr or '<none>'}",
        f"env_manifests={','.join(args.env_manifest) if args.env_manifest else '<none>'}",
        (
            "primary gate: "
            f"random>={args.min_random_wr:.2f}, timeouts<={args.max_timeouts}, "
            f"rr>={args.min_rr_mean:.2f} OR env_score>={args.score_floor:.0f} "
            f"OR env_weight>={args.env_weight_floor:.0f}"
        ),
        "",
    ]
    for filename, pool_rows in outputs.items():
        lines.extend(stats_lines(filename, pool_rows))
        lines.append("")
    summary = out_dir / "rr_pool_filter_summary.txt"
    summary.write_text("\n".join(lines))
    print(summary.read_text())
    for filename in outputs:
        print(f"Wrote {out_dir / filename}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
