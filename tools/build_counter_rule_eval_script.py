#!/usr/bin/env python3
"""Generate pairwise RR commands comparing plain BC to counter_plan rules."""
from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
from pathlib import Path


ARCHETYPES = (
    "Marnie Grimmsnarl",
    "Teal Mask Ogerpon",
    "Mega Lucario",
    "Mega Lopunny",
    "Alakazam",
    "Dragapult",
    "Festival Lead",
    "Crustle Wall",
    "Cynthia Garchomp",
    "Team Rocket Mewtwo",
    "Mega Starmie",
)


def clean_name(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "entry"


def clean_arch(name: str) -> str:
    return str(name or "").strip().replace(" ", "_")


def infer_arch(row: dict[str, str]) -> str:
    for key in ("archetype", "candidate_archetype", "deck_archetype", "opponent_archetype"):
        if row.get(key):
            return str(row[key]).strip().replace("_", " ")
    hay = " ".join(str(row.get(k, "")) for k in ("name", "team_name", "shadow_name", "deck_path", "policy_path", "eval_entry")).lower()
    for arch in ARCHETYPES:
        token = arch.lower().replace(" ", "_")
        if arch.lower() in hay or token in hay:
            return arch
    aliases = {
        "ogerpon": "Teal Mask Ogerpon",
        "crustle": "Crustle Wall",
        "marnie": "Marnie Grimmsnarl",
        "trmewtwo": "Team Rocket Mewtwo",
        "team_rocket_mewtwo": "Team Rocket Mewtwo",
        "lucario": "Mega Lucario",
        "lopunny": "Mega Lopunny",
        "cynthia": "Cynthia Garchomp",
        "festival": "Festival Lead",
        "starmie": "Mega Starmie",
    }
    for token, arch in aliases.items():
        if token in hay:
            return arch
    return ""


def parse_eval_entry(entry: str) -> tuple[str, str, str]:
    entry = str(entry or "").strip()
    if not entry:
        return "", "", ""
    if "=" in entry:
        name, rest = entry.split("=", 1)
    else:
        rest = entry
        name = Path(rest.split(":", 1)[0]).stem
    if ":" in rest:
        policy, deck = rest.split(":", 1)
    else:
        policy, deck = rest, ""
    return clean_name(name), policy.strip(), deck.strip()


def row_entry(row: dict[str, str]) -> tuple[str, str, str]:
    name, policy, deck = parse_eval_entry(row.get("eval_entry", "") or row.get("entry", ""))
    if not policy:
        policy = (row.get("policy_path") or row.get("checkpoint_path") or row.get("policy") or "").strip()
    if not deck:
        deck = (row.get("deck_path") or row.get("deck") or "").strip()
    if not name:
        name = clean_name(row.get("name") or row.get("shadow_name") or row.get("team_name") or Path(policy).stem)
    return name, policy, deck


def load_candidates(paths: list[str]) -> dict[str, list[tuple[str, str, str, str]]]:
    out: dict[str, list[tuple[str, str, str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                arch = infer_arch(row)
                name, policy, deck = row_entry(row)
                if not arch or not policy or not deck:
                    continue
                key = (arch, policy, deck)
                if key in seen:
                    continue
                seen.add(key)
                out.setdefault(arch, []).append((name, policy, deck, path))
    return out


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        if row.get(name):
            return str(row[name]).strip().replace("_", " ")
    return ""


def load_weak_pairs(path: str, limit: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            arch = first_present(row, ("cand_arch", "archetype", "candidate_archetype", "row_archetype", "row"))
            opp = first_present(row, ("opp_arch", "opponent_archetype", "target_archetype", "column_archetype", "column", "opponent"))
            if not arch or not opp:
                continue
            key = (arch, opp)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            if limit and len(pairs) >= limit:
                break
    return pairs


def q(value: str) -> str:
    return shlex.quote(str(value))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", action="append", required=True)
    p.add_argument("--weak-pairs-csv", required=True)
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--per-archetype-index", type=int, default=0)
    p.add_argument("--games", type=int, default=80)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--rule-mode", default="counter_plan")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-sh", required=True)
    p.add_argument("--summary-csv", default="")
    args = p.parse_args()

    candidates = load_candidates(args.manifest)
    pairs = load_weak_pairs(args.weak_pairs_csv, args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.summary_csv or str(out_dir / "counter_rule_summary.csv")
    planned_csv = out_dir / "counter_rule_plan.csv"

    commands: list[str] = []
    planned_rows: list[dict[str, str]] = []
    for i, (arch, opp_arch) in enumerate(pairs, 1):
        arch_entries = candidates.get(arch, [])
        opp_entries = candidates.get(opp_arch, [])
        if len(arch_entries) <= args.per_archetype_index or len(opp_entries) <= args.per_archetype_index:
            planned_rows.append({
                "idx": str(i),
                "archetype": arch,
                "opponent_archetype": opp_arch,
                "status": "missing_candidate",
                "candidate_count": str(len(arch_entries)),
                "opponent_count": str(len(opp_entries)),
            })
            continue
        cand_name, cand_policy, cand_deck, cand_manifest = arch_entries[args.per_archetype_index]
        opp_name, opp_policy, opp_deck, opp_manifest = opp_entries[args.per_archetype_index]
        stem = f"{i:02d}_{clean_name(arch)}_vs_{clean_name(opp_arch)}"
        base = f"base_{stem}"
        rule = f"rule_{stem}"
        opp = f"opp_{stem}"
        csv_path = out_dir / f"{stem}_counter_rule_g{args.games}.csv"
        log_path = out_dir / f"{stem}_counter_rule_g{args.games}.log"
        cmd = " ".join([
            "python3", "tools/eval_round_robin.py",
            "--entry", q(f"{base}={cand_policy}:{cand_deck}"),
            "--entry", q(f"{rule}={cand_policy}:{cand_deck}"),
            "--entry", q(f"{opp}={opp_policy}:{opp_deck}"),
            "--rules-entry", q(f"{rule}={args.rule_mode}"),
            "--games", str(args.games),
            "--workers", str(args.workers),
            "--max-turns", str(args.max_turns),
            "--progress-every", str(args.progress_every),
            "--out-csv", q(str(csv_path)),
            "2>&1", "|", "tee", q(str(log_path)),
        ])
        commands.append(cmd)
        planned_rows.append({
            "idx": str(i),
            "archetype": arch,
            "opponent_archetype": opp_arch,
            "status": "planned",
            "candidate_name": cand_name,
            "candidate_policy": cand_policy,
            "candidate_deck": cand_deck,
            "candidate_manifest": cand_manifest,
            "opponent_name": opp_name,
            "opponent_policy": opp_policy,
            "opponent_deck": opp_deck,
            "opponent_manifest": opp_manifest,
            "csv_path": str(csv_path),
        })

    fieldnames = sorted({k for row in planned_rows for k in row})
    with planned_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(planned_rows)

    script = Path(args.out_sh)
    script.parent.mkdir(parents=True, exist_ok=True)
    with script.open("w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n")
        f.write(f"mkdir -p {q(str(out_dir))}\n")
        for cmd in commands:
            f.write(cmd + "\n")
        f.write(" ".join([
            "python3", "tools/summarize_counter_rule_eval.py",
            "--plan-csv", q(str(planned_csv)),
            "--out-csv", q(summary_csv),
        ]) + "\n")
    os.chmod(script, 0o755)
    print(f"wrote {script} commands={len(commands)} plan={planned_csv} summary={summary_csv}", flush=True)


if __name__ == "__main__":
    main()
