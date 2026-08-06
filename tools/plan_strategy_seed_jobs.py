#!/usr/bin/env python3
"""Plan seed-driven matchup trace, rule-probe, and teacher-rollout jobs.

This tool turns human strategy seeds into concrete local validation tasks. It
does not run the simulator. It reads strategy seeds, seed-card mappings, and
candidate/opponent manifests, then emits:

- a task plan CSV,
- a skipped-task CSV,
- a shell script with trace and available rule-probe commands,
- a teacher-spec JSONL file for future narrow teacher policies.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.rule_overlay import RULE_MODES


PLAN_FIELDS = [
    "task_id",
    "seed_id",
    "archetype",
    "opponent_archetype",
    "intervention",
    "teacher_status",
    "validation_status",
    "deck_sig_scope",
    "source_type",
    "source_url",
    "candidate_name",
    "candidate_archetype",
    "candidate_deck_sig",
    "candidate_team_name",
    "candidate_entry",
    "opponent_name",
    "opponent_archetype_entry",
    "opponent_deck_sig",
    "opponent_team_name",
    "opponent_entry",
    "card_check",
    "missing_required_cards",
    "unknown_required_cards",
    "rule_mode",
    "trace_prefix",
    "trace_cmd",
    "gap_cmd",
    "rule_probe_cmd",
    "teacher_spec_id",
    "sim_trigger",
    "desired_bias",
    "notes",
]

SKIPPED_FIELDS = [
    "seed_id",
    "archetype",
    "opponent_archetype",
    "candidate_name",
    "opponent_name",
    "reason",
    "details",
]


DEFAULT_GROUPS = {
    "ex-heavy decks": [
        "Teal Mask Ogerpon",
        "Marnie Grimmsnarl",
        "Dragapult",
        "Cynthia Garchomp",
        "Mega Lopunny",
        "Mega Lucario",
        "Team Rocket Mewtwo",
        "Alakazam",
    ],
    "munkidori decks": [
        "Marnie Grimmsnarl",
    ],
}

DEFAULT_RULE_MODE_BY_SEED = {
    "marnie_vs_ogerpon_setup": "marnie_setup",
}


@dataclass(frozen=True)
class Seed:
    row: dict

    @property
    def id(self) -> str:
        return str(self.row.get("id", "")).strip()

    @property
    def archetype(self) -> str:
        return str(self.row.get("archetype", "")).strip()

    @property
    def opponent_archetype(self) -> str:
        return str(self.row.get("opponent_archetype", "")).strip()

    @property
    def deck_sig_scope(self) -> str:
        return str(self.row.get("deck_sig_scope", "")).strip()

    @property
    def intervention(self) -> str:
        return str(self.row.get("intervention", "")).strip()


@dataclass(frozen=True)
class SeedCard:
    seed_id: str
    card_name: str
    card_id_hint: int
    role: str
    required_in: str
    notes: str


@dataclass
class ManifestEntry:
    name: str
    archetype: str
    team_name: str
    deck_sig: str
    policy_path: str
    deck_path: str
    eval_entry: str
    source_path: str
    manifest_index: int
    weight: float


def slugify(text: str, limit: int = 80) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", text.strip().lower()).strip("_")
    return (text[:limit].strip("_") or "item")


def clean_entry_name(value: str) -> str:
    return slugify(value, limit=80)


def split_values(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[;,]", value)
    return [p.strip() for p in parts if p.strip()]


def normalize_arch(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def parse_groups(specs: list[str]) -> dict[str, list[str]]:
    groups = {k: list(v) for k, v in DEFAULT_GROUPS.items()}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--group must be NAME=ARCH1,ARCH2: {spec}")
        name, values = spec.split("=", 1)
        groups[normalize_arch(name)] = split_values(values)
    return {normalize_arch(k): v for k, v in groups.items()}


def read_csv_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader, 2):
            if row.get(None):
                raise ValueError(f"malformed CSV row {i} in {path}: extra fields {row[None]}")
            rows.append(row)
        return rows


def read_seeds(path: str) -> list[Seed]:
    return [Seed(row) for row in read_csv_rows(path) if str(row.get("id", "")).strip()]


def read_seed_cards(path: str) -> dict[str, list[SeedCard]]:
    out: dict[str, list[SeedCard]] = {}
    for row in read_csv_rows(path):
        seed_id = str(row.get("seed_id", "")).strip()
        if not seed_id:
            continue
        try:
            card_id = int(str(row.get("card_id_hint", "") or "0"))
        except ValueError:
            card_id = 0
        out.setdefault(seed_id, []).append(
            SeedCard(
                seed_id=seed_id,
                card_name=str(row.get("card_name", "")).strip(),
                card_id_hint=card_id,
                role=str(row.get("role", "")).strip(),
                required_in=str(row.get("required_in", "")).strip(),
                notes=str(row.get("notes", "")).strip(),
            )
        )
    return out


def weight_value(row: dict) -> float:
    for key in ("weight", "trajectory_score", "random_win_rate", "avg_score", "max_score", "decisions"):
        try:
            value = str(row.get(key, "")).strip()
            if value:
                return float(value)
        except ValueError:
            continue
    try:
        rank = int(str(row.get("rank", "") or "0"))
        return 1.0 / max(rank, 1)
    except ValueError:
        return 1.0


def manifest_entry_from_row(path: str, row: dict, index: int, used_names: dict[str, int]) -> ManifestEntry | None:
    name = (
        row.get("name")
        or row.get("shadow_name")
        or row.get("team_name")
        or row.get("deck_sig")
        or ""
    )
    name = clean_entry_name(name)
    if not name:
        return None
    count = used_names.get(name, 0) + 1
    used_names[name] = count
    unique_name = name if count == 1 else f"{name}_{count}"

    policy = (row.get("policy_path") or row.get("checkpoint_path") or row.get("policy") or "").strip()
    deck = (row.get("deck_path") or row.get("deck") or "").strip()
    eval_entry = (row.get("eval_entry") or row.get("entry") or "").strip()
    if eval_entry:
        if unique_name != name and "=" in eval_entry:
            eval_entry = f"{unique_name}={eval_entry.split('=', 1)[1]}"
    elif policy and deck:
        eval_entry = f"{unique_name}={policy}:{deck}"
    elif policy:
        eval_entry = f"{unique_name}={policy}"
    elif deck:
        eval_entry = f"{unique_name}=random:{deck}"
    else:
        return None

    return ManifestEntry(
        name=unique_name,
        archetype=str(row.get("archetype", "")).strip(),
        team_name=str(row.get("team_name", "")).strip(),
        deck_sig=str(row.get("deck_sig", "")).strip(),
        policy_path=policy,
        deck_path=deck,
        eval_entry=eval_entry,
        source_path=path,
        manifest_index=index,
        weight=weight_value(row),
    )


def read_manifest_entries(paths: list[str], *, sort_by: str) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    used_names: dict[str, int] = {}
    for path in paths:
        for idx, row in enumerate(read_csv_rows(path), 1):
            entry = manifest_entry_from_row(path, row, idx, used_names)
            if entry is not None:
                entries.append(entry)
    if sort_by == "weight":
        entries.sort(key=lambda e: (-e.weight, e.source_path, e.manifest_index))
    return entries


def sig_scope_matches(seed: Seed, entry: ManifestEntry) -> bool:
    scope = normalize_arch(seed.deck_sig_scope)
    if not scope or scope in ("all", "*"):
        return True
    sigs = {x for x in split_values(seed.deck_sig_scope)}
    return not sigs or entry.deck_sig in sigs


def seed_opponent_arches(seed: Seed, groups: dict[str, list[str]], *, expand_all: bool,
                         all_archetypes: list[str]) -> list[str]:
    raw = seed.opponent_archetype
    if normalize_arch(raw) in ("all", "*"):
        return [a for a in all_archetypes if normalize_arch(a) != normalize_arch(seed.archetype)] if expand_all else []
    out: list[str] = []
    for item in split_values(raw):
        key = normalize_arch(item)
        if key in groups:
            out.extend(groups[key])
        else:
            out.append(item)
    seen: set[str] = set()
    deduped: list[str] = []
    for arch in out:
        key = normalize_arch(arch)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(arch)
    return deduped


def archetype_matches(value: str, target: str) -> bool:
    return normalize_arch(value) == normalize_arch(target)


def deck_card_ids(path: str) -> set[int] | None:
    if not path or not os.path.exists(path):
        return None
    cards: set[int] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cards.add(int(line))
            except ValueError:
                continue
    return cards


def validate_cards(cards: list[SeedCard], candidate: ManifestEntry, opponent: ManifestEntry) -> tuple[str, str, str]:
    own_deck = deck_card_ids(candidate.deck_path)
    opp_deck = deck_card_ids(opponent.deck_path)
    missing: list[str] = []
    unknown: list[str] = []

    for card in cards:
        if card.card_id_hint <= 0:
            continue
        required = normalize_arch(card.required_in)
        if required == "own_deck":
            deck = own_deck
            label = "own"
        elif required == "opponent_deck":
            deck = opp_deck
            label = "opp"
        else:
            continue
        card_label = f"{label}:{card.card_name}({card.card_id_hint})"
        if deck is None:
            unknown.append(card_label)
        elif card.card_id_hint not in deck:
            missing.append(card_label)

    if missing:
        status = "missing"
    elif unknown:
        status = "unknown"
    else:
        status = "ok"
    return status, ";".join(missing), ";".join(unknown)


def entry_rest(eval_entry: str) -> str:
    return eval_entry.split("=", 1)[1] if "=" in eval_entry else eval_entry


def shell_cmd(parts: list[str]) -> str:
    return shlex.join([str(p) for p in parts if str(p) != ""])


def rule_modes(args: argparse.Namespace) -> dict[str, str]:
    out = dict(DEFAULT_RULE_MODE_BY_SEED)
    for spec in args.rule_mode:
        if "=" not in spec:
            raise ValueError("--rule-mode must be SEED_ID=MODE")
        seed_id, mode = spec.split("=", 1)
        mode = mode.strip()
        if mode not in RULE_MODES:
            raise ValueError(f"unknown rule mode {mode}; expected one of {', '.join(RULE_MODES)}")
        out[seed_id.strip()] = mode
    return out


def teacher_status(seed: Seed, rule_mode: str) -> str:
    intervention = normalize_arch(seed.intervention)
    if intervention == "teacher_rollout":
        return "needs_teacher_policy"
    if intervention == "rerank_guard":
        return "rule_probe_available" if rule_mode else "needs_rule_implementation"
    if intervention == "matchup_bc":
        return "trace_then_matchup_bc"
    if intervention == "deck_sig_shadow":
        return "trace_then_shadow_or_specialist"
    if intervention == "do_not_train":
        return "record_only"
    return "trace_first"


def teacher_spec(task_id: str, seed: Seed, cards: list[SeedCard],
                 candidate: ManifestEntry, opponent: ManifestEntry, status: str) -> dict:
    return {
        "teacher_spec_id": f"teacher_{task_id}",
        "task_id": task_id,
        "seed_id": seed.id,
        "status": status,
        "archetype": seed.archetype,
        "opponent_archetype": seed.opponent_archetype,
        "candidate": {
            "name": candidate.name,
            "archetype": candidate.archetype,
            "deck_sig": candidate.deck_sig,
            "team_name": candidate.team_name,
            "entry": candidate.eval_entry,
        },
        "opponent": {
            "name": opponent.name,
            "archetype": opponent.archetype,
            "deck_sig": opponent.deck_sig,
            "team_name": opponent.team_name,
            "entry": opponent.eval_entry,
        },
        "source": {
            "type": seed.row.get("source_type", ""),
            "url": seed.row.get("source_url", ""),
            "claim": seed.row.get("source_claim", ""),
        },
        "strategy": {
            "sim_trigger": seed.row.get("sim_trigger", ""),
            "desired_bias": seed.row.get("desired_bias", ""),
            "intervention": seed.intervention,
        },
        "target_cards": [
            {
                "name": c.card_name,
                "card_id_hint": c.card_id_hint,
                "role": c.role,
                "required_in": c.required_in,
                "notes": c.notes,
            }
            for c in cards
        ],
    }


def make_trace_cmd(args: argparse.Namespace, candidate: ManifestEntry, opponent: ManifestEntry, prefix: str) -> str:
    return shell_cmd([
        args.python,
        "tools/trace_matchup_decisions.py",
        "--candidate",
        candidate.eval_entry,
        "--opponent",
        opponent.eval_entry,
        "--games",
        args.games,
        "--seed",
        args.seed,
        "--max-turns",
        args.max_turns,
        "--progress-every",
        args.progress_every,
        "--out-prefix",
        prefix,
    ])


def make_gap_cmd(args: argparse.Namespace, prefix: str) -> str:
    return shell_cmd([
        args.python,
        "tools/trace_outcome_gap_report.py",
        f"{prefix}.decisions.csv",
        "--out-csv",
        f"{prefix}.gap.csv",
        "--top",
        args.gap_top,
    ])


def make_rule_probe_cmd(args: argparse.Namespace, candidate: ManifestEntry, opponent: ManifestEntry,
                        prefix: str, rule_mode: str) -> str:
    if not rule_mode:
        return ""
    plain = f"{slugify(candidate.name, 45)}_plain={entry_rest(candidate.eval_entry)}"
    ruled_name = f"{slugify(candidate.name, 45)}_{rule_mode}"
    ruled = f"{ruled_name}={entry_rest(candidate.eval_entry)}"
    opp = f"{slugify(opponent.name, 55)}={entry_rest(opponent.eval_entry)}"
    return shell_cmd([
        args.python,
        "tools/eval_round_robin.py",
        "--entry",
        plain,
        "--entry",
        ruled,
        "--entry",
        opp,
        "--rules-entry",
        f"{ruled_name}={rule_mode}",
        "--games",
        args.rule_games,
        "--workers",
        args.workers,
        "--max-turns",
        args.max_turns,
        "--progress-every",
        args.progress_every,
        "--skip-bad-entries",
        "--out-csv",
        f"{prefix}.rule_probe.csv",
    ])


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_shell(path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(Path(args.out_dir) / 'trace'))}",
        "",
    ]
    for row in rows:
        lines.extend([
            f"echo '[trace] {row['task_id']} {row['seed_id']}'",
            row["trace_cmd"],
            row["gap_cmd"],
            "",
        ])
        if row.get("rule_probe_cmd"):
            lines.extend([
                f"echo '[rule_probe] {row['task_id']} {row['rule_mode']}'",
                row["rule_probe_cmd"],
                "",
            ])
    if rows:
        aggregate = shell_cmd([
            args.python,
            "tools/trace_outcome_gap_report.py",
            str(Path(args.out_dir) / "trace" / "*.decisions.csv"),
            "--out-csv",
            str(Path(args.out_dir) / "strategy_seed_gap_report.csv"),
            "--top",
            args.gap_top,
        ])
        lines.extend([
            "echo '[aggregate_gap]'",
            aggregate,
            "",
        ])
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def build_plan(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    seeds = read_seeds(args.seeds)
    seed_cards = read_seed_cards(args.seed_cards)
    selected_seed_ids = {x for x in args.seed_id}
    selected_arches = {normalize_arch(x) for x in args.archetype}
    if selected_seed_ids:
        seeds = [s for s in seeds if s.id in selected_seed_ids]
    if selected_arches:
        seeds = [s for s in seeds if normalize_arch(s.archetype) in selected_arches]

    candidate_entries = read_manifest_entries(args.candidate_manifest, sort_by=args.sort_by)
    opponent_entries = read_manifest_entries(args.opponent_manifest or args.candidate_manifest, sort_by=args.sort_by)
    groups = parse_groups(args.group)
    rule_by_seed = rule_modes(args)
    all_arches = sorted({e.archetype for e in [*candidate_entries, *opponent_entries] if e.archetype})

    rows: list[dict] = []
    skipped: list[dict] = []
    teacher_rows: list[dict] = []

    for seed in seeds:
        candidates = [
            e for e in candidate_entries
            if archetype_matches(e.archetype, seed.archetype) and sig_scope_matches(seed, e)
        ][: args.limit_candidates]
        if not candidates:
            skipped.append({
                "seed_id": seed.id,
                "archetype": seed.archetype,
                "opponent_archetype": seed.opponent_archetype,
                "candidate_name": "",
                "opponent_name": "",
                "reason": "no_candidate",
                "details": f"scope={seed.deck_sig_scope}",
            })
            continue

        opp_arches = seed_opponent_arches(seed, groups, expand_all=args.expand_all, all_archetypes=all_arches)
        if not opp_arches:
            skipped.append({
                "seed_id": seed.id,
                "archetype": seed.archetype,
                "opponent_archetype": seed.opponent_archetype,
                "candidate_name": "",
                "opponent_name": "",
                "reason": "no_expanded_opponent_archetype",
                "details": "use --expand-all or --group NAME=ARCH1,ARCH2",
            })
            continue

        opponents: list[ManifestEntry] = []
        for arch in opp_arches:
            matches = [e for e in opponent_entries if archetype_matches(e.archetype, arch)]
            opponents.extend(matches[: args.limit_opponents])
        if not opponents:
            skipped.append({
                "seed_id": seed.id,
                "archetype": seed.archetype,
                "opponent_archetype": seed.opponent_archetype,
                "candidate_name": "",
                "opponent_name": "",
                "reason": "no_opponent",
                "details": ";".join(opp_arches),
            })
            continue

        for candidate in candidates:
            for opponent in opponents:
                cards = seed_cards.get(seed.id, [])
                card_check, missing_cards, unknown_cards = validate_cards(cards, candidate, opponent)
                if card_check == "missing" and not args.include_missing_required_cards:
                    skipped.append({
                        "seed_id": seed.id,
                        "archetype": seed.archetype,
                        "opponent_archetype": seed.opponent_archetype,
                        "candidate_name": candidate.name,
                        "opponent_name": opponent.name,
                        "reason": "missing_required_cards",
                        "details": missing_cards,
                    })
                    continue
                task_id = slugify(
                    f"{seed.id}_{candidate.name}_vs_{opponent.name}",
                    limit=120,
                )
                prefix = str(Path(args.out_dir) / "trace" / task_id)
                rule_mode = rule_by_seed.get(seed.id, "")
                status = teacher_status(seed, rule_mode)
                spec = teacher_spec(task_id, seed, cards, candidate, opponent, status)
                teacher_rows.append(spec)
                row = {
                    "task_id": task_id,
                    "seed_id": seed.id,
                    "archetype": seed.archetype,
                    "opponent_archetype": seed.opponent_archetype,
                    "intervention": seed.intervention,
                    "teacher_status": status,
                    "validation_status": seed.row.get("validation_status", ""),
                    "deck_sig_scope": seed.deck_sig_scope,
                    "source_type": seed.row.get("source_type", ""),
                    "source_url": seed.row.get("source_url", ""),
                    "candidate_name": candidate.name,
                    "candidate_archetype": candidate.archetype,
                    "candidate_deck_sig": candidate.deck_sig,
                    "candidate_team_name": candidate.team_name,
                    "candidate_entry": candidate.eval_entry,
                    "opponent_name": opponent.name,
                    "opponent_archetype_entry": opponent.archetype,
                    "opponent_deck_sig": opponent.deck_sig,
                    "opponent_team_name": opponent.team_name,
                    "opponent_entry": opponent.eval_entry,
                    "card_check": card_check,
                    "missing_required_cards": missing_cards,
                    "unknown_required_cards": unknown_cards,
                    "rule_mode": rule_mode,
                    "trace_prefix": prefix,
                    "trace_cmd": make_trace_cmd(args, candidate, opponent, prefix),
                    "gap_cmd": make_gap_cmd(args, prefix),
                    "rule_probe_cmd": make_rule_probe_cmd(args, candidate, opponent, prefix, rule_mode),
                    "teacher_spec_id": spec["teacher_spec_id"],
                    "sim_trigger": seed.row.get("sim_trigger", ""),
                    "desired_bias": seed.row.get("desired_bias", ""),
                    "notes": seed.row.get("notes", ""),
                }
                rows.append(row)
    return rows, skipped, teacher_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="data/matchup_strategy_seeds_v1.csv")
    p.add_argument("--seed-cards", default="data/matchup_strategy_seed_cards_v1.csv")
    p.add_argument("--candidate-manifest", action="append", required=True)
    p.add_argument("--opponent-manifest", action="append", default=[],
                   help="defaults to --candidate-manifest when omitted")
    p.add_argument("--out-dir", default="logs/strategy_seed_jobs")
    p.add_argument("--seed-id", action="append", default=[])
    p.add_argument("--archetype", action="append", default=[])
    p.add_argument("--group", action="append", default=[],
                   help="opponent group expansion, e.g. 'wall decks=Crustle Wall'")
    p.add_argument("--expand-all", action="store_true",
                   help="expand opponent_archetype=ALL into all known non-self archetypes")
    p.add_argument("--limit-candidates", type=int, default=2)
    p.add_argument("--limit-opponents", type=int, default=2)
    p.add_argument("--sort-by", choices=["manifest", "weight"], default="manifest")
    p.add_argument("--include-missing-required-cards", action="store_true",
                   help="emit tasks even when local deck validation proves required cards are absent")
    p.add_argument("--rule-mode", action="append", default=[],
                   help="override/add existing rule probe mode, e.g. seed_id=marnie_setup")
    p.add_argument("--python", default="python3")
    p.add_argument("--games", type=int, default=120)
    p.add_argument("--rule-games", type=int, default=80)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gap-top", type=int, default=60)
    args = p.parse_args()

    rows, skipped, teacher_rows = build_plan(args)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "strategy_seed_tasks.csv", PLAN_FIELDS, rows)
    write_csv(out_dir / "strategy_seed_skipped.csv", SKIPPED_FIELDS, skipped)
    write_jsonl(out_dir / "teacher_specs.jsonl", teacher_rows)
    write_shell(out_dir / "run_strategy_seed_traces.sh", rows, args)

    print(
        f"Wrote {out_dir} tasks={len(rows)} skipped={len(skipped)} "
        f"teacher_specs={len(teacher_rows)}",
        flush=True,
    )
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row["teacher_status"]] = by_status.get(row["teacher_status"], 0) + 1
    for status, n in sorted(by_status.items()):
        print(f"  {status}: {n}", flush=True)
    if skipped:
        by_reason: dict[str, int] = {}
        for row in skipped:
            by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
        for reason, n in sorted(by_reason.items()):
            print(f"  skipped {reason}: {n}", flush=True)


if __name__ == "__main__":
    main()
