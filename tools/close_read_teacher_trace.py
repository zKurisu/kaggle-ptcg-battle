#!/usr/bin/env python3
"""Create a close-read timeline from a teacher trace markdown file.

The goal is not to replace manual review.  It turns the verbose trace into a
decision-by-decision scaffold so the reviewer can read every turn without
losing the route.
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path


DEC_RE = re.compile(r"^### Decision (?P<idx>\d+) step=(?P<step>-?\d+) turn=(?P<turn>-?\d+) tac=(?P<tac>-?\d+)")
KV_RE = re.compile(r"^- step=(?P<step>-?\d+) turn=(?P<turn>-?\d+) tac=(?P<tac>-?\d+) context=(?P<context>[^ ]+) select_type=(?P<select_type>[^ ]+) min/max=(?P<mn>\d+)/(?P<mx>\d+) options=(?P<options>\d+)")


@dataclass
class Decision:
    idx: int
    step: int
    turn: int
    tac: int
    context: str = ""
    select_type: str = ""
    min_max: str = ""
    options: str = ""
    board: str = ""
    chosen: str = ""
    target: str = ""
    flags: str = ""
    candidate_options: str = ""
    public_summary: str = ""
    public_logs: list[str] = field(default_factory=list)


def _strip_prefix(line: str, prefix: str) -> str:
    s = line.strip()
    if s.startswith(prefix):
        return s[len(prefix):].strip()
    return ""


def parse_trace(path: Path) -> tuple[dict[str, str], list[Decision]]:
    meta: dict[str, str] = {}
    decisions: list[Decision] = []
    current: Decision | None = None
    in_public_logs = False

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip("\n")
        if line.startswith("date:") or line.startswith("episode_id:") or line.startswith("team_name:") or line.startswith("deck_sig:") or line.startswith("score:") or line.startswith("opponent_team_name:") or line.startswith("opponent_deck_sig:") or line.startswith("opponent_score:") or line.startswith("won:") or line.startswith("steps:"):
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip("`")
            continue

        m = DEC_RE.match(line)
        if m:
            if current is not None:
                decisions.append(current)
            current = Decision(
                idx=int(m.group("idx")),
                step=int(m.group("step")),
                turn=int(m.group("turn")),
                tac=int(m.group("tac")),
            )
            in_public_logs = False
            continue
        if current is None:
            continue

        s = line.strip()
        if s.startswith("- step="):
            km = KV_RE.match(s)
            if km:
                current.context = km.group("context")
                current.select_type = km.group("select_type")
                current.min_max = f"{km.group('mn')}/{km.group('mx')}"
                current.options = km.group("options")
            continue
        if s.startswith("board:"):
            current.board = _strip_prefix(s, "board:")
            in_public_logs = False
            continue
        if s.startswith("chosen:"):
            current.chosen = _strip_prefix(s, "chosen:")
            in_public_logs = False
            continue
        if s.startswith("target:"):
            current.target = _strip_prefix(s, "target:")
            in_public_logs = False
            continue
        if s.startswith("flags:"):
            current.flags = _strip_prefix(s, "flags:")
            in_public_logs = False
            continue
        if s.startswith("candidate_options:"):
            current.candidate_options = _strip_prefix(s, "candidate_options:")
            in_public_logs = False
            continue
        if s.startswith("public_logs:"):
            txt = _strip_prefix(s, "public_logs:")
            if txt:
                current.public_summary = txt
            in_public_logs = True
            continue
        if in_public_logs and s.startswith("- "):
            current.public_logs.append(s[2:].strip())

    if current is not None:
        decisions.append(current)
    return meta, decisions


def tag_decision(d: Decision) -> list[str]:
    text = " ".join([d.context, d.chosen, d.target, d.flags, d.public_summary, " ".join(d.public_logs)])
    tags: list[str] = []

    def add(cond: bool, tag: str) -> None:
        if cond and tag not in tags:
            tags.append(tag)

    add("SETUP" in d.context, "setup")
    add("Crispin" in text, "crispin")
    add("Basic {R} Energy" in d.chosen or "Basic {P} Energy" in d.chosen, "rp_energy")
    add("Basic {D} Energy" in d.chosen, "dark_energy")
    add("Dreepy" in d.chosen or "Dreepy" in d.target, "dreepy")
    add("Drakloak" in d.chosen or "Drakloak" in d.target or "Drakloak" in text, "drakloak")
    add("Dragapult ex" in d.chosen or "Dragapult ex" in d.target or "Dragapult ex" in text, "dragapult_ex")
    add("Munkidori" in d.chosen or "Munkidori" in d.target or "Munkidori" in text, "munkidori")
    add("Buddy-Buddy Poffin" in text, "poffin")
    add("Crushing Hammer" in text, "hammer")
    add("Jamming Tower" in text, "jamming")
    add("Boss" in text, "boss")
    add("ATTACK" in d.chosen or "attack card" in text, "attack")
    add("ABILITY" in d.chosen or "ability" in text, "ability")
    add("EVOLVE" in d.chosen or "evolve" in text, "evolve")
    add("RETREAT" in d.chosen or "switch active" in text, "pivot")
    add("Dwebble" in text, "opp_dwebble")
    add("Crustle" in text, "opp_crustle")
    add("Mega Kangaskhan" in text, "opp_kangaskhan")
    add("counter Crustle" in text or "dca" in d.flags.lower(), "dca")
    add("move/unknown deck->prize" in text, "prize_setup")
    add("lost_zone" in text, "lost_zone_choice")
    return tags


def route_note(d: Decision, tags: list[str]) -> str:
    chosen = d.chosen
    target = d.target
    board = d.board
    chosen_text = " ".join([d.context, d.chosen, d.target, d.flags])
    if "setup" in tags:
        return "setup: choose opener/bench; important because Crustle games are decided by early line and engine."
    if "Crispin" in chosen:
        return "energy/search plan: start or continue the R/P line for Phantom Dive; check the follow-up energy picks."
    if "Basic {R} Energy" in chosen or "Basic {P} Energy" in chosen:
        if "Dreepy" in target or "Drakloak" in target or "Dragapult ex" in target:
            return "energy plan: put R/P on the Dragapult line instead of wasting tempo elsewhere."
        return "energy pick: R/P is preserved for Phantom Dive readiness."
    if chosen.startswith("ABILITY Drakloak"):
        return "engine: use Drakloak ability before committing later actions; this is a tempo/search decision."
    if "Buddy-Buddy Poffin" in chosen:
        return "board plan: use Poffin to build multiple Dreepy lines instead of relying on one attacker."
    if "Dreepy" in chosen and d.context in {"TO_BENCH", "SETUP_BENCH"}:
        return "board plan: add backup Dreepy lines; this protects the long game after the first attacker is answered."
    if chosen.startswith("EVOLVE Dragapult ex"):
        return "main attacker online: evolve only after R/P line is ready or close to ready."
    if chosen.startswith("EVOLVE Drakloak"):
        return "engine setup: evolve into Drakloak before using ability/search and before stage-2 commitment."
    if chosen.startswith("RETREAT") or d.context == "SWITCH":
        return "pivot: move charged Dragapult ex active once attack route is online."
    if chosen.startswith("ATTACK") and "Dragapult ex" in board:
        return "pressure: Phantom Dive active target plus counters on bench wall/engine."
    if d.context == "DAMAGE_COUNTER_ANY" and "Crustle" in chosen:
        return "DCA: counters are assigned to Crustle; check whether this is resource-war setup or wasted into wall."
    if "Crushing Hammer" in chosen:
        return "denial: Hammer participates in the Crustle resource-war route."
    if "Jamming Tower" in chosen:
        return "stadium denial: Jamming Tower/Watchtower-style replacement matters vs Crustle engine/stadium."
    if chosen.startswith("ABILITY Munkidori"):
        return "damage transfer: Munkidori converts scattered damage into a long-game win condition."
    if "Boss" in chosen:
        return "gust plan: avoid active wall, or force a vulnerable target when one exists."
    if "lost_zone_choice" in tags:
        return "search tradeoff: note the chosen keep/discard; this can reveal which resources are expendable."
    return ""


def write_outputs(meta: dict[str, str], decisions: list[Decision], out_md: Path, out_csv: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in decisions:
        tags = tag_decision(d)
        rows.append({
            "idx": d.idx,
            "step": d.step,
            "turn": d.turn,
            "tac": d.tac,
            "context": d.context,
            "min_max": d.min_max,
            "options": d.options,
            "chosen": d.chosen,
            "target": d.target,
            "flags": d.flags,
            "tags": "|".join(tags),
            "note": route_note(d, tags),
            "board": d.board,
            "public_logs": " || ".join(d.public_logs[:12]),
        })
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["idx"])
        w.writeheader()
        w.writerows(rows)

    tag_counts: dict[str, int] = {}
    for r in rows:
        for tag in r["tags"].split("|"):
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Close Read Timeline\n\n")
        for key in ("date", "episode_id", "team_name", "deck_sig", "score", "opponent_team_name", "opponent_deck_sig", "opponent_score", "won", "steps"):
            if key in meta:
                f.write(f"- {key}: `{meta[key]}`\n")
        f.write(f"- decisions: `{len(decisions)}`\n\n")
        f.write("## Tag Counts\n\n")
        for tag, count in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"- {tag}: {count}\n")
        f.write("\n## Timeline\n\n")
        f.write("| idx | turn | context | chosen | tags | note |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            chosen = str(r["chosen"]).replace("|", "/")
            note = str(r["note"]).replace("|", "/")
            tags = str(r["tags"]).replace("|", ",")
            f.write(f"| {r['idx']} | {r['turn']} | {r['context']} | {chosen} | {tags} | {note} |\n")
        f.write("\n## Full Decision Notes\n\n")
        for r, d in zip(rows, decisions):
            f.write(f"### Decision {r['idx']} step={r['step']} turn={r['turn']} context={r['context']}\n\n")
            f.write(f"- board: {r['board']}\n")
            f.write(f"- chosen: {r['chosen']}\n")
            if r["target"]:
                f.write(f"- target: {r['target']}\n")
            if r["flags"]:
                f.write(f"- flags: {r['flags']}\n")
            if r["tags"]:
                f.write(f"- tags: {r['tags']}\n")
            if r["note"]:
                f.write(f"- route_note: {r['note']}\n")
            if d.public_logs:
                f.write("- public_logs:\n")
                for log in d.public_logs[:14]:
                    f.write(f"  - {log}\n")
            f.write("\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("trace_md")
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()
    meta, decisions = parse_trace(Path(args.trace_md))
    write_outputs(meta, decisions, Path(args.out_md), Path(args.out_csv))
    print(f"wrote close-read decisions={len(decisions)} md={args.out_md} csv={args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
