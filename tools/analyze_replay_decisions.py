#!/usr/bin/env python3
"""Analyze actual submitted-agent decisions inside downloaded replay JSONs."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.deck_plans import get_plan
from tools.bc2_accuracy import OPT_NAMES


FIELDS = [
    "group", "games", "decisions", "wins", "losses", "draws",
    "attack_available", "attack_chosen", "miss_attack_rate",
    "ability_available", "ability_chosen", "miss_ability_rate",
    "attach_available", "attach_chosen", "miss_attach_rate",
    "end_chosen", "early_end_rate",
]


def _opt_type(opt: dict) -> int:
    return int(opt.get("type", 0) or 0)


def _choice_type(sel: dict, action: list) -> int:
    opts = sel.get("option") or []
    if not action:
        return -1
    idx = action[0]
    if isinstance(idx, int) and 0 <= idx < len(opts):
        return _opt_type(opts[idx])
    return -1


def _option_types(sel: dict) -> set[int]:
    return {_opt_type(opt) for opt in (sel.get("option") or [])}


def _safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _iter_agent_decisions(replay: dict, agent_index: int):
    pending = None
    for step in replay.get("steps") or []:
        if not isinstance(step, list) or agent_index >= len(step):
            continue
        pd = step[agent_index]
        if not isinstance(pd, dict):
            continue
        action = pd.get("action")
        if pending is not None and isinstance(action, list) and len(action) != 60:
            yield pending, action
            pending = None
        obs = pd.get("observation")
        obs = obs if isinstance(obs, dict) else None
        sel = obs.get("select") if obs else None
        if pd.get("status") == "ACTIVE" and sel and sel.get("option"):
            pending = obs


def _empty_row():
    return Counter({
        "games": 0,
        "decisions": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "attack_available": 0,
        "attack_chosen": 0,
        "ability_available": 0,
        "ability_chosen": 0,
        "attach_available": 0,
        "attach_chosen": 0,
        "end_chosen": 0,
        "early_end": 0,
    })


def _add_game(row: Counter, won: int, draw: int) -> None:
    row["games"] += 1
    row["wins"] += int(won == 1)
    row["draws"] += int(draw == 1)
    row["losses"] += int(won == 0 and draw == 0)


def _add_decision(row: Counter, sel: dict, chosen_type: int) -> None:
    types = _option_types(sel)
    row["decisions"] += 1
    for opt_type, avail_key, chosen_key in (
        (13, "attack_available", "attack_chosen"),
        (10, "ability_available", "ability_chosen"),
        (8, "attach_available", "attach_chosen"),
    ):
        if opt_type in types:
            row[avail_key] += 1
            row[chosen_key] += int(chosen_type == opt_type)
    row["end_chosen"] += int(chosen_type == 14)
    if chosen_type == 14 and any(t in types for t in (13, 10, 8, 9, 7)):
        row["early_end"] += 1


def _finalize(group: str, row: Counter) -> dict:
    def miss(avail_key: str, chosen_key: str) -> float:
        avail = row[avail_key]
        if avail <= 0:
            return 0.0
        return 1.0 - row[chosen_key] / avail
    return {
        "group": group,
        "games": row["games"],
        "decisions": row["decisions"],
        "wins": row["wins"],
        "losses": row["losses"],
        "draws": row["draws"],
        "attack_available": row["attack_available"],
        "attack_chosen": row["attack_chosen"],
        "miss_attack_rate": f"{miss('attack_available', 'attack_chosen'):.4f}",
        "ability_available": row["ability_available"],
        "ability_chosen": row["ability_chosen"],
        "miss_ability_rate": f"{miss('ability_available', 'ability_chosen'):.4f}",
        "attach_available": row["attach_available"],
        "attach_chosen": row["attach_chosen"],
        "miss_attach_rate": f"{miss('attach_available', 'attach_chosen'):.4f}",
        "end_chosen": row["end_chosen"],
        "early_end_rate": f"{row['early_end'] / max(row['decisions'], 1):.4f}",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", required=True, help="CSV from analyze_kaggle_replays.py")
    p.add_argument("--group-by", default="opponent_deck_sig",
                   choices=["opponent_deck_sig", "opponent_deck_name", "opponent_team", "won"])
    p.add_argument("--archetype", default="", help="optional plan name for future extensions")
    p.add_argument("--out", default="")
    args = p.parse_args()

    if args.archetype:
        get_plan(args.archetype)

    by_group = defaultdict(_empty_row)
    overall = _empty_row()
    choice_types = Counter()
    examples = []
    with open(args.rows, newline="") as f:
        for r in csv.DictReader(f):
            replay_path = r.get("replay_path") or ""
            if not replay_path or not Path(replay_path).exists():
                continue
            won = _safe_int(r.get("won"))
            draw = _safe_int(r.get("draw"))
            group = r.get(args.group_by, "") if args.group_by != "won" else ("win" if won else "loss")
            import json
            replay = json.loads(Path(replay_path).read_text())
            agent_index = _safe_int(r.get("agent_index"))
            _add_game(overall, won, draw)
            _add_game(by_group[group], won, draw)
            for obs, action in _iter_agent_decisions(replay, agent_index):
                sel = obs.get("select") or {}
                chosen = _choice_type(sel, action)
                _add_decision(overall, sel, chosen)
                _add_decision(by_group[group], sel, chosen)
                choice_types[(won, chosen)] += 1
                types = _option_types(sel)
                if len(examples) < 50 and won == 0 and chosen in (14, 7, 10) and 13 in types:
                    examples.append({
                        "episode_id": r.get("episode_id", ""),
                        "opponent": r.get("opponent_deck_name", "") or r.get("opponent_deck_sig", ""),
                        "chosen_type": OPT_NAMES.get(chosen, str(chosen)),
                        "available_types": " ".join(OPT_NAMES.get(t, str(t)) for t in sorted(types)),
                        "action": " ".join(map(str, action)),
                    })

    rows = [_finalize("overall", overall)]
    for group, row in sorted(by_group.items(), key=lambda kv: kv[1]["losses"], reverse=True):
        rows.append(_finalize(group, row))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out}")

    print("Decision summary:")
    for row in rows[:25]:
        print(
            f"  {row['group']:<36} games={row['games']:3} W/L/D={row['wins']}/{row['losses']}/{row['draws']} "
            f"dec={row['decisions']:5} miss_atk={row['miss_attack_rate']} "
            f"miss_abil={row['miss_ability_rate']} miss_attach={row['miss_attach_rate']} "
            f"early_end={row['early_end_rate']}",
            flush=True,
        )
    print("\nChosen type counts by outcome:")
    for (won, typ), n in choice_types.most_common(30):
        print(f"  {'win' if won else 'loss':4s} {OPT_NAMES.get(typ, str(typ)):<12} {n}")
    if examples:
        print("\nLoss examples where attack was available but not chosen:")
        for ex in examples[:20]:
            print(
                f"  {ex['episode_id']} vs {ex['opponent']} chose={ex['chosen_type']} "
                f"available={ex['available_types']} action={ex['action']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
