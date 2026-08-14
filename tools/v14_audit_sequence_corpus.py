#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.seq.constants import ACTION_TYPE_NAMES, FUTURE_PLAN_DIM, type_id_name
from ptcg_rl.seq.data import discover_sequence_npz


def _split_csv(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--archetype", required=True)
    p.add_argument("--score-bands", nargs="+", default=["600-699", "700-799", "800-899", "900-999", "1000-1099", "1100-1199", "1200+"])
    p.add_argument("--date-from", default="")
    p.add_argument("--date-to", default="")
    p.add_argument("--deck-sig", action="append", default=[])
    p.add_argument("--team-name", action="append", default=[])
    p.add_argument("--out-csv", default="")
    args = p.parse_args()

    deck_sigs = set(_split_csv(args.deck_sig))
    teams = {x.lower() for x in _split_csv(args.team_name)}
    paths = discover_sequence_npz(
        args.corpus,
        args.archetype,
        _split_csv(args.score_bands),
        date_from=args.date_from,
        date_to=args.date_to,
    )
    print(f"paths={len(paths)} archetype={args.archetype}")
    rows_out: list[dict[str, object]] = []
    type_counts: Counter[int] = Counter()
    future_sums = np.zeros(FUTURE_PLAN_DIM, dtype=np.float64)
    future_n = 0
    games: dict[str, list[tuple[int, int, int, float, str, str]]] = defaultdict(list)
    scores: list[float] = []
    wins = draws = total = 0
    opp_arch: Counter[str] = Counter()
    deck_counter: Counter[str] = Counter()

    for path in paths:
        with np.load(path, allow_pickle=True) as z:
            n = len(z["board"])
            for i in range(n):
                deck_sig = str(z["deck_sig"][i]) if "deck_sig" in z else ""
                team = str(z["team_name"][i]) if "team_name" in z else ""
                if deck_sigs and deck_sig not in deck_sigs:
                    continue
                if teams and team.lower() not in teams:
                    continue
                game_key = str(z["game_key"][i]) if "game_key" in z else f"{z['episode_id'][i]}:{z['player_index'][i]}"
                typ = int(z["act_type"][i]) if "act_type" in z else -1
                dec = int(z["decision_index"][i]) if "decision_index" in z else i
                won = int(z["won"][i]) if "won" in z else 0
                score = float(z["score"][i]) if "score" in z else 0.0
                opp = str(z["opponent_archetype"][i]) if "opponent_archetype" in z else ""
                games[game_key].append((dec, typ, won, score, deck_sig, opp))
                type_counts[typ] += 1
                total += 1
                wins += won
                draws += int(z["draw"][i]) if "draw" in z else 0
                scores.append(score)
                opp_arch[opp] += 1
                deck_counter[deck_sig] += 1
                if "future_plan" in z:
                    future_sums += np.asarray(z["future_plan"][i], dtype=np.float64)[:FUTURE_PLAN_DIM]
                    future_n += 1

    game_lengths = [len(v) for v in games.values()]
    print(json.dumps({
        "rows": total,
        "games": len(games),
        "win_row_rate": wins / max(total, 1),
        "draw_row_rate": draws / max(total, 1),
        "score_mean": float(np.mean(scores)) if scores else 0.0,
        "score_min": float(np.min(scores)) if scores else 0.0,
        "score_max": float(np.max(scores)) if scores else 0.0,
        "game_len_mean": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "game_len_p50": float(np.percentile(game_lengths, 50)) if game_lengths else 0.0,
        "game_len_p90": float(np.percentile(game_lengths, 90)) if game_lengths else 0.0,
    }, ensure_ascii=False, sort_keys=True))

    print("action_types")
    for typ, count in type_counts.most_common():
        print(f"  {type_id_name(typ):<18} {count:>8} {count / max(total, 1):.3f}")

    print("top_opponent_archetypes")
    for name, count in opp_arch.most_common(20):
        print(f"  {name:<24} {count:>8} {count / max(total, 1):.3f}")

    print("top_deck_sigs")
    for sig, count in deck_counter.most_common(20):
        print(f"  {sig:<16} {count:>8} {count / max(total, 1):.3f}")

    if future_n:
        means = future_sums / future_n
        print("future_plan_mean")
        for i, value in enumerate(means):
            print(f"  f{i:02d} {value:.4f}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["game_key", "decisions", "won", "score", "deck_sig", "opponent_archetype", "attack_count", "attach_before_attack", "evolve_before_attack", "ability_before_evolve"])
            writer.writeheader()
            for key, vals in games.items():
                vals = sorted(vals)
                types = [v[1] for v in vals]
                score = vals[-1][3]
                won = vals[-1][2]
                deck_sig = vals[-1][4]
                opp = vals[-1][5]
                writer.writerow({
                    "game_key": key,
                    "decisions": len(vals),
                    "won": won,
                    "score": score,
                    "deck_sig": deck_sig,
                    "opponent_archetype": opp,
                    "attack_count": sum(1 for t in types if ACTION_TYPE_NAMES.get(t) == "ATTACK"),
                    "attach_before_attack": _before_type(types, "ATTACH", "ATTACK"),
                    "evolve_before_attack": _before_type(types, "EVOLVE", "ATTACK"),
                    "ability_before_evolve": _before_type(types, "ABILITY", "EVOLVE"),
                })
        print(f"wrote {args.out_csv}")


def _before_type(types: list[int], a_name: str, b_name: str) -> int:
    rev = {v: k for k, v in ACTION_TYPE_NAMES.items()}
    a = rev.get(a_name)
    b = rev.get(b_name)
    if a is None or b is None:
        return 0
    try:
        return int(types.index(a) < types.index(b))
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
