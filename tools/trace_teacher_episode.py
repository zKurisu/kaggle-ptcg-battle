#!/usr/bin/env python3
"""Emit human-readable traces for real Kaggle teacher episodes.

This complements v15_trace_game.py.  v15_trace_game traces our local model;
this script traces the actual actions stored in Kaggle episode JSON files so
we can inspect how high-scoring teams play specific matchups.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.deck_registry import deck_signature
from ptcg_rl.encoder import FastEncoder
from tools.bc_extract_v2 import classify, score_band
from tools.trace_matchup_decisions import card_name, encode_decision, safe_int
from tools.v15_trace_game import board_line, format_decision, option_detail


AREA_NAMES = {
    1: "deck",
    2: "hand",
    3: "discard",
    4: "active",
    5: "bench",
    6: "prize",
    7: "stadium",
    8: "energy",
    10: "evolution_stack",
    12: "look",
    14: "lost_zone",
}


def parse_date(path_or_name: str) -> str:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", str(path_or_name))
    return match.group(1) if match else ""


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def iter_zips(path: Path, date_from: str, date_to: str) -> list[Path]:
    zips = [path] if path.is_file() else sorted(path.glob("pokemon-tcg-ai-battle-episodes-*.zip"))
    out = []
    for zpath in zips:
        date = parse_date(zpath.name)
        if date_from and date and date < date_from:
            continue
        if date_to and date and date > date_to:
            continue
        out.append(zpath)
    return out


def load_manifest(paths: list[str]) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                sig = (row.get("deck_sig") or "").strip()
                if not sig:
                    continue
                prev = meta.get(sig)
                score = safe_float(row.get("score"))
                date = row.get("date") or parse_date(raw)
                replace = prev is None
                if prev is not None:
                    prev_date = prev.get("date") or ""
                    prev_score = safe_float(prev.get("score"))
                    replace = bool(date and date >= prev_date) or score > prev_score
                if replace:
                    meta[sig] = {
                        "deck_sig": sig,
                        "team_name": row.get("team_name") or row.get("name") or sig,
                        "score": f"{score:.1f}" if score else row.get("score", ""),
                        "score_band": row.get("score_band") or (score_band(score) if score else ""),
                        "archetype": row.get("archetype") or "",
                        "date": date,
                    }
    return meta


def team_allowed(team_name: str, filters: set[str]) -> bool:
    if not filters:
        return True
    low = team_name.lower()
    return any(item in low for item in filters)


def team_exact_allowed(team_name: str, filters: set[str]) -> bool:
    if not filters:
        return True
    return team_name.lower() in filters


def action_valid(action: Any, sel: dict) -> bool:
    if not isinstance(action, list) or len(action) == 60:
        return False
    options = sel.get("option") or []
    return all(isinstance(idx, int) and 0 <= idx < len(options) for idx in action)


def iter_teacher_decisions(data: dict, player_index: int):
    pending: dict | None = None
    decision_index = 0
    for step_index, step in enumerate(data.get("steps") or []):
        if not isinstance(step, list) or player_index >= len(step):
            continue
        pd = step[player_index]
        if not isinstance(pd, dict):
            continue
        action = pd.get("action")
        if pending is not None and action_valid(action, pending.get("select") or {}):
            yield pending["obs"], action, pending["step_index"], pending["decision_index"]
            pending = None

        obs = pd.get("observation")
        obs = obs if isinstance(obs, dict) else None
        sel = obs.get("select") if obs else None
        if pd.get("status") == "ACTIVE" and sel and sel.get("option"):
            pending = {
                "obs": obs,
                "select": sel,
                "step_index": step_index,
                "decision_index": decision_index,
            }
            decision_index += 1


def area_name(area: Any) -> str:
    value = safe_int(area, -1)
    return AREA_NAMES.get(value, str(area))


def format_log_event(event: dict) -> str:
    typ = safe_int(event.get("type"), -1)
    pid = event.get("playerIndex")
    prefix = f"p{pid} " if pid is not None else ""
    cid = safe_int(event.get("cardId"))
    cname = card_name(cid) if cid else ""
    if typ == 4 and cid:
        return f"{prefix}draw/reveal {cname}({cid}) serial={event.get('serial', '')}"
    if typ == 6:
        card = f"{cname}({cid}) " if cid else ""
        return (
            f"{prefix}move {card}{area_name(event.get('fromArea'))}"
            f"->{area_name(event.get('toArea'))} serial={event.get('serial', '')}"
        )
    if typ == 7:
        return f"{prefix}move/unknown {area_name(event.get('fromArea'))}->{area_name(event.get('toArea'))}"
    if typ == 8:
        active = card_name(safe_int(event.get("cardIdActive")))
        bench = card_name(safe_int(event.get("cardIdBench")))
        return f"{prefix}switch active={active} bench={bench}"
    if typ == 10 and cid:
        return f"{prefix}play/ability/effect {cname}({cid}) serial={event.get('serial', '')}"
    if typ == 11 and cid:
        target = card_name(safe_int(event.get("cardIdTarget")))
        return f"{prefix}attach {cname}({cid}) -> {target}({event.get('cardIdTarget', '')})"
    if typ == 12 and cid:
        target = card_name(safe_int(event.get("cardIdTarget")))
        return f"{prefix}evolve {target}({event.get('cardIdTarget', '')}) -> {cname}({cid})"
    if typ == 15:
        return f"{prefix}attack card={cname}({cid}) attackId={event.get('attackId', '')}"
    if typ == 16 and cid:
        value = event.get("value", "")
        mode = "counter" if event.get("putDamageCounter") else "damage"
        return f"{prefix}{mode} {cname}({cid}) value={value}"
    if typ == 22:
        return f"{prefix}coin head={event.get('head')}"
    if "hasBasicPokemon" in event:
        return f"{prefix}hasBasicPokemon={event.get('hasBasicPokemon')}"
    return json.dumps(event, ensure_ascii=False, sort_keys=True)[:300]


def public_logs(obs: dict, limit: int) -> list[str]:
    logs = obs.get("logs") or []
    if not isinstance(logs, list):
        return []
    out = []
    for item in logs[-limit:]:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            text = item.get("message") or item.get("text") or item.get("log") or ""
            if text:
                out.append(str(text))
            else:
                out.append(format_log_event(item))
        else:
            out.append(str(item))
    return out


def option_sample(obs: dict, limit: int) -> str:
    sel = obs.get("select") or {}
    options = sel.get("option") or []
    wanted = []
    for idx, opt in enumerate(options):
        typ = safe_int((opt or {}).get("type"), -1)
        if typ in (7, 8, 9, 10, 12, 13) or safe_int(sel.get("context")) in (13, 14, 16, 35, 37):
            wanted.append(idx)
    if not wanted:
        wanted = list(range(min(len(options), limit)))
    return " || ".join(option_detail(obs, idx) for idx in wanted[:limit])


def decks_from_episode(data: dict) -> list[list[int]] | None:
    try:
        decks = data["steps"][0][0].get("visualize", [{}])[0].get("action", [])
    except Exception:
        return None
    if not isinstance(decks, list) or len(decks) != 2:
        return None
    if any(not isinstance(deck, list) or len(deck) != 60 for deck in decks[:2]):
        return None
    return [list(map(int, decks[0])), list(map(int, decks[1]))]


def episode_summary(
    data: dict,
    zpath: Path,
    fname: str,
    player_index: int,
    decks: list[list[int]],
    meta: dict[str, dict[str, str]],
) -> dict[str, str]:
    teams = (data.get("info") or {}).get("TeamNames") or []
    sigs = [deck_signature(decks[0]), deck_signature(decks[1])]
    archs = [classify(decks[0]), classify(decks[1])]
    rewards = data.get("rewards") or [0, 0]
    reward = safe_float(rewards[player_index] if player_index < len(rewards) else 0)
    opp_reward = safe_float(rewards[1 - player_index] if 1 - player_index < len(rewards) else 0)
    known = meta.get(sigs[player_index], {})
    opp_known = meta.get(sigs[1 - player_index], {})
    return {
        "episode_id": str(data.get("id") or (data.get("info") or {}).get("EpisodeId") or Path(fname).stem),
        "zip": str(zpath),
        "file": fname,
        "date": parse_date(zpath.name),
        "player_index": str(player_index),
        "team_name": str(teams[player_index] if player_index < len(teams) else known.get("team_name", "")),
        "deck_sig": sigs[player_index],
        "archetype": known.get("archetype") or archs[player_index],
        "score": known.get("score", ""),
        "score_band": known.get("score_band", ""),
        "opponent_team_name": str(teams[1 - player_index] if 1 - player_index < len(teams) else opp_known.get("team_name", "")),
        "opponent_deck_sig": sigs[1 - player_index],
        "opponent_archetype": opp_known.get("archetype") or archs[1 - player_index],
        "opponent_score": opp_known.get("score", ""),
        "opponent_score_band": opp_known.get("score_band", ""),
        "reward": f"{reward:.1f}",
        "opponent_reward": f"{opp_reward:.1f}",
        "won": str(int(reward > opp_reward)),
        "draw": str(int(reward == opp_reward)),
        "steps": str(len(data.get("steps") or [])),
    }


def passes_filters(row: dict[str, str], args: argparse.Namespace) -> bool:
    if args.episode_id and row["episode_id"] not in set(args.episode_id):
        return False
    if args.archetype and row["archetype"] != args.archetype:
        return False
    if args.opponent_archetype and row["opponent_archetype"] != args.opponent_archetype:
        return False
    if args.score_floor and safe_float(row.get("score")) < args.score_floor:
        return False
    if args.opponent_score_floor and safe_float(row.get("opponent_score")) < args.opponent_score_floor:
        return False
    if args.deck_sig and row["deck_sig"] not in set(args.deck_sig):
        return False
    if args.opponent_deck_sig and row["opponent_deck_sig"] not in set(args.opponent_deck_sig):
        return False
    if not team_allowed(row["team_name"], {x.lower() for x in args.team_name}):
        return False
    if not team_allowed(row["opponent_team_name"], {x.lower() for x in args.opponent_team_name}):
        return False
    if not team_exact_allowed(row["team_name"], {x.lower() for x in args.exact_team_name}):
        return False
    if not team_exact_allowed(row["opponent_team_name"], {x.lower() for x in args.exact_opponent_team_name}):
        return False
    if args.outcome == "win" and row["won"] != "1":
        return False
    if args.outcome == "loss" and row["won"] != "0":
        return False
    return True


def trace_one_game(
    data: dict,
    row: dict[str, str],
    decks: list[list[int]],
    args: argparse.Namespace,
    encoder: FastEncoder,
) -> tuple[list[dict], list[str]]:
    decisions = []
    notes = []
    player_index = int(row["player_index"])
    for obs, action, step_index, decision_index in iter_teacher_decisions(data, player_index):
        try:
            dec = encode_decision(
                encoder,
                obs,
                action,
                game=0,
                step=step_index,
                candidate_side=player_index,
                policy=None,
            )
            dec["decision_index"] = decision_index
            dec["_obs"] = obs
            decisions.append(dec)
        except Exception as exc:
            notes.append(f"encode_error step={step_index} decision={decision_index}: {exc}")
        if args.max_decisions and len(decisions) >= args.max_decisions:
            break
    return decisions, notes


def write_trace(path: Path, row: dict[str, str], decisions: list[dict], notes: list[str], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Teacher Episode Trace\n\n")
        for key in (
            "date", "episode_id", "team_name", "deck_sig", "archetype", "score", "score_band",
            "opponent_team_name", "opponent_deck_sig", "opponent_archetype", "opponent_score",
            "opponent_score_band", "won", "reward", "opponent_reward", "steps",
        ):
            f.write(f"{key}: `{row.get(key, '')}`\n\n")
        if notes:
            f.write("## Notes\n\n")
            for note in notes:
                f.write(f"- {note}\n")
            f.write("\n")
        f.write("## Decisions\n\n")
        for i, dec in enumerate(decisions):
            obs = dec.pop("_obs")
            cur = obs.get("current") or {}
            side = safe_int(cur.get("yourIndex"))
            f.write(f"### Decision {i} step={dec.get('step')} turn={dec.get('turn')} tac={dec.get('turn_action_count')}\n\n")
            for line in format_decision(dec, obs):
                if line.strip().startswith("top:"):
                    continue
                f.write(line + "\n")
            sample = option_sample(obs, args.option_sample)
            if sample:
                f.write(f"  candidate_options: {sample}\n")
            logs = public_logs(obs, args.log_lines)
            if logs:
                f.write("  public_logs:\n")
                for item in logs:
                    f.write(f"    - {item}\n")
            f.write(f"  board_snapshot: {board_line(cur, side)}\n\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("episodes", help="episode zip or directory")
    p.add_argument("--deck-manifest", action="append", default=[])
    p.add_argument("--date-from", default="")
    p.add_argument("--date-to", default="")
    p.add_argument("--episode-id", action="append", default=[],
                   help="exact episode id filter; may be repeated")
    p.add_argument("--archetype", default="")
    p.add_argument("--opponent-archetype", default="")
    p.add_argument("--score-floor", type=float, default=0.0)
    p.add_argument("--opponent-score-floor", type=float, default=0.0)
    p.add_argument("--deck-sig", action="append", default=[])
    p.add_argument("--opponent-deck-sig", action="append", default=[])
    p.add_argument("--team-name", action="append", default=[],
                   help="substring filter; may be repeated")
    p.add_argument("--opponent-team-name", action="append", default=[])
    p.add_argument("--exact-team-name", action="append", default=[],
                   help="case-insensitive exact team filter; may be repeated")
    p.add_argument("--exact-opponent-team-name", action="append", default=[])
    p.add_argument("--outcome", choices=["any", "win", "loss"], default="any")
    p.add_argument("--limit-games", type=int, default=5)
    p.add_argument("--max-decisions", type=int, default=0)
    p.add_argument("--option-sample", type=int, default=16)
    p.add_argument("--log-lines", type=int, default=8)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--progress-every", type=int, default=500)
    args = p.parse_args()

    zips = iter_zips(Path(args.episodes), args.date_from, args.date_to)
    if not zips:
        raise FileNotFoundError(f"no episode zips found under {args.episodes}")
    meta = load_manifest(args.deck_manifest)
    encoder = FastEncoder()
    out_dir = Path(args.out_dir)
    index_rows: list[dict[str, str]] = []
    counters = Counter()
    t0 = time.time()

    for zpath in zips:
        with zipfile.ZipFile(zpath) as zf:
            names = [n for n in zf.namelist() if n.endswith(".json")]
            print(f"{zpath.name}: {len(names)} episodes", flush=True)
            for idx, name in enumerate(names, 1):
                counters["episodes_seen"] += 1
                try:
                    data = json.loads(zf.read(name).decode("utf-8"))
                    decks = decks_from_episode(data)
                    if decks is None:
                        counters["bad_decks"] += 1
                        continue
                    for pi in (0, 1):
                        row = episode_summary(data, zpath, name, pi, decks, meta)
                        if not passes_filters(row, args):
                            continue
                        decisions, notes = trace_one_game(data, row, decks, args, encoder)
                        row["decisions"] = str(len(decisions))
                        row["trace_path"] = str(
                            out_dir / f"{row['date']}_{row['episode_id']}_p{pi}_{row['deck_sig']}_vs_{row['opponent_deck_sig']}.md"
                        )
                        write_trace(Path(row["trace_path"]), row, decisions, notes, args)
                        index_rows.append(row)
                        counters["games_written"] += 1
                        print(
                            f"  wrote {row['episode_id']} p{pi} {row['archetype']} {row['deck_sig']} "
                            f"{row['team_name']} vs {row['opponent_archetype']} {row['opponent_deck_sig']} "
                            f"won={row['won']} decisions={row['decisions']}",
                            flush=True,
                        )
                        if counters["games_written"] >= args.limit_games:
                            break
                    if counters["games_written"] >= args.limit_games:
                        break
                except Exception as exc:
                    counters["errors"] += 1
                    if counters["errors"] <= 10:
                        print(f"  error {name}: {exc}", flush=True)
                if args.progress_every and idx % args.progress_every == 0:
                    elapsed = time.time() - t0
                    rate = counters["episodes_seen"] / max(elapsed, 1e-9)
                    print(
                        f"  progress seen={counters['episodes_seen']} written={counters['games_written']} "
                        f"errors={counters['errors']} rate={rate:.1f}/s",
                        flush=True,
                    )
            if counters["games_written"] >= args.limit_games:
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.csv"
    fields = [
        "date", "episode_id", "player_index", "team_name", "deck_sig", "archetype", "score",
        "score_band", "opponent_team_name", "opponent_deck_sig", "opponent_archetype",
        "opponent_score", "opponent_score_band", "won", "draw", "reward", "opponent_reward",
        "steps", "decisions", "zip", "file", "trace_path",
    ]
    with index_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(index_rows)
    print(f"Done counters={dict(counters)} wrote {index_path}", flush=True)

if __name__ == "__main__":
    main()
