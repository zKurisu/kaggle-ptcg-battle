#!/usr/bin/env python3
"""Aggregate matchup relations from Kaggle episode ZIPs and replay CSVs.

The episode table is directional: one physical game contributes one row for
player A vs player B and one row for player B vs player A. That makes it easy
to ask "what does this archetype lose to?" without rebuilding the table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from tools.bc_extract_v2 import classify, score_band


ARCH_FIELDS = [
    "source",
    "archetype",
    "opponent_archetype",
    "games",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "edge",
    "first_games",
    "first_win_rate",
    "second_games",
    "second_win_rate",
    "avg_steps",
]

DECK_FIELDS = [
    "source",
    "deck_sig",
    "archetype",
    "team_name",
    "score",
    "score_band",
    "opponent_deck_sig",
    "opponent_archetype",
    "opponent_team_name",
    "opponent_score",
    "opponent_score_band",
    "games",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "edge",
    "first_games",
    "first_win_rate",
    "second_games",
    "second_win_rate",
    "avg_steps",
]

EDGE_FIELDS = [
    "source",
    "level",
    "winner",
    "winner_archetype",
    "loser",
    "loser_archetype",
    "games",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "edge",
    "first_games",
    "first_win_rate",
    "second_games",
    "second_win_rate",
    "avg_steps",
    "winner_score",
    "loser_score",
]


def parse_date(path_or_name: str) -> str:
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", path_or_name)
    return m.group(1) if m else ""


def deck_sig(cards: list[int] | None) -> str:
    if not cards:
        return ""
    compact = ",".join(f"{card}:{count}" for card, count in sorted(Counter(cards).items()))
    return hashlib.sha1(compact.encode("ascii")).hexdigest()[:12]


def parse_cards(text: str | None) -> list[int]:
    if not text:
        return []
    out = []
    for token in str(text).replace(",", " ").split():
        try:
            out.append(int(token))
        except Exception:
            continue
    return out


def read_deck_file(path: str | Path) -> list[int]:
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def metadata_from_deck_path(path: str) -> dict[str, Any]:
    cards = read_deck_file(path)
    sig = deck_sig(cards)
    return {
        "deck_sig": sig,
        "archetype": classify(cards),
        "team_name": Path(path).stem,
        "score": 0.0,
        "score_band": "",
        "deck_path": str(path),
    }


def load_deck_metadata(manifest_paths: list[str], deck_dirs: list[str]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}

    for deck_dir in deck_dirs:
        for path in sorted(Path(deck_dir).glob("*.csv")):
            try:
                row = metadata_from_deck_path(path)
            except Exception:
                continue
            meta.setdefault(row["deck_sig"], row)

    for manifest_path in manifest_paths:
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                sig = (row.get("deck_sig") or "").strip()
                if not sig:
                    deck_path = (row.get("deck_path") or row.get("deck") or "").strip()
                    if deck_path:
                        try:
                            sig = metadata_from_deck_path(deck_path)["deck_sig"]
                        except Exception:
                            sig = ""
                if not sig:
                    continue

                date = row.get("date") or parse_date(manifest_path)
                score = safe_float(row.get("score"))
                team = (
                    row.get("team_name")
                    or row.get("name")
                    or row.get("shadow_name")
                    or row.get("eval_entry", "").split("=", 1)[0]
                    or sig
                )
                arch = row.get("archetype") or ""
                if not arch or arch == "Other":
                    deck_path = (row.get("deck_path") or row.get("deck") or "").strip()
                    if deck_path:
                        try:
                            arch = metadata_from_deck_path(deck_path)["archetype"]
                        except Exception:
                            arch = arch or "Other"
                prev = meta.get(sig)
                should_replace = prev is None
                if prev is not None:
                    prev_date = str(prev.get("date") or "")
                    if date and date >= prev_date:
                        should_replace = True
                    elif score > safe_float(prev.get("score")) and not prev_date:
                        should_replace = True
                if should_replace:
                    meta[sig] = {
                        "deck_sig": sig,
                        "archetype": arch or "Other",
                        "team_name": team,
                        "score": score,
                        "score_band": row.get("score_band") or (score_band(score) if score else ""),
                        "date": date,
                        "deck_path": row.get("deck_path") or row.get("deck") or "",
                    }

    return meta


def default_meta(sig: str, cards: list[int] | None, meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    known = meta.get(sig)
    if known:
        return known
    arch = classify(cards or []) if cards else "Other"
    return {
        "deck_sig": sig,
        "archetype": arch,
        "team_name": sig,
        "score": 0.0,
        "score_band": "",
        "date": "",
        "deck_path": "",
    }


def new_stat() -> dict[str, Any]:
    return {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "first_games": 0,
        "first_wins": 0,
        "second_games": 0,
        "second_wins": 0,
        "steps_total": 0,
    }


def add_stat(
    table: dict[tuple[str, ...], dict[str, Any]],
    key: tuple[str, ...],
    *,
    won: bool,
    draw: bool,
    first: bool | None,
    steps: int,
) -> None:
    row = table.setdefault(key, new_stat())
    row["games"] += 1
    row["wins"] += int(won)
    row["draws"] += int(draw)
    row["losses"] += int((not won) and (not draw))
    row["steps_total"] += int(steps)
    if first is True:
        row["first_games"] += 1
        row["first_wins"] += int(won)
    elif first is False:
        row["second_games"] += 1
        row["second_wins"] += int(won)


def merge_table(dst: dict[tuple[str, ...], dict[str, Any]], src: dict[tuple[str, ...], dict[str, Any]]) -> None:
    for key, row in src.items():
        out = dst.setdefault(key, new_stat())
        for field in out:
            out[field] += row.get(field, 0)


def first_player_from_steps(steps: list[Any]) -> int | None:
    for step in steps[:12]:
        if not isinstance(step, list):
            continue
        for agent in step[:2]:
            if not isinstance(agent, dict):
                continue
            places = []
            obs = agent.get("observation") or {}
            if isinstance(obs, dict):
                places.append(obs.get("current"))
            for view in agent.get("visualize") or []:
                if isinstance(view, dict):
                    places.append(view.get("current"))
            for cur in places:
                if isinstance(cur, dict) and cur.get("firstPlayer") in (0, 1):
                    return int(cur["firstPlayer"])
    return None


def decks_from_episode(data: dict[str, Any]) -> list[list[int]] | None:
    steps = data.get("steps") or []
    if not steps or not isinstance(steps[0], list) or not steps[0]:
        return None
    first = steps[0][0] if isinstance(steps[0][0], dict) else {}
    views = first.get("visualize") or []
    action = views[0].get("action") if views and isinstance(views[0], dict) else None
    if not isinstance(action, list) or len(action) != 2:
        return None
    decks: list[list[int]] = []
    for cards in action[:2]:
        if not isinstance(cards, list) or len(cards) != 60:
            return None
        decks.append([int(x) for x in cards])
    return decks


def episode_passes_score_filter(
    sigs: list[str],
    meta: dict[str, dict[str, Any]],
    min_score: float,
    require_known_score: bool,
) -> bool:
    if min_score <= 0 and not require_known_score:
        return True
    for sig in sigs:
        score = safe_float(meta.get(sig, {}).get("score"))
        if require_known_score and score <= 0:
            return False
        if score < min_score:
            return False
    return True


def process_episode_zip(
    zip_path_text: str,
    meta: dict[str, dict[str, Any]],
    min_score: float,
    require_known_score: bool,
    include_other: bool,
    progress_every: int,
) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[tuple[str, ...], dict[str, Any]], dict[str, int]]:
    zip_path = Path(zip_path_text)
    arch_stats: dict[tuple[str, ...], dict[str, Any]] = {}
    deck_stats: dict[tuple[str, ...], dict[str, Any]] = {}
    counters = Counter()
    t0 = time.time()

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        counters["episode_files"] += len(names)
        for i, name in enumerate(names, 1):
            try:
                data = json.loads(zf.read(name).decode("utf-8"))
                decks = decks_from_episode(data)
                if decks is None:
                    counters["bad_decks"] += 1
                    continue
                rewards = data.get("rewards") or [0, 0]
                if len(rewards) < 2:
                    counters["bad_rewards"] += 1
                    continue
                sigs = [deck_sig(decks[0]), deck_sig(decks[1])]
                if not episode_passes_score_filter(sigs, meta, min_score, require_known_score):
                    counters["score_filtered"] += 1
                    continue
                metas = [default_meta(sigs[0], decks[0], meta), default_meta(sigs[1], decks[1], meta)]
                if not include_other and (metas[0]["archetype"] == "Other" or metas[1]["archetype"] == "Other"):
                    counters["other_filtered"] += 1
                    continue

                first_player = first_player_from_steps(data.get("steps") or [])
                steps_count = len(data.get("steps") or [])
                for pi in (0, 1):
                    oi = 1 - pi
                    reward = safe_float(rewards[pi])
                    opp_reward = safe_float(rewards[oi])
                    won = reward > opp_reward
                    draw = reward == opp_reward
                    first = None if first_player is None else pi == first_player
                    add_stat(
                        arch_stats,
                        (str(metas[pi]["archetype"]), str(metas[oi]["archetype"])),
                        won=won,
                        draw=draw,
                        first=first,
                        steps=steps_count,
                    )
                    add_stat(
                        deck_stats,
                        (sigs[pi], sigs[oi]),
                        won=won,
                        draw=draw,
                        first=first,
                        steps=steps_count,
                    )
                counters["games_used"] += 1
            except Exception:
                counters["errors"] += 1

            if progress_every and (i == 1 or i % progress_every == 0 or i == len(names)):
                rate = i / max(time.time() - t0, 1e-9)
                print(
                    f"{zip_path.name}: {i}/{len(names)} files used_games={counters['games_used']} "
                    f"filtered={counters['score_filtered']} rate={rate:.1f}/s",
                    flush=True,
                )

    return arch_stats, deck_stats, dict(counters)


def iter_episode_zips(episodes_dir: str, date_from: str, date_to: str) -> list[Path]:
    paths = []
    for path in sorted(Path(episodes_dir).glob("pokemon-tcg-ai-battle-episodes-*.zip")):
        date = parse_date(path.name)
        if date_from and date < date_from:
            continue
        if date_to and date > date_to:
            continue
        paths.append(path)
    return paths


def format_rate(num: int, den: int) -> str:
    if den <= 0:
        return ""
    return f"{num / den:.4f}"


def stat_win_rate(row: dict[str, Any]) -> float:
    return float(row["wins"]) / max(int(row["games"]), 1)


def stat_to_common(row: dict[str, Any]) -> dict[str, str]:
    games = int(row["games"])
    wr = stat_win_rate(row)
    first_games = int(row["first_games"])
    second_games = int(row["second_games"])
    return {
        "games": str(games),
        "wins": str(int(row["wins"])),
        "losses": str(int(row["losses"])),
        "draws": str(int(row["draws"])),
        "win_rate": f"{wr:.4f}",
        "edge": f"{wr - 0.5:.4f}",
        "first_games": str(first_games),
        "first_win_rate": format_rate(int(row["first_wins"]), first_games),
        "second_games": str(second_games),
        "second_win_rate": format_rate(int(row["second_wins"]), second_games),
        "avg_steps": f"{int(row['steps_total']) / max(games, 1):.1f}",
    }


def arch_rows(source: str, stats: dict[tuple[str, ...], dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    for (arch, opp_arch), row in stats.items():
        item = {
            "source": source,
            "archetype": arch,
            "opponent_archetype": opp_arch,
        }
        item.update(stat_to_common(row))
        out.append(item)
    return sorted(out, key=lambda r: (-int(r["games"]), r["archetype"], r["opponent_archetype"]))


def deck_rows(
    source: str,
    stats: dict[tuple[str, ...], dict[str, Any]],
    meta: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    out = []
    for (sig, opp_sig), row in stats.items():
        m = default_meta(sig, None, meta)
        om = default_meta(opp_sig, None, meta)
        item = {
            "source": source,
            "deck_sig": sig,
            "archetype": str(m.get("archetype") or "Other"),
            "team_name": str(m.get("team_name") or sig),
            "score": f"{safe_float(m.get('score')):.1f}",
            "score_band": str(m.get("score_band") or ""),
            "opponent_deck_sig": opp_sig,
            "opponent_archetype": str(om.get("archetype") or "Other"),
            "opponent_team_name": str(om.get("team_name") or opp_sig),
            "opponent_score": f"{safe_float(om.get('score')):.1f}",
            "opponent_score_band": str(om.get("score_band") or ""),
        }
        item.update(stat_to_common(row))
        out.append(item)
    return sorted(out, key=lambda r: (-int(r["games"]), r["deck_sig"], r["opponent_deck_sig"]))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def relation_edges_from_arch(
    source: str,
    rows: list[dict[str, str]],
    min_games: int,
    edge_threshold: float,
) -> list[dict[str, str]]:
    out = []
    for row in rows:
        arch = row["archetype"]
        opp = row["opponent_archetype"]
        if arch == opp or arch == "Other" or opp == "Other":
            continue
        games = int(row["games"])
        wr = float(row["win_rate"])
        if games < min_games or wr < 0.5 + edge_threshold:
            continue
        out.append(
            {
                "source": source,
                "level": "archetype",
                "winner": arch,
                "winner_archetype": arch,
                "loser": opp,
                "loser_archetype": opp,
                "winner_score": "",
                "loser_score": "",
                **{k: row[k] for k in stat_to_common(new_stat()).keys() if k in row},
            }
        )
    return sorted(out, key=lambda r: (-float(r["edge"]), -int(r["games"]), r["winner"], r["loser"]))


def relation_edges_from_deck(
    source: str,
    rows: list[dict[str, str]],
    min_games: int,
    edge_threshold: float,
) -> list[dict[str, str]]:
    out = []
    for row in rows:
        winner = row["deck_sig"]
        loser = row["opponent_deck_sig"]
        if winner == loser:
            continue
        games = int(row["games"])
        wr = float(row["win_rate"])
        if games < min_games or wr < 0.5 + edge_threshold:
            continue
        out.append(
            {
                "source": source,
                "level": "deck_sig",
                "winner": winner,
                "winner_archetype": row["archetype"],
                "loser": loser,
                "loser_archetype": row["opponent_archetype"],
                "winner_score": row.get("score", ""),
                "loser_score": row.get("opponent_score", ""),
                **{k: row[k] for k in stat_to_common(new_stat()).keys() if k in row},
            }
        )
    return sorted(out, key=lambda r: (-float(r["edge"]), -int(r["games"]), r["winner"], r["loser"]))


def add_replay_rows(
    replay_paths: list[str],
    meta: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[tuple[str, ...], dict[str, Any]], Counter]:
    arch_stats: dict[tuple[str, ...], dict[str, Any]] = {}
    deck_stats: dict[tuple[str, ...], dict[str, Any]] = {}
    counters = Counter()
    for replay_path in replay_paths:
        with open(replay_path, newline="") as f:
            for row in csv.DictReader(f):
                our_cards = parse_cards(row.get("our_deck_ids"))
                opp_cards = parse_cards(row.get("opponent_deck_ids"))
                our_sig = row.get("our_deck_sig") or deck_sig(our_cards)
                opp_sig = row.get("opponent_deck_sig") or deck_sig(opp_cards)
                if not our_sig or not opp_sig:
                    counters["missing_deck"] += 1
                    continue
                our_meta = default_meta(our_sig, our_cards, meta)
                opp_meta = default_meta(opp_sig, opp_cards, meta)
                won = str(row.get("won") or "0") == "1"
                draw = str(row.get("draw") or "0") == "1"
                first_text = str(row.get("our_first") if row.get("our_first") is not None else "")
                first = None
                if first_text == "1":
                    first = True
                elif first_text == "0":
                    first = False
                steps = int(safe_float(row.get("steps")))
                add_stat(
                    arch_stats,
                    (str(our_meta["archetype"]), str(opp_meta["archetype"])),
                    won=won,
                    draw=draw,
                    first=first,
                    steps=steps,
                )
                add_stat(deck_stats, (our_sig, opp_sig), won=won, draw=draw, first=first, steps=steps)
                counters["replay_rows"] += 1
    return arch_stats, deck_stats, counters


def row_label(row: dict[str, str], winner_field: str, arch_field: str) -> str:
    name = row.get(winner_field, "")
    arch = row.get(arch_field, "")
    if name == arch or len(name) <= 12:
        return f"{arch} ({name})" if name and name != arch else arch
    return f"{arch} {name}"


def md_table(rows: list[dict[str, str]], fields: list[str], limit: int) -> list[str]:
    rows = rows[:limit]
    if not rows:
        return ["No rows."]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    return lines


def summarize_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    episode_counters: Counter,
    episode_arch_rows: list[dict[str, str]],
    episode_deck_rows: list[dict[str, str]],
    episode_arch_edges: list[dict[str, str]],
    episode_deck_edges: list[dict[str, str]],
    replay_arch_rows: list[dict[str, str]],
    replay_deck_rows: list[dict[str, str]],
    replay_arch_edges: list[dict[str, str]],
) -> None:
    lines: list[str] = []
    lines.append("# Matchup Relations")
    lines.append("")
    lines.append(f"Generated from `{args.episodes_dir}`.")
    lines.append(f"Episode date window: `{args.date_from or 'begin'}` to `{args.date_to or 'end'}`.")
    lines.append(
        f"Filters: min_score={args.score_floor:.1f}, require_known_score={int(args.require_known_score)}, "
        f"include_other={int(args.include_other)}, edge_threshold={args.edge_threshold:.3f}."
    )
    lines.append(
        "Interpretation: rows are policy-and-deck results from ladder episodes. They are not pure card matchup odds; "
        "player policy quality and first/second player bias are part of the signal."
    )
    lines.append("")

    lines.append("## Episode Counters")
    lines.append("")
    for key in sorted(episode_counters):
        lines.append(f"- {key}: {episode_counters[key]}")
    lines.append("")

    lines.append("## Archetype Counter Edges")
    lines.append("")
    lines.extend(
        md_table(
            episode_arch_edges,
            ["winner", "loser", "games", "win_rate", "edge", "first_games", "first_win_rate", "second_games", "second_win_rate", "avg_steps"],
            args.md_limit,
        )
    )
    lines.append("")

    lines.append("## Per-Archetype Weaknesses")
    lines.append("")
    weaknesses = []
    for row in episode_arch_rows:
        if row["archetype"] == row["opponent_archetype"] or "Other" in (row["archetype"], row["opponent_archetype"]):
            continue
        if int(row["games"]) >= args.min_games and float(row["win_rate"]) <= 0.5 - args.edge_threshold:
            weaknesses.append(row)
    weaknesses.sort(key=lambda r: (r["archetype"], float(r["win_rate"]), -int(r["games"])))
    lines.extend(md_table(weaknesses, ["archetype", "opponent_archetype", "games", "win_rate", "edge", "avg_steps"], args.md_limit * 2))
    lines.append("")

    lines.append("## Deck-Sig Counter Edges")
    lines.append("")
    lines.extend(
        md_table(
            episode_deck_edges,
            ["winner_archetype", "winner", "loser_archetype", "loser", "games", "win_rate", "edge", "winner_score", "loser_score"],
            args.md_limit,
        )
    )
    lines.append("")

    if replay_arch_rows:
        lines.append("## Latest Kaggle Replay Observations")
        lines.append("")
        lines.append(
            "Replay rows only cover the currently pulled submissions, so treat them as live probes rather than a full ladder matrix."
        )
        lines.append("")
        lines.extend(
            md_table(
                sorted(replay_arch_rows, key=lambda r: (-int(r["games"]), r["archetype"], r["opponent_archetype"])),
                ["archetype", "opponent_archetype", "games", "wins", "losses", "win_rate", "edge", "avg_steps"],
                args.md_limit,
            )
        )
        lines.append("")
        if replay_arch_edges:
            lines.append("### Replay Counter Edges")
            lines.append("")
            lines.extend(md_table(replay_arch_edges, ["winner", "loser", "games", "win_rate", "edge", "avg_steps"], args.md_limit))
            lines.append("")
        replay_deck_focus = [
            r for r in replay_deck_rows
            if int(r["games"]) >= max(2, min(args.min_deck_games, 5))
        ]
        replay_deck_focus.sort(key=lambda r: (-int(r["games"]), float(r["win_rate"]), r["deck_sig"]))
        lines.append("### Replay Deck Observations")
        lines.append("")
        lines.extend(
            md_table(
                replay_deck_focus,
                ["archetype", "deck_sig", "opponent_archetype", "opponent_deck_sig", "games", "wins", "losses", "win_rate"],
                args.md_limit,
            )
        )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-dir", default="/home/jie/Do/0_PTCG/workspace/episodes_raw")
    p.add_argument("--date-from", default="")
    p.add_argument("--date-to", default="")
    p.add_argument("--deck-manifest", action="append", default=[])
    p.add_argument("--known-decks-dir", action="append", default=[])
    p.add_argument("--replay-rows", action="append", default=[])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--source-label", default="episodes")
    p.add_argument("--score-floor", type=float, default=0.0)
    p.add_argument("--require-known-score", action="store_true")
    p.add_argument("--include-other", action="store_true")
    p.add_argument("--min-games", type=int, default=50)
    p.add_argument("--min-deck-games", type=int, default=10)
    p.add_argument("--edge-threshold", type=float, default=0.08)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--md-limit", type=int, default=30)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_deck_metadata(args.deck_manifest, args.known_decks_dir)
    zips = iter_episode_zips(args.episodes_dir, args.date_from, args.date_to)
    if not zips:
        raise FileNotFoundError(f"no episode zips found in {args.episodes_dir}")
    print(f"loaded deck metadata: {len(meta)} signatures", flush=True)
    print(f"episode zips: {len(zips)}", flush=True)

    episode_arch_stats: dict[tuple[str, ...], dict[str, Any]] = {}
    episode_deck_stats: dict[tuple[str, ...], dict[str, Any]] = {}
    episode_counters: Counter = Counter()

    if args.workers <= 1 or len(zips) == 1:
        for path in zips:
            arch_part, deck_part, cnt_part = process_episode_zip(
                str(path),
                meta,
                args.score_floor,
                args.require_known_score,
                args.include_other,
                args.progress_every,
            )
            merge_table(episode_arch_stats, arch_part)
            merge_table(episode_deck_stats, deck_part)
            episode_counters.update(cnt_part)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(
                    process_episode_zip,
                    str(path),
                    meta,
                    args.score_floor,
                    args.require_known_score,
                    args.include_other,
                    args.progress_every,
                )
                for path in zips
            ]
            for fut in as_completed(futs):
                arch_part, deck_part, cnt_part = fut.result()
                merge_table(episode_arch_stats, arch_part)
                merge_table(episode_deck_stats, deck_part)
                episode_counters.update(cnt_part)

    episode_arch = arch_rows(args.source_label, episode_arch_stats)
    episode_deck = deck_rows(args.source_label, episode_deck_stats, meta)
    episode_arch_edges = relation_edges_from_arch(args.source_label, episode_arch, args.min_games, args.edge_threshold)
    episode_deck_edges = relation_edges_from_deck(args.source_label, episode_deck, args.min_deck_games, args.edge_threshold)

    replay_arch_stats: dict[tuple[str, ...], dict[str, Any]] = {}
    replay_deck_stats: dict[tuple[str, ...], dict[str, Any]] = {}
    replay_counters: Counter = Counter()
    if args.replay_rows:
        replay_arch_stats, replay_deck_stats, replay_counters = add_replay_rows(args.replay_rows, meta)
    replay_arch = arch_rows("latest_replay", replay_arch_stats)
    replay_deck = deck_rows("latest_replay", replay_deck_stats, meta)
    replay_arch_edges = relation_edges_from_arch("latest_replay", replay_arch, max(2, min(args.min_games, 5)), args.edge_threshold)
    replay_deck_edges = relation_edges_from_deck("latest_replay", replay_deck, max(2, min(args.min_deck_games, 5)), args.edge_threshold)

    write_csv(out_dir / "archetype_matchups.csv", ARCH_FIELDS, episode_arch)
    write_csv(out_dir / "deck_sig_matchups.csv", DECK_FIELDS, episode_deck)
    write_csv(out_dir / "archetype_counter_edges.csv", EDGE_FIELDS, episode_arch_edges)
    write_csv(out_dir / "deck_sig_counter_edges.csv", EDGE_FIELDS, episode_deck_edges)
    write_csv(out_dir / "replay_archetype_matchups.csv", ARCH_FIELDS, replay_arch)
    write_csv(out_dir / "replay_deck_sig_matchups.csv", DECK_FIELDS, replay_deck)
    write_csv(out_dir / "replay_archetype_counter_edges.csv", EDGE_FIELDS, replay_arch_edges)
    write_csv(out_dir / "replay_deck_sig_counter_edges.csv", EDGE_FIELDS, replay_deck_edges)

    all_counters = Counter(episode_counters)
    for key, value in replay_counters.items():
        all_counters[f"replay_{key}"] = value
    with (out_dir / "run_summary.txt").open("w") as f:
        for key in sorted(all_counters):
            f.write(f"{key}: {all_counters[key]}\n")

    summarize_markdown(
        out_dir / "matchup_summary.md",
        args=args,
        episode_counters=all_counters,
        episode_arch_rows=episode_arch,
        episode_deck_rows=episode_deck,
        episode_arch_edges=episode_arch_edges,
        episode_deck_edges=episode_deck_edges,
        replay_arch_rows=replay_arch,
        replay_deck_rows=replay_deck,
        replay_arch_edges=replay_arch_edges,
    )

    print(f"wrote {out_dir}", flush=True)
    print(
        f"episode_games_used={episode_counters.get('games_used', 0)} "
        f"arch_edges={len(episode_arch_edges)} deck_edges={len(episode_deck_edges)} "
        f"replay_rows={replay_counters.get('replay_rows', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
