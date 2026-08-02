#!/usr/bin/env python3
"""Download and summarize Kaggle replay outcomes for one submission."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROW_FIELDS = [
    "submission_id",
    "episode_id",
    "create_time",
    "end_time",
    "episode_state",
    "episode_type",
    "agent_index",
    "opponent_index",
    "team_name",
    "opponent_team",
    "reward",
    "opponent_reward",
    "won",
    "draw",
    "status",
    "opponent_status",
    "steps",
    "first_player",
    "our_first",
    "our_deck_name",
    "opponent_deck_name",
    "our_deck_sig",
    "opponent_deck_sig",
    "our_deck_ids",
    "opponent_deck_ids",
    "replay_path",
]

SUMMARY_FIELDS = [
    "group",
    "games",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "avg_reward",
    "avg_steps",
]


def run_kaggle(cmd: list[str], timeout: float) -> str:
    proc = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc.stdout


def parse_json_text(text: str) -> Any:
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        raise ValueError(f"no JSON found in Kaggle output: {text[:200]}")
    data, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return data


def fetch_episodes(submission_id: str, timeout: float) -> list[dict]:
    text = run_kaggle(
        ["kaggle", "competitions", "episodes", str(submission_id), "--format", "json"],
        timeout=timeout,
    )
    data = parse_json_text(text)
    if not isinstance(data, list):
        raise ValueError("unexpected episodes JSON; expected a list")
    return data


def replay_path(cache_dir: Path, episode_id: str) -> Path | None:
    matches = sorted(cache_dir.glob(f"*{episode_id}*replay*.json"))
    return matches[0] if matches else None


def download_replay(episode_id: str, cache_dir: Path, timeout: float, force: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    old = replay_path(cache_dir, episode_id)
    if old is not None and not force:
        return old
    run_kaggle(
        ["kaggle", "competitions", "replay", str(episode_id), "-p", str(cache_dir), "-q"],
        timeout=timeout,
    )
    new = replay_path(cache_dir, episode_id)
    if new is None:
        raise FileNotFoundError(f"downloaded replay not found for episode {episode_id} in {cache_dir}")
    return new


def download_logs(episode_id: str, cache_dir: Path, timeout: float) -> None:
    for agent_index in (0, 1):
        run_kaggle(
            [
                "kaggle",
                "competitions",
                "logs",
                str(episode_id),
                str(agent_index),
                "-p",
                str(cache_dir),
                "-q",
            ],
            timeout=timeout,
        )


def read_deck(path: str | None) -> list[int] | None:
    if not path:
        return None
    with open(path) as f:
        cards = [int(line.strip()) for line in f if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"deck must contain 60 cards: {path} has {len(cards)}")
    return cards


def deck_counter(cards: list[int] | None) -> Counter[int] | None:
    return Counter(cards) if cards is not None else None


def deck_sig(cards: list[int] | None) -> str:
    if not cards:
        return ""
    compact = ",".join(f"{card}:{count}" for card, count in sorted(Counter(cards).items()))
    return hashlib.sha1(compact.encode("ascii")).hexdigest()[:12]


def ids_text(cards: list[int] | None) -> str:
    return " ".join(str(x) for x in cards) if cards else ""


def load_known_decks(paths: list[str], deck_dir: str) -> dict[str, str]:
    known: dict[str, str] = {}
    all_paths = [Path(p) for p in paths]
    if deck_dir:
        all_paths.extend(sorted(Path(deck_dir).glob("*.csv")))
    for path in all_paths:
        try:
            cards = read_deck(str(path))
        except Exception:
            continue
        known[deck_sig(cards)] = path.stem
    return known


def first_decks_from_replay(replay: dict) -> list[list[int] | None]:
    decks: list[list[int] | None] = [None, None]
    for step in replay.get("steps", [])[:8]:
        if not isinstance(step, list):
            continue
        for i, agent in enumerate(step[:2]):
            action = agent.get("action") if isinstance(agent, dict) else None
            if (
                isinstance(action, list)
                and len(action) == 60
                and all(isinstance(x, int) for x in action)
                and decks[i] is None
            ):
                decks[i] = list(action)
        if decks[0] is not None and decks[1] is not None:
            break
    return decks


def agent_names(replay: dict) -> list[str]:
    info = replay.get("info") or {}
    names = info.get("TeamNames")
    if isinstance(names, list) and len(names) >= 2:
        return [str(names[0]), str(names[1])]
    agents = info.get("Agents")
    if isinstance(agents, list) and len(agents) >= 2:
        out = []
        for i, agent in enumerate(agents[:2]):
            out.append(str((agent or {}).get("Name") or f"agent_{i}"))
        return out
    return ["agent_0", "agent_1"]


def choose_agent_index(
    replay: dict,
    args: argparse.Namespace,
    our_deck: list[int] | None,
    decks: list[list[int] | None],
) -> int:
    if args.agent_index is not None:
        if args.agent_index not in (0, 1):
            raise ValueError("--agent-index must be 0 or 1")
        return int(args.agent_index)

    names = agent_names(replay)
    if args.team_name:
        needle = args.team_name.lower()
        matches = [i for i, name in enumerate(names[:2]) if needle in name.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"--team-name matched both agents: {names}")

    if our_deck is not None:
        target = deck_counter(our_deck)
        matches = [i for i, cards in enumerate(decks) if cards is not None and deck_counter(cards) == target]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError("our deck matched both agents; pass --agent-index")

    if args.assume_agent_index is not None:
        return int(args.assume_agent_index)

    raise ValueError(
        "could not identify our agent. Pass --agent-index, --team-name, --deck, "
        "or --assume-agent-index."
    )


def final_current(replay: dict, agent_index: int) -> dict:
    steps = replay.get("steps") or []
    for step in reversed(steps):
        if not isinstance(step, list) or agent_index >= len(step):
            continue
        obs = (step[agent_index].get("observation") or {}) if isinstance(step[agent_index], dict) else {}
        cur = obs.get("current")
        if isinstance(cur, dict):
            return cur
    return {}


def analyze_one(
    submission_id: str,
    episode: dict,
    replay_file: Path,
    args: argparse.Namespace,
    our_deck: list[int] | None,
    known_decks: dict[str, str],
) -> dict:
    replay = json.loads(replay_file.read_text())
    decks = first_decks_from_replay(replay)
    idx = choose_agent_index(replay, args, our_deck, decks)
    opp = 1 - idx
    names = agent_names(replay)
    rewards = replay.get("rewards") or [None, None]
    statuses = replay.get("statuses") or ["", ""]
    reward = float(rewards[idx]) if rewards[idx] is not None else 0.0
    opp_reward = float(rewards[opp]) if rewards[opp] is not None else 0.0
    won = int(reward > opp_reward)
    draw = int(reward == opp_reward)
    cur = final_current(replay, idx)
    first_player = cur.get("firstPlayer", "")
    our_first = int(first_player == idx) if first_player in (0, 1) else ""
    our_cards = decks[idx]
    opp_cards = decks[opp]
    our_sig = deck_sig(our_cards)
    opp_sig = deck_sig(opp_cards)
    return {
        "submission_id": submission_id,
        "episode_id": episode.get("id", ""),
        "create_time": episode.get("createTime", ""),
        "end_time": episode.get("endTime", ""),
        "episode_state": episode.get("state", ""),
        "episode_type": episode.get("type", ""),
        "agent_index": idx,
        "opponent_index": opp,
        "team_name": names[idx],
        "opponent_team": names[opp],
        "reward": reward,
        "opponent_reward": opp_reward,
        "won": won,
        "draw": draw,
        "status": statuses[idx] if idx < len(statuses) else "",
        "opponent_status": statuses[opp] if opp < len(statuses) else "",
        "steps": len(replay.get("steps") or []),
        "first_player": first_player,
        "our_first": our_first,
        "our_deck_name": known_decks.get(our_sig, ""),
        "opponent_deck_name": known_decks.get(opp_sig, ""),
        "our_deck_sig": our_sig,
        "opponent_deck_sig": opp_sig,
        "our_deck_ids": ids_text(our_cards),
        "opponent_deck_ids": ids_text(opp_cards),
        "replay_path": str(replay_file),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def summary_rows(rows: list[dict], group_field: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_field) or "<unknown>")
        if group_field == "opponent_deck_name" and key == "<unknown>":
            key = str(row.get("opponent_deck_sig") or "<unknown>")
        groups[key].append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        games = len(items)
        wins = sum(int(r["won"]) for r in items)
        draws = sum(int(r["draw"]) for r in items)
        losses = games - wins - draws
        reward = sum(float(r["reward"]) for r in items) / max(games, 1)
        steps = sum(int(r["steps"]) for r in items) / max(games, 1)
        out.append(
            {
                "group": key,
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": f"{wins / max(games, 1):.4f}",
                "avg_reward": f"{reward:.4f}",
                "avg_steps": f"{steps:.1f}",
            }
        )
    return out


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def print_summary(rows: list[dict], group_field: str, limit: int) -> None:
    games = len(rows)
    wins = sum(int(r["won"]) for r in rows)
    draws = sum(int(r["draw"]) for r in rows)
    losses = games - wins - draws
    print(
        f"\nOverall: games={games} wins={wins} losses={losses} draws={draws} "
        f"wr={wins / max(games, 1):.3f}",
        flush=True,
    )
    print(f"\nBy {group_field}:")
    for row in summary_rows(rows, group_field)[:limit]:
        print(
            f"  {row['group']}: n={row['games']} wr={float(row['win_rate']):.3f} "
            f"w/l/d={row['wins']}/{row['losses']}/{row['draws']} "
            f"avg_steps={row['avg_steps']}",
            flush=True,
        )


def write_unknown_opponent_decks(rows: list[dict], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    seen: set[str] = set()
    for row in rows:
        sig = str(row.get("opponent_deck_sig") or "")
        if not sig or sig in seen:
            continue
        seen.add(sig)
        cards = [int(x) for x in str(row.get("opponent_deck_ids") or "").split() if x]
        if len(cards) != 60:
            continue
        team = str(row.get("opponent_team") or "unknown")
        safe_team = "".join(c if c.isalnum() else "_" for c in team.lower()).strip("_")[:32]
        path = out_dir / f"opp_{sig}_{safe_team or 'unknown'}.csv"
        if path.exists():
            continue
        path.write_text("\n".join(str(x) for x in cards) + "\n")
        written += 1
    return written


def default_summary_path(out: str) -> Path:
    path = Path(out)
    return path.with_name(path.stem + "_summary.csv")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("submission_id", help="Kaggle submission id")
    p.add_argument("--team-name", default="", help="substring used to identify our agent in replays")
    p.add_argument("--agent-index", type=int, default=None, help="force our replay agent index, 0 or 1")
    p.add_argument("--assume-agent-index", type=int, default=None,
                   help="fallback index if team/deck matching cannot identify our agent")
    p.add_argument("--deck", default="", help="our submitted deck CSV; used for agent detection")
    p.add_argument("--known-decks-dir", default="decks", help="directory of deck CSVs for naming deck signatures")
    p.add_argument("--known-deck", action="append", default=[], help="extra known deck CSV; repeatable")
    p.add_argument("--cache-dir", default="logs/kaggle_replays", help="where replay/log files are cached")
    p.add_argument("--out", default="", help="episode-level CSV output")
    p.add_argument("--summary-out", default="", help="summary CSV output")
    p.add_argument("--group-by", default="opponent_deck_name",
                   choices=["opponent_deck_name", "opponent_team", "opponent_deck_sig", "episode_type"])
    p.add_argument("--max-episodes", type=int, default=0, help="limit newest episodes; 0 means all")
    p.add_argument("--download-logs", action="store_true", help="also cache Kaggle logs for both agents")
    p.add_argument("--write-opponent-decks", action="store_true",
                   help="write each unique opponent deck as a CSV for later local round-robin")
    p.add_argument("--opponent-decks-dir", default="logs/kaggle_opponent_decks",
                   help="output directory used with --write-opponent-decks")
    p.add_argument("--force-download", action="store_true", help="redownload cached replay files")
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()

    submission_id = str(args.submission_id)
    out = Path(args.out or f"logs/kaggle_replay_analysis_{submission_id}.csv")
    summary_out = Path(args.summary_out) if args.summary_out else default_summary_path(str(out))
    cache_dir = Path(args.cache_dir) / submission_id
    our_deck = read_deck(args.deck) if args.deck else None
    known_decks = load_known_decks(args.known_deck, args.known_decks_dir)
    if our_deck is not None:
        known_decks.setdefault(deck_sig(our_deck), Path(args.deck).stem)

    print(f"Fetching episodes for submission {submission_id}", flush=True)
    episodes = fetch_episodes(submission_id, args.timeout)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    print(f"Analyzing {len(episodes)} episodes; cache={cache_dir}", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for i, episode in enumerate(episodes, 1):
        episode_id = str(episode.get("id"))
        try:
            rp = download_replay(episode_id, cache_dir, args.timeout, args.force_download)
            if args.download_logs:
                download_logs(episode_id, cache_dir, args.timeout)
            rows.append(analyze_one(submission_id, episode, rp, args, our_deck, known_decks))
        except Exception as e:
            print(f"  episode {episode_id}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        if args.progress_every and (i == 1 or i % args.progress_every == 0 or i == len(episodes)):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-9)
            eta = (len(episodes) - i) / max(rate, 1e-9)
            print(f"  {i}/{len(episodes)} episodes {rate:.2f}/s eta={eta:.0f}s", flush=True)

    if not rows:
        raise SystemExit("no episodes were analyzed")
    write_rows(out, rows)
    groups = summary_rows(rows, args.group_by)
    write_summary(summary_out, groups)
    print_summary(rows, args.group_by, limit=20)
    if args.write_opponent_decks:
        n = write_unknown_opponent_decks(rows, Path(args.opponent_decks_dir) / submission_id)
        print(f"Wrote {n} new opponent deck CSVs under {Path(args.opponent_decks_dir) / submission_id}")
    print(f"\nWrote {out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"kaggle command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except subprocess.TimeoutExpired:
        print("kaggle command timed out", file=sys.stderr)
        sys.exit(124)
