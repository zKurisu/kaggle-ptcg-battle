#!/usr/bin/env python3
"""Build a weighted local ladder opponent deck pool from episodes and replays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from tools.bc_extract_v2 import ARCHETYPES, classify, load_leaderboard_scores, score_band


MANIFEST_FIELDS = [
    "deck_sig",
    "deck_path",
    "source",
    "date",
    "team_name",
    "score",
    "score_band",
    "archetype",
    "games",
    "wins",
    "losses",
    "weight",
    "cards",
]


def deck_sig(cards: list[int]) -> str:
    compact = ",".join(f"{card}:{count}" for card, count in sorted(Counter(cards).items()))
    return hashlib.sha1(compact.encode("ascii")).hexdigest()[:12]


def safe_name(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip().lower()).strip("_")
    return text[:48] or "unknown"


def parse_date(path_or_name: str) -> str:
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", path_or_name)
    return m.group(1) if m else ""


def recency_weight(date: str, newest: str) -> float:
    if not date or not newest:
        return 1.0
    try:
        from datetime import date as date_cls

        d = date_cls.fromisoformat(date)
        n = date_cls.fromisoformat(newest)
        age = max((n - d).days, 0)
    except Exception:
        return 1.0
    return math.exp(-age / 5.0)


def score_weight(score: float) -> float:
    if score >= 1200:
        return 4.0
    if score >= 1100:
        return 3.0
    if score >= 1000:
        return 2.0
    if score >= 900:
        return 1.2
    if score >= 800:
        return 0.7
    return 0.35


def read_episode_zip(zip_path: Path, name_to_score: dict[str, float], newest_date: str,
                     progress_every: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    zdate = parse_date(zip_path.name)
    rw = recency_weight(zdate, newest_date)
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        for i, name in enumerate(names, 1):
            try:
                data = json.loads(zf.read(name).decode("utf-8"))
                steps = data.get("steps") or []
                if not steps:
                    continue
                decks_raw = steps[0][0].get("visualize", [{}])[0].get("action", [])
                if not isinstance(decks_raw, list) or len(decks_raw) != 2:
                    continue
                teams = (data.get("info") or {}).get("TeamNames") or []
                rewards = data.get("rewards") or [0, 0]
                for pi, cards in enumerate(decks_raw[:2]):
                    if not isinstance(cards, list) or len(cards) != 60:
                        continue
                    team = str(teams[pi]) if pi < len(teams) else ""
                    score = float(name_to_score.get(team, 0.0))
                    sig = deck_sig(cards)
                    row = out.setdefault(
                        sig,
                        {
                            "deck_sig": sig,
                            "cards": list(map(int, cards)),
                            "teams": Counter(),
                            "source": "episodes",
                            "date": zdate,
                            "score": 0.0,
                            "score_band": "600-699",
                            "archetype": classify(cards),
                            "games": 0,
                            "wins": 0,
                            "losses": 0,
                            "weight": 0.0,
                        },
                    )
                    row["games"] += 1
                    if rewards and pi < len(rewards):
                        if float(rewards[pi] or 0) > float(rewards[1 - pi] or 0):
                            row["wins"] += 1
                        elif float(rewards[pi] or 0) < float(rewards[1 - pi] or 0):
                            row["losses"] += 1
                    if team:
                        row["teams"][team] += 1
                    if score > row["score"]:
                        row["score"] = score
                        row["score_band"] = score_band(score)
                    row["weight"] += rw * score_weight(score)
            except Exception:
                continue
            if progress_every and (i == 1 or i % progress_every == 0 or i == len(names)):
                rate = i / max(time.time() - t0, 1e-9)
                print(
                    f"  {zip_path.name} {i}/{len(names)} eps decks={len(out)} {rate:.1f}/s",
                    flush=True,
                )
    return out


def read_deck_csv(path: Path) -> list[int] | None:
    try:
        cards = [int(x.strip()) for x in path.read_text().splitlines() if x.strip()]
    except Exception:
        return None
    return cards if len(cards) == 60 else None


def read_personal_loss_dirs(dirs: list[str], source_weight: float) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in dirs:
        for path in sorted(Path(d).glob("*.csv")):
            cards = read_deck_csv(path)
            if cards is None:
                continue
            sig = deck_sig(cards)
            team = path.stem
            row = out.setdefault(
                sig,
                {
                    "deck_sig": sig,
                    "cards": cards,
                    "teams": Counter(),
                    "source": "personal_loss",
                    "date": "",
                    "score": 0.0,
                    "score_band": "personal",
                    "archetype": classify(cards),
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "weight": 0.0,
                },
            )
            row["teams"][team] += 1
            row["games"] += 1
            row["weight"] += source_weight
    return out


def merge_rows(rows: list[dict[str, dict]]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for table in rows:
        for sig, row in table.items():
            dst = merged.setdefault(sig, dict(row))
            if dst is row:
                continue
            dst["games"] += row["games"]
            dst["wins"] += row["wins"]
            dst["losses"] += row["losses"]
            dst["weight"] += row["weight"]
            dst["teams"].update(row["teams"])
            if float(row["score"]) > float(dst["score"]):
                dst["score"] = row["score"]
                dst["score_band"] = row["score_band"]
            if dst["source"] != row["source"]:
                dst["source"] = "mixed"
            if row.get("date", "") > dst.get("date", ""):
                dst["date"] = row["date"]
    return merged


def write_deck(path: Path, cards: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in cards) + "\n")


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    deck_dir = out_dir / "decks"
    deck_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "pool_manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in rows:
            top_team = row["teams"].most_common(1)[0][0] if row["teams"] else ""
            deck_name = f"{row['deck_sig']}_{safe_name(row['archetype'])}_{safe_name(top_team)}.csv"
            deck_path = deck_dir / deck_name
            write_deck(deck_path, row["cards"])
            w.writerow(
                {
                    "deck_sig": row["deck_sig"],
                    "deck_path": str(deck_path),
                    "source": row["source"],
                    "date": row["date"],
                    "team_name": top_team,
                    "score": f"{float(row['score']):.1f}" if row["score"] else "",
                    "score_band": row["score_band"],
                    "archetype": row["archetype"],
                    "games": row["games"],
                    "wins": row["wins"],
                    "losses": row["losses"],
                    "weight": f"{float(row['weight']):.4f}",
                    "cards": " ".join(str(x) for x in row["cards"]),
                }
            )

    arch_path = out_dir / "archetype_stats.csv"
    stats: dict[str, dict] = defaultdict(lambda: {"decks": 0, "games": 0, "weight": 0.0})
    for row in rows:
        st = stats[row["archetype"]]
        st["decks"] += 1
        st["games"] += row["games"]
        st["weight"] += row["weight"]
    with arch_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["archetype", "decks", "games", "weight"])
        for arch, st in sorted(stats.items(), key=lambda kv: kv[1]["weight"], reverse=True):
            w.writerow([arch, st["decks"], st["games"], f"{st['weight']:.4f}"])
    print(f"Wrote {manifest}")
    print(f"Wrote {arch_path}")
    print(f"Wrote {len(rows)} deck CSVs under {deck_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-dir", default="raw_episode")
    p.add_argument("--out", default="logs/ladder_pool")
    p.add_argument("--lb-csv", default="")
    p.add_argument("--personal-loss-dir", action="append", default=[],
                   help="directory of exported opponent deck CSVs; repeatable")
    p.add_argument("--top", type=int, default=80)
    p.add_argument("--min-games", type=int, default=1)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--include-score-bands", nargs="*", default=[],
                   help="optional score bands to keep, e.g. 1200+ 1100-1199 1000-1099")
    p.add_argument("--personal-loss-weight", type=float, default=25.0)
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--workers", type=int, default=1, help="episode zip files to process concurrently")
    args = p.parse_args()

    name_to_score = load_leaderboard_scores(args.lb_csv or None)
    zips = sorted(Path(args.episodes_dir).glob("*.zip"))
    newest = max((parse_date(z.name) for z in zips), default="")
    print(
        f"Building ladder pool: zips={len(zips)} leaderboard_teams={len(name_to_score)} newest={newest}",
        flush=True,
    )
    tables = []
    if args.workers <= 1:
        for zpath in zips:
            print(f"\nProcessing {zpath.name}", flush=True)
            tables.append(read_episode_zip(zpath, name_to_score, newest, args.progress_every))
    else:
        workers = min(args.workers, len(zips))
        print(f"Processing with {workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(read_episode_zip, zpath, name_to_score, newest, args.progress_every): zpath
                for zpath in zips
            }
            done = 0
            t0 = time.time()
            for fut in as_completed(futs):
                zpath = futs[fut]
                tables.append(fut.result())
                done += 1
                print(
                    f"Finished {done}/{len(futs)} {zpath.name} in {time.time() - t0:.0f}s",
                    flush=True,
                )
    if args.personal_loss_dir:
        tables.append(read_personal_loss_dirs(args.personal_loss_dir, args.personal_loss_weight))
    rows = list(merge_rows(tables).values())
    if args.include_score_bands:
        keep = set(args.include_score_bands)
        rows = [r for r in rows if r["score_band"] in keep or r["source"] in ("personal_loss", "mixed")]
    rows = [
        r for r in rows
        if int(r["games"]) >= args.min_games
        and (float(r["score"]) >= args.min_score or r["source"] in ("personal_loss", "mixed"))
    ]
    rows.sort(key=lambda r: (float(r["weight"]), int(r["games"]), float(r["score"])), reverse=True)
    if args.top:
        rows = rows[: args.top]
    print(f"\nSelected {len(rows)} decks")
    for r in rows[:20]:
        team = r["teams"].most_common(1)[0][0] if r["teams"] else ""
        print(
            f"  {r['deck_sig']} w={float(r['weight']):.1f} games={r['games']} "
            f"score={r['score'] or '-'} {r['score_band']} {r['archetype']} {team}",
            flush=True,
        )
    write_outputs(rows, Path(args.out))


if __name__ == "__main__":
    main()
