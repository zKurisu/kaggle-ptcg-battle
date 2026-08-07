#!/usr/bin/env python3
"""Build a teacher/clone manifest for stronger local validation.

The manifest ranks stable team+deck trajectories from extracted BC corpora and
emits shell-safe BC training commands for opponent clones. These clones are
intended to be stronger validation opponents than random:deck entries.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from ptcg_rl.bc2.data import discover_npz_paths
from ptcg_rl.deck_registry import signature_for_deck


DEFAULT_ARCHETYPES = [
    "Marnie Grimmsnarl",
    "Teal Mask Ogerpon",
    "Mega Lopunny",
    "Mega Lucario",
    "Alakazam",
    "Dragapult",
    "Festival Lead",
    "Crustle Wall",
    "Cynthia Garchomp",
    "Team Rocket Mewtwo",
    "Mega Abomasnow",
    "Mega Starmie",
    "Archaludon",
    "Hop Trevenant",
]

FIELDS = [
    "rank",
    "shadow_name",
    "trajectory_score",
    "archetype",
    "team_name",
    "deck_sig",
    "deck_path",
    "bands",
    "dates",
    "files",
    "episodes",
    "decisions",
    "wins",
    "losses",
    "draws",
    "decision_win_rate",
    "avg_score",
    "max_score",
    "first_date",
    "last_date",
    "opponent_filters",
    "arch",
    "init_path",
    "checkpoint_path",
    "eval_entry",
    "train_cmd",
]


def safe_name(text: str, limit: int = 40) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", text.strip().lower()).strip("_")
    return (text[:limit].strip("_") or "unknown")


def date_from_path(path: str | Path) -> str:
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", str(path))
    return m.group(1) if m else ""


def index_decks(dirs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for d in dirs:
        root = Path(d)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            try:
                sig = signature_for_deck(path)
            except Exception:
                continue
            out.setdefault(sig, str(path))
    return out


def scan_corpus(args: argparse.Namespace) -> list[dict]:
    archetypes = args.archetype or DEFAULT_ARCHETYPES
    opponent_deck_sigs = {str(x) for x in args.opponent_deck_sig}
    opponent_archetypes = {str(x).lower() for x in args.opponent_archetype}
    opponent_team_names = {str(x).lower() for x in args.opponent_team_name}
    paths: list[tuple[str, str]] = []
    for arch in archetypes:
        paths.extend((arch, p) for p in discover_npz_paths(args.corpus, arch, args.score_bands))
    if not paths:
        raise FileNotFoundError(f"no corpus files found under {args.corpus}")

    rows = defaultdict(lambda: {
        "bands": Counter(),
        "dates": Counter(),
        "files": Counter(),
        "episodes": set(),
        "decisions": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "score_sum": 0.0,
        "score_n": 0,
        "max_score": 0.0,
    })
    t0 = time.time()
    total_rows = 0
    for file_i, (arch, path) in enumerate(paths, 1):
        band = Path(path).parent.name.replace("_", " ")
        date = date_from_path(path)
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        for key in ("team_name", "deck_sig", "board"):
            if key not in data:
                raise ValueError(f"{path} lacks {key}; re-extract with current bc_extract_v2.py")
        if args.opponent_deck_sig and "opponent_deck_sig" not in data:
            raise ValueError(f"{path} lacks opponent_deck_sig; re-extract with current bc_extract_v2.py")
        if args.opponent_archetype and "opponent_archetype" not in data:
            raise ValueError(f"{path} lacks opponent_archetype; re-extract with current bc_extract_v2.py")
        if args.opponent_team_name and "opponent_team_name" not in data:
            raise ValueError(f"{path} lacks opponent_team_name; re-extract with current bc_extract_v2.py")
        n = len(data["board"])
        total_rows += n
        for i in range(n):
            if opponent_deck_sigs and str(data["opponent_deck_sig"][i]) not in opponent_deck_sigs:
                continue
            if (
                opponent_archetypes
                and str(data["opponent_archetype"][i]).lower() not in opponent_archetypes
            ):
                continue
            if (
                opponent_team_names
                and str(data["opponent_team_name"][i]).lower() not in opponent_team_names
            ):
                continue
            team = str(data["team_name"][i])
            sig = str(data["deck_sig"][i])
            if not team or not sig:
                continue
            row = rows[(arch, team, sig)]
            row["bands"][band] += 1
            if date:
                row["dates"][date] += 1
            row["files"][Path(path).name] += 1
            row["decisions"] += 1
            if "episode_id" in data:
                row["episodes"].add(str(data["episode_id"][i]))
            won = int(data["won"][i]) if "won" in data else 0
            draw = int(data["draw"][i]) if "draw" in data else 0
            row["wins"] += won
            row["draws"] += draw
            row["losses"] += int(won == 0 and draw == 0)
            if "score" in data:
                score = float(data["score"][i])
                row["score_sum"] += score
                row["score_n"] += 1
                row["max_score"] = max(float(row["max_score"]), score)
        if args.progress_every_files and (
            file_i == 1 or file_i % args.progress_every_files == 0 or file_i == len(paths)
        ):
            rate = total_rows / max(time.time() - t0, 1e-9)
            print(
                f"  files {file_i}/{len(paths)} rows={total_rows} "
                f"trajectories={len(rows)} {rate:.0f} rows/s",
                flush=True,
            )

    out = []
    for (arch, team, sig), row in rows.items():
        decisions = int(row["decisions"])
        episodes = len(row["episodes"])
        if decisions < args.min_decisions or episodes < args.min_episodes:
            continue
        dates = sorted(row["dates"])
        win_rate = int(row["wins"]) / max(decisions, 1)
        avg_score = float(row["score_sum"]) / max(int(row["score_n"]), 1)
        max_score = float(row["max_score"])
        score = trajectory_score(decisions, episodes, len(dates), win_rate, max_score)
        out.append({
            "trajectory_score": score,
            "archetype": arch,
            "team_name": team,
            "deck_sig": sig,
            "bands": " ".join(f"{k}:{v}" for k, v in row["bands"].most_common()),
            "dates": " ".join(dates),
            "files": len(row["files"]),
            "episodes": episodes,
            "decisions": decisions,
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "draws": int(row["draws"]),
            "decision_win_rate": win_rate,
            "avg_score": avg_score,
            "max_score": max_score,
            "first_date": dates[0] if dates else "",
            "last_date": dates[-1] if dates else "",
        })
    out.sort(key=lambda r: float(r["trajectory_score"]), reverse=True)
    return select_rows(out, args)


def trajectory_score(decisions: int, episodes: int, dates: int, win_rate: float, max_score: float) -> float:
    return (
        math.log1p(decisions)
        * (1.0 + 0.18 * max(dates - 1, 0))
        * (0.55 + win_rate)
        * (0.50 + min(max_score, 1300.0) / 1300.0)
        * (1.0 + min(episodes, 500) / 1000.0)
    )


def select_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    selected: list[dict] = []
    by_arch: Counter[str] = Counter()
    by_sig: Counter[str] = Counter()
    seen_team_sig: set[tuple[str, str]] = set()
    for row in rows:
        if args.top and len(selected) >= args.top:
            break
        arch = str(row["archetype"])
        sig = str(row["deck_sig"])
        team = str(row["team_name"])
        if args.top_per_archetype and by_arch[arch] >= args.top_per_archetype:
            continue
        if args.max_per_deck_sig and by_sig[sig] >= args.max_per_deck_sig:
            continue
        if (team, sig) in seen_team_sig:
            continue
        selected.append(row)
        by_arch[arch] += 1
        by_sig[sig] += 1
        seen_team_sig.add((team, sig))
    return selected


def train_cmd(row: dict, args: argparse.Namespace, checkpoint_path: str) -> str:
    init_path = render_init_path(row, args)
    cmd = [
        "python3", "-u", "tools/bc2_train.py",
        "--corpus", args.corpus,
        "--archetype", str(row["archetype"]),
        "--score-bands", *args.score_bands,
        "--deck-sig", str(row["deck_sig"]),
        "--team-name", str(row["team_name"]),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--width", str(args.width),
        "--arch", args.arch,
        "--state-layers", str(args.state_layers),
        "--device", args.device,
        "--cuda-memory-gb", str(args.cuda_memory_gb),
        "--cuda-memory-fraction", str(args.cuda_memory_fraction),
        "--first-action-weight", str(args.first_action_weight),
        "--value-weight", str(args.value_weight),
        "--set-loss-weight", str(args.set_loss_weight),
        "--set-loss-min-count", str(args.set_loss_min_count),
        "--set-loss-negative-weight", str(args.set_loss_negative_weight),
        "--option-weight", str(args.option_weight),
        "--win-weight", str(args.win_weight),
        "--loss-weight", str(args.loss_weight),
        "--draw-weight", str(args.draw_weight),
        "--multi-select-weight", str(args.multi_select_weight),
        "--checkpoint-every", str(args.checkpoint_every),
        "--save", checkpoint_path,
    ]
    if init_path:
        cmd.extend(["--init", init_path])
        if args.init_partial:
            cmd.append("--init-partial")
    if args.state_feat_dim:
        cmd.extend(["--state-feat-dim", str(args.state_feat_dim)])
    if args.opt_feat_dim:
        cmd.extend(["--opt-feat-dim", str(args.opt_feat_dim)])
    for value in args.opponent_deck_sig:
        cmd.extend(["--opponent-deck-sig", value])
    for value in args.opponent_archetype:
        cmd.extend(["--opponent-archetype", value])
    for value in args.opponent_team_name:
        cmd.extend(["--opponent-team-name", value])
    for spec in args.context_weight:
        cmd.extend(["--context-weight", spec])
    for spec in args.type_weight:
        cmd.extend(["--type-weight", spec])
    if args.legacy_state_pool:
        cmd.append("--legacy-state-pool")
    if args.include_empty:
        cmd.append("--include-empty")
    return " ".join(shlex.quote(x) for x in cmd)


def render_opponent_filters(args: argparse.Namespace) -> str:
    parts = []
    parts.extend(f"deck_sig={x}" for x in args.opponent_deck_sig)
    parts.extend(f"archetype={x}" for x in args.opponent_archetype)
    parts.extend(f"team={x}" for x in args.opponent_team_name)
    return " ".join(parts)


def render_init_path(row: dict, args: argparse.Namespace) -> str:
    if not args.init_template:
        return ""
    return args.init_template.format(
        archetype=str(row["archetype"]),
        archetype_slug=safe_name(str(row["archetype"]), 80),
        team_name=str(row["team_name"]),
        team_slug=safe_name(str(row["team_name"]), 80),
        deck_sig=str(row["deck_sig"]),
        deck_sig8=str(row["deck_sig"])[:8],
        width=f"{args.width:g}",
    )


def write_manifest(rows: list[dict], args: argparse.Namespace) -> None:
    deck_paths = index_decks(args.known_decks_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen_shadow_names: set[str] = set()
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            shadow = (
                f"shadow_{safe_name(row['archetype'], 22)}_"
                f"{str(row['deck_sig'])[:8]}_{safe_name(row['team_name'], 24)}"
            )
            if shadow in seen_shadow_names:
                shadow = f"{shadow}_{rank:03d}"
            seen_shadow_names.add(shadow)
            suffix = f"w{args.width:g}" if args.arch == "pointer" else f"{args.arch}_w{args.width:g}"
            checkpoint = str(Path(args.checkpoint_dir) / f"{shadow}_{suffix}.npz")
            init_path = render_init_path(row, args)
            deck_path = deck_paths.get(str(row["deck_sig"]), "")
            eval_entry = f"{shadow}={checkpoint}:{deck_path}" if deck_path else ""
            writer.writerow({
                "rank": rank,
                "shadow_name": shadow,
                "trajectory_score": f"{float(row['trajectory_score']):.4f}",
                "archetype": row["archetype"],
                "team_name": row["team_name"],
                "deck_sig": row["deck_sig"],
                "deck_path": deck_path,
                "bands": row["bands"],
                "dates": row["dates"],
                "files": row["files"],
                "episodes": row["episodes"],
                "decisions": row["decisions"],
                "wins": row["wins"],
                "losses": row["losses"],
                "draws": row["draws"],
                "decision_win_rate": f"{float(row['decision_win_rate']):.4f}",
                "avg_score": f"{float(row['avg_score']):.1f}",
                "max_score": f"{float(row['max_score']):.1f}",
                "first_date": row["first_date"],
                "last_date": row["last_date"],
                "opponent_filters": render_opponent_filters(args),
                "arch": args.arch,
                "init_path": init_path,
                "checkpoint_path": checkpoint,
                "eval_entry": eval_entry,
                "train_cmd": train_cmd(row, args, checkpoint),
            })
    print(f"Wrote {out}: {len(rows)} rows")
    missing = sum(1 for row in rows if str(row["deck_sig"]) not in deck_paths)
    if missing:
        print(f"WARNING: {missing}/{len(rows)} rows have no deck_path in known deck dirs", flush=True)
    print("\nTop shadow rows:")
    for row in rows[: args.print_top]:
        print(
            f"  {row['archetype']:<20} sig={row['deck_sig']} "
            f"dec={int(row['decisions']):7d} eps={int(row['episodes']):4d} "
            f"wr={float(row['decision_win_rate']):.2f} max={float(row['max_score']):.1f} "
            f"team={str(row['team_name'])[:32]}",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/bc_corpus_banded_v10_all_0803")
    p.add_argument("--archetype", action="append", default=[],
                   help="repeatable; omit to scan common meta archetypes")
    p.add_argument("--score-bands", nargs="+",
                   default=["1200+", "1100-1199", "1000-1099", "900-999"])
    p.add_argument("--min-decisions", type=int, default=3000)
    p.add_argument("--min-episodes", type=int, default=20)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--top-per-archetype", type=int, default=4)
    p.add_argument("--max-per-deck-sig", type=int, default=2)
    p.add_argument("--known-decks-dir", action="append", default=["logs/ladder_pool_0802_all/decks", "decks"],
                   help="deck CSV directories used to resolve deck_sig -> deck_path")
    p.add_argument("--checkpoint-dir", default="checkpoints/shadow")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--width", type=float, default=2.0)
    p.add_argument("--arch", choices=["pointer", "cross_attn"], default="pointer")
    p.add_argument("--state-layers", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cuda-memory-gb", type=float, default=0.0,
                   help="emit an approximate GiB CUDA allocator cap for each shadow train command")
    p.add_argument("--cuda-memory-fraction", type=float, default=0.0,
                   help="emit a visible-GPU CUDA allocator fraction cap for each shadow train command")
    p.add_argument("--init-template", default="",
                   help="optional format string for specialist init, e.g. checkpoints/pop/bc2_{archetype_slug}_tag.npz")
    p.add_argument("--init-partial", action="store_true",
                   help="pass --init-partial to specialist train commands")
    p.add_argument("--include-empty", action="store_true")
    p.add_argument("--first-action-weight", type=float, default=1.5)
    p.add_argument("--value-weight", type=float, default=0.0)
    p.add_argument("--set-loss-weight", type=float, default=0.0)
    p.add_argument("--set-loss-min-count", type=int, default=2)
    p.add_argument("--set-loss-negative-weight", type=float, default=0.25)
    p.add_argument("--option-weight", type=float, default=0.15)
    p.add_argument("--win-weight", type=float, default=1.5)
    p.add_argument("--loss-weight", type=float, default=0.4)
    p.add_argument("--draw-weight", type=float, default=0.8)
    p.add_argument("--multi-select-weight", type=float, default=1.0)
    p.add_argument("--context-weight", action="append", default=[])
    p.add_argument("--type-weight", action="append", default=[])
    p.add_argument("--legacy-state-pool", action="store_true")
    p.add_argument("--state-feat-dim", type=int, default=0,
                   help="override state feature width in emitted train commands")
    p.add_argument("--opt-feat-dim", type=int, default=0,
                   help="override per-option feature width in emitted train commands")
    p.add_argument("--opponent-deck-sig", action="append", default=[],
                   help="append opponent deck-signature filters to emitted train commands")
    p.add_argument("--opponent-archetype", action="append", default=[],
                   help="append opponent archetype filters to emitted train commands")
    p.add_argument("--opponent-team-name", action="append", default=[],
                   help="append opponent team-name filters to emitted train commands")
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--progress-every-files", type=int, default=10)
    p.add_argument("--print-top", type=int, default=20)
    p.add_argument("--out", default="logs/shadow_pool_manifest.csv")
    args = p.parse_args()

    rows = scan_corpus(args)
    write_manifest(rows, args)


if __name__ == "__main__":
    main()
