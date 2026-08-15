#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_WS = _REPO.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_WS))

from ptcg_rl.encoder import FastEncoder
from ptcg_rl.policy_loader import load_policy
from tools.v15_scripted_random_trace import (
    load_deck,
    play_game,
    replay_recorded_states,
    write_state_replay,
    write_trace,
)


def action_text(row: dict[str, Any]) -> str:
    return (
        f"{row.get('chosen_type_names', '')} {row.get('chosen_card_names', '')} "
        f"target={row.get('chosen_target_name', '')} rank={row.get('chosen_first_rank', '')}"
    ).strip()


def row_type(row: dict[str, Any]) -> str:
    txt = str(row.get("chosen_type_names") or "").strip()
    if not txt:
        return "-"
    return txt.split(",")[0].strip()


def write_summary(
    path: Path,
    *,
    baseline_policy: str,
    candidate_policy: str,
    deck: str,
    trace_path: Path,
    script_path: Path,
    replay_path: Path,
    record_meta: dict[str, Any],
    replay_meta: dict[str, Any],
    rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    changed_rows: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    by_type: dict[str, int] = {}
    for i, (orig, new, _obs) in enumerate(rows):
        if bool(new.get("_same_action", 0)):
            continue
        changed_rows.append((i, orig, new))
        key = f"{row_type(orig)} -> {row_type(new)}"
        by_type[key] = by_type.get(key, 0) + 1

    with path.open("w") as f:
        f.write("# v15 fixed-random loss comparison\n\n")
        f.write(f"baseline: `{baseline_policy}`\n\n")
        f.write(f"candidate: `{candidate_policy}`\n\n")
        f.write(f"deck: `{deck}`\n\n")
        f.write(
            f"recorded_game={record_meta.get('game')} seed={record_meta.get('seed')} "
            f"outcome={record_meta.get('outcome')} candidate_side={record_meta.get('candidate_side')} "
            f"steps={record_meta.get('steps')}\n\n"
        )
        f.write(
            f"snapshots={replay_meta.get('snapshots')} same={replay_meta.get('same')} "
            f"changed={replay_meta.get('changed')} state_history={replay_meta.get('state_history')}\n\n"
        )
        f.write(f"baseline_trace: `{trace_path}`\n\n")
        f.write(f"random_script: `{script_path}`\n\n")
        f.write(f"candidate_state_replay: `{replay_path}`\n\n")
        if replay_meta.get("first_change"):
            f.write(f"first_change={replay_meta.get('first_change')}\n\n")
        if by_type:
            f.write("## Changed action-type pairs\n\n")
            for key, count in sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0])):
                f.write(f"- {key}: {count}\n")
            f.write("\n")
        f.write("## First changed decisions\n\n")
        if not changed_rows:
            f.write("No behavior change on recorded candidate states.\n")
            return
        for i, orig, new in changed_rows[:25]:
            f.write(
                f"- snapshot={i} step={orig.get('step')} turn={orig.get('turn')} "
                f"original={action_text(orig)} new={action_text(new)}\n"
            )


def make_play_args(args: argparse.Namespace, policy: str) -> SimpleNamespace:
    deck_cards = load_deck(args.deck)
    opponent_deck_cards = load_deck(args.opponent_deck) if args.opponent_deck else list(deck_cards)
    return SimpleNamespace(
        policy=policy,
        deck=args.deck,
        opponent_deck=args.opponent_deck,
        deck_cards=deck_cards,
        opponent_deck_cards=opponent_deck_cards,
        max_turns=args.max_turns,
        strict_script=False,
    )


def find_or_load_script(args: argparse.Namespace, encoder: FastEncoder, out_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    script_path = out_dir / "baseline_first_loss_random_script.json"
    trace_path = out_dir / "baseline_first_loss_trace.md"
    if args.script_in:
        script = json.loads(Path(args.script_in).read_text())
        script_path.write_text(json.dumps(script, indent=2, ensure_ascii=False))
        trace_path.write_text(
            "# existing random script\n\n"
            f"Loaded from `{args.script_in}`. No baseline trace was regenerated.\n"
        )
        return script, script_path, trace_path

    play_args = make_play_args(args, args.baseline_policy)
    policy = load_policy(args.baseline_policy, device="cpu")
    counts = {"win": 0, "loss": 0, "draw": 0}
    t0 = time.time()
    selected: tuple[str, list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]] | None = None
    for g in range(args.games):
        outcome, trace, meta, _steps = play_game(
            play_args,
            game=g,
            seed=args.seed + g,
            policy=policy,
            encoder=encoder,
            script=None,
        )
        counts[outcome] += 1
        if args.progress_every and (g == 0 or (g + 1) % args.progress_every == 0):
            print(
                f"record {g + 1}/{args.games} win={counts['win']} loss={counts['loss']} "
                f"draw={counts['draw']} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        if args.target_outcome == "any" or outcome == args.target_outcome:
            selected = (outcome, trace, meta)
            break
    if selected is None:
        raise SystemExit(f"no game matched target_outcome={args.target_outcome}; counts={counts}")

    _outcome, trace, meta = selected
    write_trace(trace_path, play_args, meta, trace)
    script = dict(meta)
    script.pop("replay_notes", None)
    script_path.write_text(json.dumps(script, indent=2, ensure_ascii=False))
    print(f"Wrote baseline trace {trace_path}", flush=True)
    print(f"Wrote random script {script_path}", flush=True)
    return script, script_path, trace_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("baseline_policy")
    p.add_argument("candidate_policy")
    p.add_argument("--deck", required=True)
    p.add_argument("--opponent-deck", default="")
    p.add_argument("--script-in", default="", help="reuse an existing random-script JSON instead of searching a new loss")
    p.add_argument("--target-outcome", choices=["loss", "win", "draw", "any"], default="loss")
    p.add_argument("--state-history", choices=["original", "new", "none"], default="original")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = FastEncoder()
    script, script_path, trace_path = find_or_load_script(args, encoder, out_dir)

    candidate_policy = load_policy(args.candidate_policy, device="cpu")
    replay_args = SimpleNamespace(
        policy=args.candidate_policy,
        deck=args.deck,
        state_history=args.state_history,
    )
    rows, replay_meta = replay_recorded_states(replay_args, candidate_policy, encoder, script)
    replay_path = out_dir / "candidate_state_replay.md"
    write_state_replay(replay_path, replay_args, replay_meta, rows)

    summary_path = out_dir / "summary.md"
    write_summary(
        summary_path,
        baseline_policy=args.baseline_policy,
        candidate_policy=args.candidate_policy,
        deck=args.deck,
        trace_path=trace_path,
        script_path=script_path,
        replay_path=replay_path,
        record_meta=script,
        replay_meta=replay_meta,
        rows=rows,
    )
    print(
        f"V15_RANDOM_LOSS_COMPARE snapshots={replay_meta.get('snapshots')} "
        f"same={replay_meta.get('same')} changed={replay_meta.get('changed')} "
        f"summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    os.chdir(_REPO)
    main()
