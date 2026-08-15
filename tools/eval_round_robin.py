#!/usr/bin/env python3
"""Evaluate a local policy population with pairwise battles."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ptcg_rl.policy_loader import load_policy
from ptcg_rl.deck_registry import registry_deck_for_policy
from ptcg_rl.resource_planner import apply_rule_decision, make_rule_planner
from ptcg_rl.rule_overlay import RULE_MODES

_WORKER_A: "Entry | None" = None
_WORKER_B: "Entry | None" = None
_WORKER_USE_MCTS = False
_WORKER_SIMS = 48
_WORKER_TIME_BUDGET = 4.0
_WORKER_MAX_TURNS = 700


@dataclass
class Entry:
    name: str
    policy_path: str
    deck_path: str
    policy: object | None
    deck: list[int]
    rules: str = ""
    mcts: bool = False
    planner: object | None = None


def read_deck(path: str) -> list[int]:
    path = clean_shell_token(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"deck file not found: {path}")
    with open(path) as f:
        deck = [int(line.strip()) for line in f if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck must contain 60 cards: {path} has {len(deck)}")
    return deck


def clean_shell_token(value: str) -> str:
    """Undo common quote litter from unquoted shell variable expansion."""
    value = value.strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value.strip("'\"")


def clean_entry_name(value: str) -> str:
    value = clean_shell_token(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def legal_random(sel: dict) -> list[int]:
    opts = sel.get("option", [])
    mn = int(sel.get("minCount", 0))
    mc = int(sel.get("maxCount", 0))
    if not opts or mc <= 0:
        return []
    hi = min(mc, len(opts))
    lo = min(max(mn, 0), hi)
    k = random.randint(lo, hi)
    return random.sample(range(len(opts)), k) if k > 0 else []


def policy_action(entry: Entry, obs: dict, use_mcts: bool, sims: int, time_budget: float) -> list[int]:
    sel = obs.get("select") or {}
    opts = sel.get("option", [])
    mn = int(sel.get("minCount", 0))
    mc = int(sel.get("maxCount", 0))
    n = len(opts)
    if n == 0 or mc <= 0:
        return []
    if entry.policy is None:
        return legal_random(sel)

    try:
        if use_mcts or entry.mcts:
            picks = entry.policy.select_mcts(
                obs,
                entry.deck,
                sims=sims,
                time_budget=time_budget,
                update_history=False,
            )
        else:
            picks = entry.policy.select(obs, greedy=True, update_history=False)
    except Exception:
        picks = legal_random(sel)
        try:
            entry.policy.remember_decision(obs, picks)
        except Exception:
            pass
        return picks
    if entry.rules:
        try:
            picks = apply_rule_decision(obs, picks, entry.deck, mode=entry.rules, planner=entry.planner).action
        except Exception:
            pass

    picks = [p for p in picks if 0 <= p < n]
    picks = list(dict.fromkeys(picks))
    if mn <= len(picks) <= mc:
        picks = picks[:mc]
    else:
        picks = legal_random(sel)
    try:
        entry.policy.remember_decision(obs, picks)
    except Exception:
        pass
    return picks


def parse_entry(spec: str, default_deck: str, registry: str = "") -> tuple[str, str, str]:
    """Parse NAME=POLICY[:DECK]. POLICY can be 'random'."""
    spec = clean_shell_token(spec)
    if "=" in spec:
        name, rest = spec.split("=", 1)
    else:
        rest = spec
        name = "random" if rest == "random" else Path(rest).stem
    name = clean_entry_name(name)
    rest = clean_shell_token(rest)
    parts = rest.split(":", 1)
    policy_path = clean_shell_token(parts[0])
    if len(parts) == 2:
        deck_path = clean_shell_token(parts[1])
    elif registry and policy_path != "random":
        deck_path = registry_deck_for_policy(registry, policy_path) or default_deck
    else:
        deck_path = default_deck
    deck_path = clean_shell_token(deck_path)
    if not name:
        raise ValueError(f"empty entry name: {spec}")
    return name, policy_path, deck_path


def load_entries(specs: list[str], default_deck: str, include_random: bool,
                 registry: str = "", skip_bad_entries: bool = False,
                 rules_by_name: dict[str, str] | None = None,
                 mcts_by_name: set[str] | None = None) -> list[Entry]:
    entries: list[Entry] = []
    if include_random:
        specs = [f"random=random:{default_deck}"] + specs

    seen: set[str] = set()
    for spec in specs:
        name, policy_path, deck_path = parse_entry(spec, default_deck, registry)
        if name in seen:
            raise ValueError(f"duplicate entry name: {name}")
        seen.add(name)
        try:
            deck = read_deck(deck_path)
            policy = None if policy_path == "random" else load_policy(policy_path, device="cpu")
        except Exception as exc:
            if skip_bad_entries:
                print(f"Skipping bad entry {name}: {exc}", flush=True)
                continue
            raise
        entry_rules = (rules_by_name or {}).get(name, "")
        entries.append(Entry(
            name,
            policy_path,
            deck_path,
            policy,
            deck,
            entry_rules,
            name in (mcts_by_name or set()),
            make_rule_planner(entry_rules, deck),
        ))
    if len(entries) < 2:
        raise ValueError("need at least two entries; pass --include-random or multiple --entry values")
    return entries


def read_manifest_entry_specs(path: str, *, limit: int = 0, random_from_deck: bool = False) -> list[str]:
    """Read eval entries from a CSV manifest.

    Supports manifests produced by build_shadow_pool.py and candidate manifests
    that contain eval_entry/checkpoint_path/deck_path columns.
    """
    specs: list[str] = []
    seen_entries: set[str] = set()
    seen_names: dict[str, int] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if limit and len(specs) >= limit:
                break
            name = row.get("shadow_name") or row.get("name") or row.get("team_name") or row.get("deck_sig") or ""
            name = clean_entry_name(name)
            deck = (row.get("deck_path") or row.get("deck") or "").strip()
            policy = (row.get("policy_path") or row.get("checkpoint_path") or row.get("policy") or "").strip()
            if random_from_deck:
                if not deck:
                    continue
                entry = f"{name}=random:{deck}"
            else:
                entry = (row.get("eval_entry") or row.get("entry") or "").strip()
                if not entry:
                    if policy and deck:
                        entry = f"{name}={policy}:{deck}"
                    elif policy:
                        entry = f"{name}={policy}"
            if not entry or entry in seen_entries:
                continue
            seen_entries.add(entry)

            raw_name = entry.split("=", 1)[0] if "=" in entry else entry
            base_name = clean_entry_name(raw_name)
            count = seen_names.get(base_name, 0) + 1
            seen_names[base_name] = count
            unique_name = base_name if count == 1 else f"{base_name}_{count}"
            if unique_name != base_name:
                if "=" in entry:
                    entry = f"{unique_name}={entry.split('=', 1)[1]}"
                else:
                    entry = f"{unique_name}={entry}"
            specs.append(entry)
    return specs


def entry_payload(e: Entry) -> tuple[str, str, str, str, bool]:
    return e.name, e.policy_path, e.deck_path, e.rules, e.mcts


def entry_from_payload(payload: tuple[str, str, str, str, bool]) -> Entry:
    name, policy_path, deck_path, rules, mcts = payload
    deck = read_deck(deck_path)
    policy = None if policy_path == "random" else load_policy(policy_path, device="cpu")
    planner = make_rule_planner(rules, deck)
    return Entry(name, policy_path, deck_path, policy, deck, rules, bool(mcts), planner)


def play_game(a: Entry, b: Entry, swapped: bool, use_mcts: bool, sims: int,
              time_budget: float, max_turns: int, seed: int | None = None) -> int:
    """Return 0 if entry a wins, 1 if entry b wins, 2 draw/error."""
    from cg.game import battle_finish, battle_select, battle_start

    if seed is not None:
        random.seed(seed)
        np.random.seed(int(seed) & 0xFFFFFFFF)
    first, second = (b, a) if swapped else (a, b)
    for entry in (first, second):
        if entry.policy is not None and hasattr(entry.policy, "reset_history"):
            entry.policy.reset_history()
        if entry.planner is not None:
            entry.planner.reset(entry.deck)
    obs, sd = battle_start(first.deck, second.deck)
    if obs is None:
        return 2

    try:
        for _ in range(max_turns):
            cur = obs.get("current", {})
            res = int(cur.get("result", -1))
            if res != -1:
                if swapped:
                    return 1 if res == 0 else 0 if res == 1 else 2
                return res if res in (0, 1) else 2
            if obs.get("select") is None:
                return 2
            side = int(cur.get("yourIndex", 0))
            entry = first if side == 0 else second
            obs = battle_select(policy_action(entry, obs, use_mcts, sims, time_budget))
            if obs is None:
                return 2
        return 2
    finally:
        battle_finish()


def _init_match_worker(a_payload, b_payload, use_mcts: bool, sims: int,
                       time_budget: float, max_turns: int) -> None:
    global _WORKER_A, _WORKER_B, _WORKER_USE_MCTS, _WORKER_SIMS, _WORKER_TIME_BUDGET, _WORKER_MAX_TURNS
    _WORKER_A = entry_from_payload(a_payload)
    _WORKER_B = entry_from_payload(b_payload)
    _WORKER_USE_MCTS = use_mcts
    _WORKER_SIMS = sims
    _WORKER_TIME_BUDGET = time_budget
    _WORKER_MAX_TURNS = max_turns


def _play_match_worker(args) -> int:
    g, seed = args
    return play_game(
        _WORKER_A, _WORKER_B, swapped=bool(g % 2),
        use_mcts=_WORKER_USE_MCTS, sims=_WORKER_SIMS,
        time_budget=_WORKER_TIME_BUDGET, max_turns=_WORKER_MAX_TURNS,
        seed=seed,
    )


def _print_match_progress(a: Entry, b: Entry, done: int, games: int,
                          wins_a: int, wins_b: int, draws: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate = done / max(elapsed, 1e-9)
    eta = (games - done) / max(rate, 1e-9)
    print(
        f"  {a.name} vs {b.name}: {done}/{games} "
        f"{wins_a}-{wins_b}-{draws} wr={wins_a/done:.3f} "
        f"{rate:.2f} games/s eta={eta:.0f}s",
        flush=True,
    )


def play_matchup(a: Entry, b: Entry, games: int, use_mcts: bool, sims: int,
                 time_budget: float, max_turns: int, progress_every: int,
                 workers: int = 1, seed: int = 1, fresh_workers: bool = False) -> dict:
    wins_a = wins_b = draws = 0
    t0 = time.time()
    workers = max(1, min(int(workers), games))

    def record(done: int, res: int) -> None:
        nonlocal wins_a, wins_b, draws
        if res == 0:
            wins_a += 1
        elif res == 1:
            wins_b += 1
        else:
            draws += 1
        if progress_every and (done == 1 or done % progress_every == 0 or done == games):
            _print_match_progress(a, b, done, games, wins_a, wins_b, draws, t0)

    if workers == 1:
        for g in range(games):
            res = play_game(
                a, b, swapped=bool(g % 2), use_mcts=use_mcts, sims=sims,
                time_budget=time_budget, max_turns=max_turns, seed=seed + g,
            )
            record(g + 1, res)
    else:
        tasks = [(g, seed + g) for g in range(games)]
        ex_kwargs = {
            "max_workers": workers,
            "initializer": _init_match_worker,
            "initargs": (entry_payload(a), entry_payload(b), use_mcts, sims, time_budget, max_turns),
        }
        if fresh_workers:
            ex_kwargs["max_tasks_per_child"] = 1
        with ProcessPoolExecutor(**ex_kwargs) as ex:
            futs = [ex.submit(_play_match_worker, t) for t in tasks]
            for done, fut in enumerate(as_completed(futs), 1):
                record(done, fut.result())
    return {"a": a.name, "b": b.name, "wins_a": wins_a, "wins_b": wins_b, "draws": draws}


def print_matrix(entries: list[Entry], results: dict[tuple[int, int], dict], games: int) -> None:
    names = [e.name for e in entries]
    widths = [max(8, len(n)) for n in names]
    first_w = max(12, max(len(n) for n in names))
    print("\nWin-rate matrix: row beats column")
    print(" " * first_w + "  " + "  ".join(n.rjust(w) for n, w in zip(names, widths)))
    for i, row_name in enumerate(names):
        cells = []
        for j in range(len(names)):
            if i == j:
                cells.append("-".rjust(widths[j]))
            else:
                key = (min(i, j), max(i, j))
                if key not in results:
                    cells.append("?".rjust(widths[j]))
                    continue
                r = results[key]
                wins = r["wins_a"] if i < j else r["wins_b"]
                cells.append(f"{wins / games:.3f}".rjust(widths[j]))
        print(row_name.rjust(first_w) + "  " + "  ".join(cells))


def write_csv(path: str, entries: list[Entry], results: dict[tuple[int, int], dict], games: int) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "column", "games", "row_wins", "column_wins", "draws", "row_win_rate"])
        for i, a in enumerate(entries):
            for j, b in enumerate(entries):
                if i == j:
                    continue
                if (min(i, j), max(i, j)) not in results:
                    continue
                r = results[(min(i, j), max(i, j))]
                row_wins = r["wins_a"] if i < j else r["wins_b"]
                col_wins = r["wins_b"] if i < j else r["wins_a"]
                w.writerow([a.name, b.name, games, row_wins, col_wins, r["draws"], row_wins / games])
    print(f"\nWrote {out}", flush=True)


def has_cg_engine() -> bool:
    try:
        return importlib.util.find_spec("cg.game") is not None
    except ModuleNotFoundError:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entry", action="append", default=[],
                   help="NAME=POLICY[:DECK]. POLICY may be 'random'. Repeat for each agent.")
    p.add_argument("--manifest", action="append", default=[],
                   help="CSV with eval_entry, or policy/checkpoint + deck_path columns. Repeatable.")
    p.add_argument("--manifest-limit", type=int, default=0,
                   help="max entries read from each manifest; 0 means all")
    p.add_argument("--manifest-random", action="store_true",
                   help="use random:deck from manifest deck_path instead of policy entries")
    p.add_argument("--policy", action="append", default=[],
                   help="Shortcut for --entry basename=POLICY using --deck.")
    p.add_argument("--deck", default="deck.csv", help="default deck for entries without :DECK")
    p.add_argument("--registry", default="",
                   help="CSV mapping policy_path to deck_path for entries without explicit :DECK")
    p.add_argument("--include-random", action="store_true", help="prepend random=random:DECK")
    p.add_argument("--skip-bad-entries", action="store_true",
                   help="skip entries with missing/invalid decks or unloadable policies")
    p.add_argument("--candidate-only", action="store_true",
                   help="only evaluate entry 0 against every other entry; useful for ladder pools")
    p.add_argument("--games", type=int, default=50, help="games per pair")
    p.add_argument("--mcts", action="store_true", help="use NumpyPolicy.select_mcts for policy entries")
    p.add_argument("--mcts-sims", type=int, default=48)
    p.add_argument("--time-budget", type=float, default=4.0)
    p.add_argument("--mcts-entry", action="append", default=[],
                   help="enable MCTS only for one named entry; repeatable")
    p.add_argument("--max-turns", type=int, default=700)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel game worker processes per pair; each worker loads both policies once")
    p.add_argument("--fresh-workers", action="store_true",
                   help="start a fresh worker process for each game; slower but avoids engine state leakage")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rules-entry", action="append", default=[],
                   help="experimental rule overlay for one entry, e.g. candidate=conservative")
    p.add_argument("--out-csv", default="", help="optional pairwise matrix CSV output")
    args = p.parse_args()

    random.seed(args.seed)
    if not has_cg_engine():
        p.error(
            "cg.game not found. Run this in the Kaggle/remote repo with the cg engine "
            "available, or set PYTHONPATH to the parent directory containing cg/."
        )

    specs = list(args.entry)
    for manifest in args.manifest:
        specs.extend(
            read_manifest_entry_specs(
                manifest, limit=args.manifest_limit, random_from_deck=args.manifest_random
            )
        )
    for policy in args.policy:
        specs.append(f"{Path(policy).stem}={policy}")
    rules_by_name = {}
    for spec in args.rules_entry:
        if "=" not in spec:
            p.error(f"--rules-entry must be NAME=<mode>, where mode is one of: {', '.join(RULE_MODES)}")
        name, mode = spec.split("=", 1)
        mode = mode.strip()
        if mode not in RULE_MODES:
            p.error(f"--rules-entry mode must be one of: {', '.join(RULE_MODES)}")
        rules_by_name[clean_entry_name(name)] = mode
    mcts_by_name = {clean_entry_name(name) for name in args.mcts_entry}
    entries = load_entries(
        specs, args.deck, args.include_random, args.registry,
        skip_bad_entries=args.skip_bad_entries,
        rules_by_name=rules_by_name,
        mcts_by_name=mcts_by_name,
    )

    mode = f"MCTS sims={args.mcts_sims} budget={args.time_budget}s" if args.mcts else "greedy"
    if mcts_by_name:
        mode += f"+mcts_entry:{','.join(sorted(mcts_by_name))}"
    print(f"Round-robin: {len(entries)} entries, {args.games} games/pair, {mode}", flush=True)
    if int(args.workers) > 1:
        print(
            "WARNING: concurrent cg engine RR is for coarse screening only; "
            "use --workers 1 --fresh-workers and fixed trace losses for hard conclusions.",
            flush=True,
        )
    for e in entries:
        kind = "random" if e.policy is None else e.policy_path
        suffix = ""
        if e.rules:
            suffix += f" | rules={e.rules}"
        if e.mcts:
            suffix += " | mcts=on"
        print(f"  {e.name}: {kind} | deck={e.deck_path}{suffix}", flush=True)

    results: dict[tuple[int, int], dict] = {}
    if args.candidate_only:
        pair_indices = [(0, j) for j in range(1, len(entries))]
    else:
        pair_indices = [(i, j) for i in range(len(entries)) for j in range(i + 1, len(entries))]
    total_pairs = len(pair_indices)
    t0 = time.time()
    done_pairs = 0
    for i, j in pair_indices:
        done_pairs += 1
        print(f"\nPair {done_pairs}/{total_pairs}: {entries[i].name} vs {entries[j].name}", flush=True)
        results[(i, j)] = play_matchup(
            entries[i], entries[j], args.games, args.mcts, args.mcts_sims,
            args.time_budget, args.max_turns, args.progress_every,
            workers=args.workers, seed=args.seed + done_pairs * 100000,
            fresh_workers=args.fresh_workers,
        )
        elapsed = time.time() - t0
        print(f"Finished pair {done_pairs}/{total_pairs} in {elapsed:.0f}s total", flush=True)

    print_matrix(entries, results, args.games)
    if args.out_csv:
        write_csv(args.out_csv, entries, results, args.games)


if __name__ == "__main__":
    main()
