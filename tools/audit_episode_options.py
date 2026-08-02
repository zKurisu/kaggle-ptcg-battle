#!/usr/bin/env python3
"""Audit raw episode select.option fields before changing BC extraction."""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


CONTEXT_NAMES = {
    0: "MAIN", 1: "SETUP_ACTIVE", 2: "SETUP_BENCH", 3: "SWITCH", 4: "TO_ACTIVE",
    5: "TO_BENCH", 6: "TO_FIELD", 7: "TO_HAND", 8: "DISCARD", 9: "TO_DECK",
    10: "TO_DECK_BOTTOM", 11: "TO_PRIZE", 12: "NOT_MOVE", 13: "DAMAGE_COUNTER",
    14: "DAMAGE_COUNTER_ANY", 15: "DAMAGE", 16: "REMOVE_DAMAGE_COUNTER", 17: "HEAL",
    18: "EVOLVES_FROM", 19: "EVOLVES_TO", 20: "DEVOLVE", 21: "ATTACH_FROM",
    22: "ATTACH_TO", 23: "DETACH_FROM", 24: "LOOK", 25: "EFFECT_TARGET",
    26: "DISCARD_ENERGY_CARD", 27: "DISCARD_TOOL_CARD", 28: "SWITCH_ENERGY_CARD",
    29: "DISCARD_CARD_OR_ATTACHED_CARD", 30: "DISCARD_ENERGY", 31: "TO_HAND_ENERGY",
    32: "TO_DECK_ENERGY", 33: "SWITCH_ENERGY", 34: "SKILL_ORDER", 35: "ATTACK",
    36: "DISABLE_ATTACK", 37: "EVOLVE", 38: "DRAW_COUNT", 39: "DAMAGE_COUNTER_COUNT",
    40: "REMOVE_DAMAGE_COUNTER_COUNT", 41: "IS_FIRST", 42: "MULLIGAN", 43: "ACTIVATE",
    44: "FIRST_EFFECT", 45: "MORE_DEVOLVE", 46: "COIN_HEAD", 47: "AFFECT_SPECIAL_CONDITION",
    48: "RECOVER_SPECIAL_CONDITION",
}
OPT_NAMES = {
    0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD", 5: "ENERGY_CARD",
    6: "ENERGY", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 11: "DISCARD",
    12: "RETREAT", 13: "ATTACK", 14: "END", 15: "SKILL", 16: "SPECIAL_CONDITION",
}
NAME_TO_OPT = {v: k for k, v in OPT_NAMES.items()}


def _type_id(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return NAME_TO_OPT[value.upper()]


def _key_fields(option: dict) -> tuple[str, ...]:
    return tuple(sorted(k for k in option.keys() if k != "type"))


def _iter_zips(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.zip"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", help="episode zip or directory containing zips")
    parser.add_argument("--max-episodes", type=int, default=2000)
    parser.add_argument("--option-types", nargs="*", default=["PLAY", "ABILITY", "ATTACK", "SKILL", "CARD"])
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=200,
                        help="print progress every N episodes; 0 disables progress")
    args = parser.parse_args()

    zips = _iter_zips(Path(args.episodes))
    if not zips:
        raise FileNotFoundError(f"No episode zip files found: {args.episodes}")
    total_eps = 0
    for zip_path in zips:
        with zipfile.ZipFile(zip_path) as zf:
            total_eps += sum(1 for n in zf.namelist() if n.endswith(".json"))
    target_eps = min(args.max_episodes, total_eps)
    print(f"Auditing {target_eps} episodes from {len(zips)} zip files", flush=True)

    wanted = {_type_id(x) for x in args.option_types}
    key_counts = defaultdict(Counter)
    selected_key_counts = defaultdict(Counter)
    ctx_counts = Counter()
    opt_counts = Counter()
    selected_counts = Counter()
    examples = defaultdict(list)
    n_eps = n_obs = n_selected = 0
    t0 = time.time()

    for zip_path in zips:
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".json")]
            print(f"  {zip_path.name}: {len(names)} episodes", flush=True)
            for name in names:
                if n_eps >= args.max_episodes:
                    break
                n_eps += 1
                try:
                    data = json.loads(zf.read(name).decode("utf-8"))
                    steps = data.get("steps") or []
                except Exception:
                    continue
                pending = [None, None]
                for step in steps[1:]:
                    for pi, pd in enumerate(step[:2]):
                        if not isinstance(pd, dict):
                            continue
                        action = pd.get("action", [])
                        if pending[pi] is not None and isinstance(action, list) and len(action) != 60:
                            obs = pending[pi]
                            sel = obs.get("select") or {}
                            opts = sel.get("option") or []
                            ctx = int(sel.get("context", -1))
                            for idx in action:
                                if isinstance(idx, int) and 0 <= idx < len(opts):
                                    opt = opts[idx]
                                    ot = int(opt.get("type", -1))
                                    selected_counts[(ctx, ot)] += 1
                                    selected_key_counts[(ctx, ot)][_key_fields(opt)] += 1
                                    n_selected += 1
                            pending[pi] = None

                        obs = pd.get("observation")
                        obs = obs if isinstance(obs, dict) else None
                        sel = obs.get("select") if obs else None
                        if pd.get("status") == "ACTIVE" and sel and sel.get("option"):
                            pending[pi] = obs
                            n_obs += 1
                            ctx = int(sel.get("context", -1))
                            ctx_counts[ctx] += 1
                            for opt in sel.get("option") or []:
                                ot = int(opt.get("type", -1))
                                opt_counts[(ctx, ot)] += 1
                                key_counts[(ctx, ot)][_key_fields(opt)] += 1
                                if ot in wanted and len(examples[(ctx, ot)]) < args.examples:
                                    examples[(ctx, ot)].append(opt)
                if args.progress_every and n_eps % args.progress_every == 0:
                    elapsed = time.time() - t0
                    rate = n_eps / max(elapsed, 1e-9)
                    eta = max(target_eps - n_eps, 0) / max(rate, 1e-9)
                    print(
                        f"  {n_eps}/{target_eps} eps obs={n_obs} selected={n_selected} "
                        f"{rate:.1f} eps/s eta={eta:.0f}s",
                        flush=True,
                    )
            if n_eps >= args.max_episodes:
                break

    print(f"Done in {time.time() - t0:.1f}s", flush=True)
    print(f"Episodes: {n_eps}")
    print(f"Decision observations: {n_obs}")
    print(f"Selected option refs: {n_selected}")
    print("\nTop contexts:")
    for ctx, cnt in ctx_counts.most_common(30):
        print(f"  {ctx:2d} {CONTEXT_NAMES.get(ctx, '?'):<30} n={cnt}")

    print("\nTop context x option type:")
    for (ctx, ot), cnt in opt_counts.most_common(40):
        selected = selected_counts[(ctx, ot)]
        print(
            f"  ctx={ctx:2d} {CONTEXT_NAMES.get(ctx, '?'):<24} "
            f"opt={ot:2d} {OPT_NAMES.get(ot, '?'):<18} options={cnt:7d} selected={selected:6d}"
        )
        for keys, kcnt in key_counts[(ctx, ot)].most_common(3):
            print(f"      keys[{kcnt}]: {', '.join(keys) if keys else '(none)'}")
        if selected_key_counts[(ctx, ot)]:
            for keys, kcnt in selected_key_counts[(ctx, ot)].most_common(2):
                print(f"      selected_keys[{kcnt}]: {', '.join(keys) if keys else '(none)'}")

    print("\nExamples:")
    for (ctx, ot), opts in sorted(examples.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"  ctx={ctx} {CONTEXT_NAMES.get(ctx, '?')} opt={ot} {OPT_NAMES.get(ot, '?')}")
        for opt in opts:
            print(f"    {opt}")


if __name__ == "__main__":
    main()
