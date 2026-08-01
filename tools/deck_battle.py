#!/usr/bin/env python3
"""
Deck extraction + local deck-vs-deck evaluation.

Extract decks from Kaggle episode replays:
    python tools/deck_battle.py extract <replays_dir> --out decks/episode_pool/

Run deck-vs-deck battles:
    python tools/deck_battle.py battle <deck_a.csv> <deck_b.csv> --games 10
    python tools/deck_battle.py battle <deck_a.csv> <decks_dir/> --games 5
"""

import os, sys, json, random, time, hashlib
from pathlib import Path
from collections import Counter

_HERE = Path(__file__).resolve().parent
_WORKSPACE = _HERE.parent.parent  # workspace root
sys.path.insert(0, str(_WORKSPACE))


def read_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(l.strip()) for l in f if l.strip()]


# ═══════════════════════════════════════════════════════════════════════════
# Deck extraction from episode replays
# ═══════════════════════════════════════════════════════════════════════════

def extract_decks_from_replay(path: str) -> list[tuple[str, list[int]]]:
    """Extract both 60-card decks from a Kaggle episode replay JSON."""
    with open(path) as f:
        data = json.load(f)

    decks = []
    steps = data.get("steps", [])
    if not steps: return decks

    # Deck info is in step[0] → visualize → action
    step0 = steps[0]
    for player_data in step0:
        viz = player_data.get("visualize", [])
        if viz and isinstance(viz, list):
            action = viz[0].get("action", [])
            if isinstance(action, list) and len(action) == 2:
                for deck_ids in action:
                    if isinstance(deck_ids, list) and len(deck_ids) == 60:
                        h = hashlib.md5(str(sorted(deck_ids)).encode()).hexdigest()[:12]
                        decks.append((f"ep_{h}", deck_ids))
                break  # Only first visualize entry
    return decks


def extract_from_directory(replays_dir: str, max_decks: int = 200) -> dict[str, list[int]]:
    """Walk a directory of episode JSON files, extract unique decks."""
    unique: dict[str, list[int]] = {}
    files = list(Path(replays_dir).glob("*.json"))
    print(f"Scanning {len(files)} replay files...")

    for fp in files:
        try:
            for dhash, deck in extract_decks_from_replay(str(fp)):
                if dhash not in unique:
                    unique[dhash] = deck
            if max_decks and len(unique) >= max_decks:
                break
        except Exception:
            continue

    print(f"Found {len(unique)} unique decks from {len(files)} replays")
    return unique


def export_decks(decks: dict[str, list[int]], out_dir: str):
    """Write decks to CSV files. Also print a summary."""
    os.makedirs(out_dir, exist_ok=True)

    # Load card names for summary
    try:
        from cg.api import all_card_data
        cards = {c.cardId: c for c in all_card_data()}
    except Exception:
        cards = {}

    for dhash, deck in sorted(decks.items()):
        path = os.path.join(out_dir, f"{dhash}.csv")
        with open(path, "w") as f:
            for cid in deck:
                f.write(f"{cid}\n")

        # Print signature cards
        cnt = Counter(deck)
        sig = []
        for cid, count in cnt.most_common(8):
            card = cards.get(cid)
            if card and card.cardType in (5, 6): continue
            sig.append(f"{card.name if card else '#'+str(cid)}×{count}")
        print(f"  {dhash}: {', '.join(sig[:4])}")

    print(f"\nExported {len(decks)} decks to {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
# Deck-vs-deck battle
# ═══════════════════════════════════════════════════════════════════════════

def play_match(deck_a: list[int], deck_b: list[int],
               agent_a=None, agent_b=None) -> int:
    """Play one game: deck_a vs deck_b. Returns 0 (a wins), 1 (b wins), 2 (draw)."""
    from cg.game import battle_start, battle_select, battle_finish

    if agent_a is None:
        # Default: random agent
        def agent_a(obs):
            sel = obs.get("select")
            if sel is None: return list(deck_a)
            opts = sel.get("option", [])
            mc = sel.get("maxCount", 0)
            return random.sample(range(len(opts)), min(mc, len(opts))) if opts and mc > 0 else []
    if agent_b is None:
        def agent_b(obs):
            sel = obs.get("select")
            if sel is None: return list(deck_b)
            opts = sel.get("option", [])
            mc = sel.get("maxCount", 0)
            return random.sample(range(len(opts)), min(mc, len(opts))) if opts and mc > 0 else []

    obs, sd = battle_start(deck_a, deck_b)
    if obs is None:
        return -1  # error

    turn = 0
    while True:
        sel = obs.get("select")
        cur = obs.get("current", {})
        res = cur.get("result", -1)
        if res != -1:
            battle_finish()
            return res
        if sel is None: break
        cp = cur.get("yourIndex", 0)
        act = agent_a(obs) if cp == 0 else agent_b(obs)
        obs = battle_select(act)
        turn += 1
        if turn > 500: break
    battle_finish()
    return -1


def run_matchup(deck_a_path: str, deck_b_path: str, games: int = 20):
    """Run a matchup between two decks or a deck vs a pool."""
    deck_a = read_deck(deck_a_path)

    if os.path.isdir(deck_b_path):
        # A vs pool
        pool = [read_deck(os.path.join(deck_b_path, f))
                for f in sorted(os.listdir(deck_b_path)) if f.endswith(".csv")]
        print(f"Deck A vs pool of {len(pool)} decks, {games} games each...")
        total_wins = 0
        for i, opp in enumerate(pool):
            wins = 0
            for g in range(games):
                if g % 2 == 0:
                    res = play_match(deck_a, opp)
                    if res == 0: wins += 1
                else:
                    res = play_match(opp, deck_a)
                    if res == 1: wins += 1
            pct = wins / games * 100
            fname = os.path.basename(
                [f for f in sorted(os.listdir(deck_b_path)) if f.endswith(".csv")][i])
            print(f"  vs {fname[:30]}: {wins}/{games} ({pct:.0f}%)")
            total_wins += wins
        print(f"Overall: {total_wins}/{len(pool)*games} ({total_wins/(len(pool)*games)*100:.0f}%)")
    else:
        # A vs B
        deck_b = read_deck(deck_b_path)
        wins_a = wins_b = 0
        for g in range(games):
            if g % 2 == 0:
                res = play_match(deck_a, deck_b)
                if res == 0: wins_a += 1
                elif res == 1: wins_b += 1
            else:
                res = play_match(deck_b, deck_a)
                if res == 1: wins_a += 1
                elif res == 0: wins_b += 1
        print(f"Deck A wins: {wins_a}/{games} ({wins_a/games*100:.0f}%)")
        print(f"Deck B wins: {wins_b}/{games} ({wins_b/games*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "extract":
        replays_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        out_dir = "decks/episode_pool"
        args = sys.argv[3:]
        while args:
            if args[0] == "--out" and len(args) > 1:
                out_dir = args[1]; args = args[2:]
            else:
                args = args[1:]
        decks = extract_from_directory(replays_dir)
        if decks:
            export_decks(decks, out_dir)

    elif cmd == "battle":
        if len(sys.argv) < 4:
            print("Usage: python deck_battle.py battle <deck_a.csv> <deck_b_or_dir/> [--games 20]")
            sys.exit(1)
        games = 20
        args = sys.argv[3:]
        while args:
            if args[0] == "--games" and len(args) > 1:
                games = int(args[1]); args = args[2:]
            else:
                args = args[1:]
        run_matchup(sys.argv[2], sys.argv[3], games)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
