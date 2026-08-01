#!/usr/bin/env python3
"""
Extract deck lists from Kaggle daily episode replay datasets.

Each episode replay JSON includes the exact 60 Card IDs used by both players.
No mapping needed — these are native engine Card IDs.

Usage:
    # Download latest day's episodes and extract decks
    kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-07-30
    unzip pokemon-tcg-ai-battle-episodes-2026-07-30.zip -d episodes/
    python tools/extract_episode_decks.py episodes/ --min-score 1100 --out decks/from_episodes/

    # Output: one deck_NNN.csv per unique deck found
"""

import os, sys, json, hashlib
from pathlib import Path
from collections import Counter


def extract_decks_from_replay(replay_path: str) -> list[tuple[str, list[int]]]:
    """Extract both decks from a replay JSON. Returns [(deck_hash, [card_ids]), ...]."""
    with open(replay_path) as f:
        data = json.load(f)

    decks = []
    # Replays have 'configuration.decks' or 'info.decks' depending on version
    cfg = data.get("configuration") or data.get("info") or {}
    deck_list = cfg.get("decks", [])

    if isinstance(deck_list, list) and len(deck_list) >= 2:
        for i, deck in enumerate(deck_list[:2]):
            if isinstance(deck, list) and len(deck) == 60:
                # Hash for deduplication
                h = hashlib.md5(str(sorted(deck)).encode()).hexdigest()[:12]
                decks.append((f"deck_{h}", deck))

    return decks


def extract_from_directory(episodes_dir: str, min_score: int = 0,
                           max_decks: int = 100) -> dict[str, list[int]]:
    """Walk episode JSON files, extract unique decks above min_score."""
    unique_decks: dict[str, list[int]] = {}
    total_episodes = 0
    episodes_with_score = 0

    for root, dirs, files in os.walk(episodes_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            try:
                decks = extract_decks_from_replay(path)
                total_episodes += 1

                # Check score (embedded in replay or path)
                file_score = _parse_score(fname, path)

                if min_score > 0 and file_score < min_score:
                    continue

                episodes_with_score += 1
                for dhash, deck in decks:
                    if dhash not in unique_decks:
                        unique_decks[dhash] = deck

                if max_decks and len(unique_decks) >= max_decks:
                    break
            except Exception:
                continue

        if max_decks and len(unique_decks) >= max_decks:
            break

    print(f"Scanned {total_episodes} episodes, "
          f"{episodes_with_score} above score {min_score}, "
          f"{len(unique_decks)} unique decks")
    return unique_decks


def _parse_score(fname: str, path: str) -> float:
    """Try to extract a skill rating from the replay file. Returns 0 if unknown."""
    try:
        with open(path) as f:
            data = json.load(f)
        # Some replays have 'steps' with final ratings
        steps = data.get("steps", [])
        if steps:
            last = steps[-1]
            for player in last:
                if isinstance(player, dict):
                    rating = player.get("rating")
                    if rating: return float(rating)
    except Exception:
        pass
    return 0.0


def export_decks(decks: dict[str, list[int]], out_dir: str):
    """Write each deck to deck_N.csv."""
    os.makedirs(out_dir, exist_ok=True)
    for i, (dhash, card_ids) in enumerate(sorted(decks.items())):
        path = os.path.join(out_dir, f"{dhash}.csv")
        with open(path, "w") as f:
            for cid in card_ids:
                f.write(f"{cid}\n")
    print(f"Exported {len(decks)} decks to {out_dir}/")


def show_deck_summary(decks: dict[str, list[int]], top_n: int = 10):
    """Print a summary of the most common archetypes."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    try:
        from cg.api import all_card_data
        cards = {c.cardId: c for c in all_card_data()}
    except Exception:
        cards = {}

    print(f"\n=== Deck Summary ===")
    for dhash, deck in list(decks.items())[:top_n]:
        cnt = Counter(deck)
        # Find the most distinctive card (highest count among non-energy)
        signature = []
        for cid, count in cnt.most_common(5):
            card = cards.get(cid)
            if card and card.cardType in (5, 6):
                continue  # skip basic energy
            name = card.name if card else f"#{cid}"
            signature.append(f"{name}×{count}")
        print(f"  {dhash}: {', '.join(signature[:3])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_episode_decks.py <episodes_dir> [--min-score 1100] [--out decks/]")
        print("")
        print("First download the episodes dataset:")
        print("  kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-07-30")
        print("  unzip pokemon-tcg-ai-battle-episodes-2026-07-30.zip -d episodes/")
        sys.exit(1)

    episodes_dir = sys.argv[1]
    min_score = 0
    out_dir = "decks/from_episodes"

    args = sys.argv[2:]
    while args:
        if args[0] == "--min-score" and len(args) > 1:
            min_score = int(args[1]); args = args[2:]
        elif args[0] == "--out" and len(args) > 1:
            out_dir = args[1]; args = args[2:]
        else:
            args = args[1:]

    decks = extract_from_directory(episodes_dir, min_score=min_score)
    if decks:
        show_deck_summary(decks)
        export_decks(decks, out_dir)
        print(f"\nAdd to opponent pool: cp {out_dir}/*.csv ../decks/")
