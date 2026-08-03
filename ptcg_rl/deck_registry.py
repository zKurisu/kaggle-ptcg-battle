from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path


def deck_signature(cards: list[int]) -> str:
    compact = ",".join(f"{card}:{count}" for card, count in sorted(Counter(map(int, cards)).items()))
    return hashlib.sha1(compact.encode("ascii")).hexdigest()[:12]


def read_deck(path: str | Path) -> list[int]:
    with Path(path).open() as f:
        cards = [int(line.strip()) for line in f if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"deck must contain 60 cards: {path} has {len(cards)}")
    return cards


def signature_for_deck(path: str | Path) -> str:
    return deck_signature(read_deck(path))


def load_registry(path: str | Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            policy = row.get("policy_path") or row.get("policy") or ""
            if policy:
                rows[policy] = row
                rows[str(Path(policy))] = row
                try:
                    rows[str(Path(policy).resolve())] = row
                except Exception:
                    pass
                rows[Path(policy).name] = row
    return rows


def registry_deck_for_policy(registry_path: str | Path, policy_path: str | Path) -> str | None:
    rows = load_registry(registry_path)
    policy = str(policy_path)
    row = rows.get(policy) or rows.get(str(Path(policy))) or rows.get(Path(policy).name)
    if not row:
        return None
    return row.get("deck_path") or row.get("deck") or None
