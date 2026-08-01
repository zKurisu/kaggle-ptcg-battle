#!/usr/bin/env python3
"""
Convert PTCGL deck export text → Kaggle deck.csv (60 Card IDs).

Usage:
    python convert_deck.py <deck.txt> [output.csv]

Input format (PTCGL export, one card per line):
    4 Dragapult ex TWM 130
    2 Boss's Orders PAL 172
    4 Basic Fire Energy

Also supports simple format:
    4 Ultra Ball
    4 Rare Candy
"""

import os, sys, re
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_HERE)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)


class DeckConverter:
    def __init__(self, card_data_csv: str = None):
        if card_data_csv is None:
            card_data_csv = os.path.join(_WORKSPACE, "data", "EN_Card_Data.csv")
        self.df = self._load_cards(card_data_csv)
        self.composite_lookup: dict[tuple, int] = {}
        self.name_lookup: dict[str, int] = {}
        self._build_lookup()

    def _load_cards(self, path: str):
        """Load from CSV or from cg.api as fallback."""
        if os.path.exists(path):
            import pandas as pd
            return pd.read_csv(path)
        else:
            # Fallback: use engine API
            from cg.api import all_card_data
            cards = all_card_data()
            import pandas as pd
            rows = []
            for c in cards:
                rows.append({
                    "Card ID": c.cardId,
                    "Card Name": c.name,
                    "Expansion": "",
                    "Collection No.": "",
                    "Stage (Pokémon)/Type (Energy and Trainer)": "",
                })
            return pd.DataFrame(rows)

    def _build_lookup(self):
        for _, row in self.df.iterrows():
            cid = int(row["Card ID"])
            name = str(row["Card Name"]).strip()
            exp = str(row.get("Expansion", "")).strip().upper()
            num = str(row.get("Collection No.", "")).strip()

            if exp and num and exp != "NAN" and num != "NAN":
                self.composite_lookup[(exp, num)] = cid

            name_lower = name.lower()
            if name_lower not in self.name_lookup:
                self.name_lookup[name_lower] = cid

    def parse_line(self, line: str) -> dict | None:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            return None

        # "4 Dragapult ex TWM 130"
        m = re.match(r'^(\d+)\s+(.+?)\s+([A-Z0-9]+)\s+(\d+)$', line)
        if m:
            qty, name, exp, num = m.groups()
            return {"qty": int(qty), "name": name.strip(),
                    "expansion": exp.strip().upper(), "number": num.strip()}

        # "1 Basic Fire Energy" or "4 Ultra Ball"
        m = re.match(r'^(\d+)\s+(.+)$', line)
        if m:
            qty, name = m.groups()
            return {"qty": int(qty), "name": name.strip(),
                    "expansion": None, "number": None}
        return None

    def resolve(self, parsed: dict) -> int | None:
        """Resolve a parsed card entry to engine Card ID."""
        # 1. Composite key
        if parsed["expansion"] and parsed["number"]:
            cid = self.composite_lookup.get((parsed["expansion"], parsed["number"]))
            if cid: return cid

        # 2. Name match
        cid = self.name_lookup.get(parsed["name"].lower())
        if cid: return cid

        # 3. Fuzzy match (contains)
        name_lower = parsed["name"].lower()
        for k, v in self.name_lookup.items():
            if name_lower in k or k in name_lower:
                return v

        return None

    def convert(self, text: str, fallback_energy: int = 1) -> list[int]:
        """Convert PTCGL deck text → list of 60 Card IDs."""
        card_ids = []
        for line in text.strip().split("\n"):
            p = self.parse_line(line)
            if not p: continue
            cid = self.resolve(p)
            if cid:
                card_ids.extend([cid] * p["qty"])
            else:
                print(f"  [WARN] not found: {p['name']} ({p.get('expansion','')} {p.get('number','')})")

        # Pad/trim to 60
        if len(card_ids) < 60:
            missing = 60 - len(card_ids)
            print(f"  [INFO] padding {missing} slots with basic energy #{fallback_energy}")
            card_ids.extend([fallback_energy] * missing)
        elif len(card_ids) > 60:
            print(f"  [WARN] trimming {len(card_ids)-60} excess cards")
            card_ids = card_ids[:60]

        return card_ids

    def export(self, card_ids: list[int], path: str):
        assert len(card_ids) == 60
        with open(path, "w") as f:
            for cid in card_ids:
                f.write(f"{cid}\n")
        print(f"  → {path} ({len(card_ids)} cards)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_deck.py <deck.txt> [output.csv]")
        print("")
        print("Example deck.txt:")
        print("  4 Dragapult ex TWM 130")
        print("  2 Boss's Orders PAL 172")
        print("  4 Basic Fire Energy")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".txt", ".csv")

    with open(input_path) as f:
        text = f.read()

    converter = DeckConverter()
    card_ids = converter.convert(text)
    converter.export(card_ids, output_path)
