#!/usr/bin/env python3
"""Append Kaggle submission score snapshots to a local CSV log."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "snapshot_utc",
    "ref",
    "fileName",
    "submissionDate",
    "description",
    "status",
    "publicScore",
    "privateScore",
]


def fetch_submissions(competition: str, page_size: int) -> list[dict]:
    cmd = [
        "kaggle", "competitions", "submissions", competition,
        "--format", "json",
        "--page-size", str(page_size),
    ]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    text = proc.stdout.strip()
    start = min([i for i in (text.find("["), text.find("{")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError(f"no JSON object found in Kaggle output: {text[:200]}")
    data = json.loads(text[start:])
    if not isinstance(data, list):
        raise ValueError("unexpected Kaggle submissions JSON")
    return data


def read_history(path: Path) -> dict[str, list[dict]]:
    history: dict[str, list[dict]] = defaultdict(list)
    if not path.exists():
        return history
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            history[str(row.get("ref", ""))].append(row)
    return history


def score_value(value: str) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def append_snapshot(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: list[dict], history: dict[str, list[dict]]) -> None:
    print(f"Recorded {len(rows)} submissions")
    for row in rows:
        ref = str(row["ref"])
        score = score_value(row["publicScore"])
        prev = None
        for old in reversed(history.get(ref, [])):
            old_score = score_value(old.get("publicScore", ""))
            if old_score is not None:
                prev = old_score
                break
        delta = ""
        if score is not None and prev is not None:
            delta = f" ({score - prev:+.1f})"
        elif score is not None:
            delta = " (new)"
        print(
            f"  {ref} score={row['publicScore'] or '-'}{delta} "
            f"status={row['status']} msg={row['description']}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--competition", default="pokemon-tcg-ai-battle")
    p.add_argument("--out", default="logs/kaggle_submission_scores.csv")
    p.add_argument("--page-size", type=int, default=50)
    p.add_argument("--no-append", action="store_true", help="print only; do not append CSV")
    args = p.parse_args()

    out = Path(args.out)
    history = read_history(out)
    snapshot = datetime.now(timezone.utc).isoformat(timespec="seconds")
    submissions = fetch_submissions(args.competition, args.page_size)
    rows = [
        {
            "snapshot_utc": snapshot,
            "ref": item.get("ref", ""),
            "fileName": item.get("fileName", ""),
            "submissionDate": item.get("date", ""),
            "description": item.get("description", ""),
            "status": item.get("status", ""),
            "publicScore": item.get("publicScore", ""),
            "privateScore": item.get("privateScore", ""),
        }
        for item in submissions
    ]

    print_summary(rows, history)
    if not args.no_append:
        append_snapshot(out, rows)
        print(f"Wrote {out}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"kaggle command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
