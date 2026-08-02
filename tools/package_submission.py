#!/usr/bin/env python3
"""Build a Kaggle submission tarball for the PTCG agent."""
import argparse
import os
import shutil
import tarfile
import tempfile
from pathlib import Path


def _copytree(src: Path, dst: Path):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", ".pytest_cache"))


def _default_cg_dir(repo: Path) -> Path | None:
    for p in (repo / "cg", repo.parent / "cg", repo.parent.parent / "cg"):
        if (p / "libcg.so").exists() or (p / "api.py").exists():
            return p
    return None


def main():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="trained .npz checkpoint")
    p.add_argument("--deck", default=str(repo / "deck.csv"))
    p.add_argument("--cg-dir", default=None,
                   help="path to cg engine directory; defaults to nearby ../cg")
    p.add_argument("--out", default=str(repo / "submission.tar.gz"))
    args = p.parse_args()

    policy = Path(args.policy).resolve()
    deck = Path(args.deck).resolve()
    cg_dir = Path(args.cg_dir).resolve() if args.cg_dir else _default_cg_dir(repo)
    out = Path(args.out).resolve()

    if not policy.exists():
        raise FileNotFoundError(f"policy not found: {policy}")
    if not deck.exists():
        raise FileNotFoundError(f"deck not found: {deck}")
    if cg_dir is None or not cg_dir.exists():
        raise FileNotFoundError("cg engine not found; pass --cg-dir /path/to/cg")

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Packaging submission", flush=True)
    print(f"  policy: {policy}", flush=True)
    print(f"  deck:   {deck}", flush=True)
    print(f"  cg:     {cg_dir}", flush=True)
    with tempfile.TemporaryDirectory(prefix="ptcg_submit_") as td:
        root = Path(td)
        print("  copying main.py/deck/policy", flush=True)
        shutil.copy2(repo / "main.py", root / "main.py")
        shutil.copy2(deck, root / "deck.csv")
        shutil.copy2(policy, root / "policy.npz")
        print("  copying ptcg_rl", flush=True)
        _copytree(repo / "ptcg_rl", root / "ptcg_rl")
        print("  copying cg engine", flush=True)
        _copytree(cg_dir, root / "cg")

        items = sorted(root.iterdir())
        print(f"  writing tarball with {len(items)} top-level entries", flush=True)
        with tarfile.open(out, "w:gz") as tar:
            for i, item in enumerate(items, 1):
                tar.add(item, arcname=item.name)
                print(f"    [{i}/{len(items)}] {item.name}", flush=True)

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Wrote {out} ({size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()
