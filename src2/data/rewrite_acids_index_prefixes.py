#!/usr/bin/env python3
"""
Rewrite embedded absolute paths inside ACIDS partition index text files.

Index files list one .pt path per line; older indexes used paths like:
  /home/tkimura4/data/datasets/ACIDS/individual_time_samples_one_sec/...

This script replaces a prefix on every line in train/val/test/pretrain index files
under a given ACIDS dataset root (default: all random_partition_index_* folders).

Usage (on a host where the index files are writable):
  python3 src2/data/rewrite_acids_index_prefixes.py
  python3 src2/data/rewrite_acids_index_prefixes.py --dry-run
  python3 src2/data/rewrite_acids_index_prefixes.py \\
      --from-prefix /home/tkimura4/data/datasets/ACIDS/ \\
      --to-prefix /data/tkimura4/data/datasets/ACIDS/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

INDEX_NAMES = frozenset(
    {"train_index.txt", "val_index.txt", "test_index.txt", "pretrain_index.txt"}
)


def collect_index_files(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.is_dir():
        raise FileNotFoundError(f"ACIDS root is not a directory: {root}")
    for p in root.rglob("*"):
        if p.is_file() and p.name in INDEX_NAMES:
            out.append(p)
    return sorted(out)


def rewrite_file(path: Path, old: str, new: str, dry_run: bool) -> tuple[int, bool]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if old not in text:
        return 0, False
    updated = text.replace(old, new)
    if not dry_run:
        path.write_text(updated, encoding="utf-8")
    return text.count(old), True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/tkimura4/data/datasets/ACIDS"),
        help="Directory tree to search for index files (default: %(default)s)",
    )
    parser.add_argument(
        "--from-prefix",
        default="/home/tkimura4/data/datasets/ACIDS/",
        help="Substring to replace (default: %(default)s)",
    )
    parser.add_argument(
        "--to-prefix",
        default="/data/tkimura4/data/datasets/ACIDS/",
        help="Replacement prefix (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing files",
    )
    args = parser.parse_args()
    old = args.from_prefix
    new = args.to_prefix
    if old == new:
        print("error: --from-prefix and --to-prefix are identical", file=sys.stderr)
        return 2

    files = collect_index_files(args.root)
    if not files:
        print(f"No index files (*{tuple(INDEX_NAMES)}) under {args.root}", file=sys.stderr)
        return 1

    total_replacements = 0
    changed_files = 0
    for path in files:
        n, did = rewrite_file(path, old, new, args.dry_run)
        if did:
            changed_files += 1
            total_replacements += n
            mode = "would update" if args.dry_run else "updated"
            print(f"{mode}: {path} ({n} occurrence(s))")

    print(
        f"Done: {changed_files} file(s) with replacements, "
        f"{total_replacements} total line substring match(es)."
    )
    if args.dry_run and changed_files:
        print("Re-run without --dry-run to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
