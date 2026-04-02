#!/usr/bin/env python3
"""Pre-demo health checks for exported data and runtime dependencies."""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate demo prerequisites before running.")
    parser.add_argument("--labels_csv", type=Path, required=True)
    parser.add_argument("--samples_csv", type=Path, default=None)
    parser.add_argument("--require_web", action="store_true", help="Fail if Flask dependency is unavailable")
    return parser.parse_args()


def _count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _check_import(mod_name: str) -> bool:
    try:
        importlib.import_module(mod_name)
        return True
    except Exception:
        return False


def main() -> None:
    args = _parse_args()
    if not args.labels_csv.exists():
        raise FileNotFoundError(f"Missing labels csv: {args.labels_csv}")
    label_rows = _count_rows(args.labels_csv)
    if label_rows == 0:
        raise ValueError("Labels CSV has zero rows.")

    sample_rows = None
    if args.samples_csv is not None:
        if not args.samples_csv.exists():
            raise FileNotFoundError(f"Missing samples csv: {args.samples_csv}")
        sample_rows = _count_rows(args.samples_csv)
        if sample_rows == 0:
            raise ValueError("Samples CSV has zero rows.")
        if sample_rows != label_rows:
            raise ValueError(f"Row mismatch: labels={label_rows}, samples={sample_rows}")

    dep_results = {
        "numpy": _check_import("numpy"),
        "pyserial": _check_import("serial"),
        "flask": _check_import("flask"),
    }
    if args.require_web and not dep_results["flask"]:
        raise RuntimeError("Web UI dependency missing (install flask).")

    print("Demo healthcheck PASS")
    print(f"labels_rows={label_rows}")
    if sample_rows is not None:
        print(f"samples_rows={sample_rows}")
    for k, v in dep_results.items():
        print(f"{k}={'ok' if v else 'missing'}")


if __name__ == "__main__":
    main()
