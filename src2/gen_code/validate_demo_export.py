#!/usr/bin/env python3
"""Validate exported demo CSV files and class/shape consistency."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate demo export artifacts.")
    parser.add_argument("--data_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    samples_csv = args.data_dir / "demo_samples.csv"
    labels_csv = args.data_dir / "demo_labels.csv"
    metadata_json = args.data_dir / "demo_metadata.json"

    for path in (samples_csv, labels_csv, metadata_json):
        if not path.exists():
            raise FileNotFoundError(f"Missing expected file: {path}")

    meta = json.loads(metadata_json.read_text(encoding="utf-8"))

    with samples_csv.open("r", encoding="utf-8", newline="") as handle:
        sample_rows = list(csv.DictReader(handle))
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        label_rows = list(csv.DictReader(handle))

    if len(sample_rows) != len(label_rows):
        raise ValueError(f"Row count mismatch: demo_samples={len(sample_rows)} demo_labels={len(label_rows)}")
    if len(sample_rows) == 0:
        raise ValueError("Export contains zero rows.")

    feature_cols = [k for k in sample_rows[0].keys() if k.startswith("feature_")]
    if not feature_cols:
        raise ValueError("No feature_* columns found in demo_samples.csv")
    expected_feature_size = int(meta["feature_size_flat"])
    if len(feature_cols) != expected_feature_size:
        raise ValueError(
            f"Feature column count mismatch: csv={len(feature_cols)} metadata={expected_feature_size}"
        )

    sample_ids_a = [int(row["sample_id"]) for row in sample_rows]
    sample_ids_b = [int(row["sample_id"]) for row in label_rows]
    if sample_ids_a != sample_ids_b:
        raise ValueError("sample_id ordering mismatch between demo_samples.csv and demo_labels.csv")
    if len(set(sample_ids_a)) != len(sample_ids_a):
        raise ValueError("Duplicate sample_id values found.")

    targets = [int(row["target"]) for row in label_rows]
    class_counts = Counter(targets)
    allowed_targets = set(range(len(meta["class_names"])))
    bad_targets = sorted(set(targets) - allowed_targets)
    if bad_targets:
        raise ValueError(f"Unexpected target labels found: {bad_targets}")

    print("Validation passed.")
    print(f"Rows: {len(sample_rows)}")
    print(f"Feature size: {len(feature_cols)}")
    print(f"Target counts: {dict(sorted(class_counts.items()))}")
    print(f"Class names: {meta['class_names']}")


if __name__ == "__main__":
    main()
