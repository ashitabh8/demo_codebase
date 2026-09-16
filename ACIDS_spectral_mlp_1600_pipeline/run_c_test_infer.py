#!/usr/bin/env python3
"""
Run C spectral_infer on every test .bin and score against labels.

Inference itself is 100% C (Welch preprocess + scaler + W8 MLP).
This script only launches the binary and aggregates match rates.

Usage:
  python3 run_c_test_infer.py --deploy_dir deploy --test_bin_dir exported_acids/test_bin \\
      --test_y exported_acids/test_y.npy --test_names exported_acids/test_names.txt
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
CLASS_NAMES = [
    "background",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


def parse_infer_output(text: str) -> list[dict]:
    blocks = []
    cur: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if cur:
                blocks.append(cur)
                cur = {}
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        cur[k] = v
    if cur:
        blocks.append(cur)
    return blocks


def main():
    parser = argparse.ArgumentParser(description="Score ACIDS test set via C spectral_infer")
    parser.add_argument("--deploy_dir", type=Path, default=PKG / "deploy")
    parser.add_argument("--test_bin_dir", type=Path, required=True)
    parser.add_argument("--test_y", type=Path, required=True)
    parser.add_argument("--test_names", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Cap number of test bins (0 = all)",
    )
    args = parser.parse_args()

    infer = args.deploy_dir / "spectral_infer"
    if not infer.exists():
        raise FileNotFoundError(f"missing {infer}; run make -C {args.deploy_dir}")

    names = [ln.strip() for ln in args.test_names.read_text(encoding="utf-8").splitlines() if ln.strip()]
    y_true = np.load(args.test_y).astype(np.int64)
    if len(names) != len(y_true):
        raise ValueError(f"names ({len(names)}) != labels ({len(y_true)})")

    if args.max_samples > 0:
        names = names[: args.max_samples]
        y_true = y_true[: args.max_samples]

    bin_paths = []
    for name in names:
        p = args.test_bin_dir / f"{name}.bin"
        if not p.exists():
            raise FileNotFoundError(p)
        bin_paths.append(str(p))

    # Batch argv to avoid huge command lines if needed — run in chunks of 64.
    chunk = 64
    all_preds = []
    for i in range(0, len(bin_paths), chunk):
        batch = bin_paths[i : i + chunk]
        proc = subprocess.run(
            [str(infer), *batch],
            cwd=str(args.deploy_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        blocks = parse_infer_output(proc.stdout)
        if len(blocks) != len(batch):
            raise RuntimeError(
                f"expected {len(batch)} predictions, got {len(blocks)}\n{proc.stdout[-500:]}"
            )
        for b in blocks:
            all_preds.append(int(b["pred_id"]))

    y_pred = np.asarray(all_preds, dtype=np.int64)
    correct = int(np.sum(y_pred == y_true))
    n = len(y_true)
    acc = correct / max(n, 1)

    # macro-F1 without sklearn to keep this script light if needed
    f1s = []
    for c in range(len(CLASS_NAMES)):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    macro_f1 = float(np.mean(f1s))

    report = {
        "n_test": n,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "correct": correct,
        "pred_hist": {CLASS_NAMES[k]: int(v) for k, v in Counter(y_pred.tolist()).items()},
        "true_hist": {CLASS_NAMES[k]: int(v) for k, v in Counter(y_true.tolist()).items()},
        "inference": "C spectral_infer (Welch + scaler + W8 MLP)",
    }
    out = args.output_json or (args.deploy_dir.parent / "artifacts" / "c_test_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
