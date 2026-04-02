#!/usr/bin/env python3
"""Create a mock session_log.csv for replay path validation."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic session log from labels.")
    parser.add_argument("--labels_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--arduino_acc", type=float, default=0.8)
    parser.add_argument("--rpi_acc", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _draw_pred(gt: int, n_classes: int, acc: float) -> int:
    if random.random() < acc:
        return gt
    options = [i for i in range(n_classes) if i != gt]
    return random.choice(options)


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.labels_csv.open("r", encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))

    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "sample_id", "source", "pred", "target", "correct", "inf_ms", "raw"])
        ts = time.time()
        for row in labels:
            sample_id = int(row["sample_id"])
            gt = int(row["target"])
            pred_a = _draw_pred(gt, args.num_classes, args.arduino_acc)
            pred_r = _draw_pred(gt, args.num_classes, args.rpi_acc)
            for source, pred, ms in (
                ("arduino", pred_a, random.uniform(7.0, 20.0)),
                ("rpi", pred_r, random.uniform(15.0, 40.0)),
            ):
                raw = f"sample_{sample_id} src={source} pred={pred} target={gt} inf_ms={ms:.3f} logits="
                writer.writerow([ts, sample_id, source, pred, gt, int(pred == gt), f"{ms:.3f}", raw])
                ts += 0.03

    print(f"Wrote mock session log: {args.output_csv}")


if __name__ == "__main__":
    main()
