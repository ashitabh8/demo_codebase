#!/usr/bin/env python3
"""Generate deterministic mock Arduino/RPi protocol streams from labels CSV."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock protocol streams for UI testing.")
    parser.add_argument("--labels_csv", type=Path, required=True)
    parser.add_argument("--arduino_out", type=Path, required=True)
    parser.add_argument("--rpi_out", type=Path, required=True)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--arduino_acc", type=float, default=0.80)
    parser.add_argument("--rpi_acc", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _emit_line(sample_id: int, source: str, pred: int, target: int, inf_ms: float) -> str:
    return (
        f"sample_{sample_id} src={source} pred={pred} target={target} "
        f"inf_ms={inf_ms:.3f} logits="
    )


def _draw_pred(target: int, n_classes: int, acc: float) -> int:
    if random.random() < acc:
        return target
    choices = [i for i in range(n_classes) if i != target]
    return random.choice(choices)


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    args.arduino_out.parent.mkdir(parents=True, exist_ok=True)
    args.rpi_out.parent.mkdir(parents=True, exist_ok=True)

    with args.labels_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    with args.arduino_out.open("w", encoding="utf-8") as f_a, args.rpi_out.open("w", encoding="utf-8") as f_r:
        for row in rows:
            sample_id = int(row["sample_id"])
            target = int(row["target"])
            pred_a = _draw_pred(target, args.num_classes, args.arduino_acc)
            pred_r = _draw_pred(target, args.num_classes, args.rpi_acc)
            inf_a = random.uniform(8.0, 20.0)
            inf_r = random.uniform(15.0, 45.0)
            f_a.write(_emit_line(sample_id, "arduino", pred_a, target, inf_a) + "\n")
            f_r.write(_emit_line(sample_id, "rpi", pred_r, target, inf_r) + "\n")

    print(f"Wrote {len(rows)} mock lines to {args.arduino_out} and {args.rpi_out}")


if __name__ == "__main__":
    main()
