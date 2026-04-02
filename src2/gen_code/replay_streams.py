#!/usr/bin/env python3
"""Replay or convert logged session rows into stream protocol lines."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay demo session CSV as protocol lines.")
    parser.add_argument("--ui_input", type=Path, required=True, help="Session log CSV from demo_ui.py")
    parser.add_argument("--output_path", type=Path, default=None, help="Optional output file; defaults to stdout")
    parser.add_argument("--delay_s", type=float, default=0.0, help="Delay between lines for live replay")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    out_handle = sys.stdout
    close_out = False
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        out_handle = args.output_path.open("w", encoding="utf-8")
        close_out = True

    try:
        with args.ui_input.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                line = (
                    f"sample_{row['sample_id']} src={row['source']} pred={row['pred']} "
                    f"target={row['target']} inf_ms={row['inf_ms']} logits="
                )
                out_handle.write(line + "\n")
                out_handle.flush()
                if args.delay_s > 0:
                    time.sleep(args.delay_s)
    finally:
        if close_out:
            out_handle.close()


if __name__ == "__main__":
    main()
