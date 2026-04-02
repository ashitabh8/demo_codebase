#!/usr/bin/env python3
"""Copy generated model/demo headers into a single Arduino sketch bundle."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
GEN_DIR = REPO_ROOT / "src2" / "gen_code" / "generated"
DEMO_DIR = REPO_ROOT / "src2" / "gen_code" / "demo_data"
SKETCH_DIR = REPO_ROOT / "src2" / "gen_code" / "arduino_demo"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Arduino sketch bundle files.")
    parser.add_argument("--sketch_dir", type=Path, default=SKETCH_DIR)
    parser.add_argument("--generated_dir", type=Path, default=GEN_DIR)
    parser.add_argument("--demo_data_dir", type=Path, default=DEMO_DIR)
    return parser.parse_args()


def _copy(src: Path, dst_dir: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required source file: {src}")
    shutil.copy2(src, dst_dir / src.name)
    print(f"Copied {src.name}")


def main() -> None:
    args = _parse_args()
    args.sketch_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("model.h", "model.c", "weights.h", "nn_ops_float.h", "nn_ops_int8.h", "nn_ops_int16.h"):
        _copy(args.generated_dir / fname, args.sketch_dir)
    _copy(args.demo_data_dir / "demo_samples.h", args.sketch_dir)

    print(f"Bundle ready at: {args.sketch_dir}")


if __name__ == "__main__":
    main()
