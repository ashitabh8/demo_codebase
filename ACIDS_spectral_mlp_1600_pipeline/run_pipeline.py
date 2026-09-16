#!/usr/bin/env python3
"""
Full train → W8 export → package samples → make → smoke infer.

Usage:
  python run_pipeline.py
  python3 run_pipeline.py --data_root data --epochs 20
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd or PKG), check=True)


def main():
    parser = argparse.ArgumentParser(description="Spectral MLP 1600 Welch train→deploy pipeline")
    parser.add_argument("--data_root", type=Path, default=PKG / "data")
    parser.add_argument("--artifacts_dir", type=Path, default=PKG / "artifacts")
    parser.add_argument("--deploy_dir", type=Path, default=PKG / "deploy")
    parser.add_argument(
        "--tiny_nn_root",
        type=Path,
        default=PKG / "Tiny-NN-in-C",
        help="Tiny-NN-in-C root (default: vendored Tiny-NN-in-C/ in this folder)",
    )
    parser.add_argument("--sample_dir", type=Path, default=None, help="Defaults to data_root/val")
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--skip_package", action="store_true")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--skip_smoke", action="store_true")
    args = parser.parse_args()

    sample_dir = args.sample_dir or (args.data_root / "val")
    py = sys.executable
    ckpt = args.artifacts_dir / "spectral_mlp.pth"

    if not args.skip_train:
        run(
            [
                py,
                str(PKG / "train_from_1600.py"),
                "--data_root",
                str(args.data_root),
                "--output_dir",
                str(args.artifacts_dir),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--gpu",
                str(args.gpu),
            ]
        )

    if not args.skip_export:
        run(
            [
                py,
                str(PKG / "export_w8_spectral_mlp.py"),
                "--ckpt_path",
                str(ckpt),
                "--output_dir",
                str(args.deploy_dir),
                "--tiny_nn_root",
                str(args.tiny_nn_root),
            ]
        )

    if not args.skip_package:
        run(
            [
                py,
                str(PKG / "package_deploy_samples.py"),
                "--sample_dir",
                str(sample_dir),
                "--output_dir",
                str(args.deploy_dir),
                "--max_samples",
                str(args.max_samples),
            ]
        )

    if not args.skip_build:
        run(["make", "clean"], cwd=args.deploy_dir)
        run(["make"], cwd=args.deploy_dir)

    if not args.skip_smoke:
        infer = args.deploy_dir / "spectral_infer"
        if not infer.exists():
            raise FileNotFoundError(f"missing {infer}; build failed?")
        run([str(infer)], cwd=args.deploy_dir)

    print("\nPipeline complete.")
    print(f"  checkpoint: {ckpt}")
    print(f"  deploy packet (handoff): {args.deploy_dir}")
    print("  Recipient: cd deploy && make && ./spectral_infer")


if __name__ == "__main__":
    main()
