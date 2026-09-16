#!/usr/bin/env python3
"""
End-to-end ACIDS spectral MLP pipeline (self-contained folder):

  1) Export train/val/test .pt indices → 1600 streams (.npy + test .bin)
  2) Train Welch-83 Spectral MLP on train, validate on val
  3) Export W8 C (Tiny-NN) + spectral_scaler.h + Welch C frontend
  4) Package a small smoke set into deploy/samples
  5) make spectral_infer
  6) Run C inference on the full test split and write accuracy report

Paths come from paths.yaml (or CLI overrides):

  train_index_file / val_index_file / test_index_file

Usage:
  conda activate spectral_mlp_1600
  python3 run_acids_pipeline.py --paths_yaml paths.yaml --epochs 50 --gpu 0
  python3 run_acids_pipeline.py --max_export_samples 500   # smoke / debug
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parent


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd or PKG), check=True)


def main():
    parser = argparse.ArgumentParser(description="ACIDS train/val → W8 C → test infer")
    parser.add_argument("--paths_yaml", type=Path, default=PKG / "paths.yaml")
    parser.add_argument("--train_index", type=Path, default=None)
    parser.add_argument("--val_index", type=Path, default=None)
    parser.add_argument("--test_index", type=Path, default=None)
    parser.add_argument("--export_root", type=Path, default=PKG / "exported_acids")
    parser.add_argument("--artifacts_dir", type=Path, default=PKG / "artifacts")
    parser.add_argument("--deploy_dir", type=Path, default=PKG / "deploy")
    parser.add_argument("--tiny_nn_root", type=Path, default=PKG / "Tiny-NN-in-C")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument(
        "--max_export_samples",
        type=int,
        default=0,
        help="Cap each split during export (0=all). Use for quick smoke runs.",
    )
    parser.add_argument("--smoke_package_samples", type=int, default=32)
    parser.add_argument("--skip_export_data", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_w8", action="store_true")
    parser.add_argument("--skip_package", action="store_true")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--skip_test_infer", action="store_true")
    args = parser.parse_args()

    cfg = {}
    if args.paths_yaml.exists():
        with args.paths_yaml.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    train_index = args.train_index or Path(cfg["train_index_file"])
    val_index = args.val_index or Path(cfg["val_index_file"])
    test_index = args.test_index or Path(cfg["test_index_file"])
    py = sys.executable
    ckpt = args.artifacts_dir / "spectral_mlp.pth"

    if not args.skip_export_data:
        cmd = [
            py,
            str(PKG / "export_acids_indices.py"),
            "--paths_yaml",
            str(args.paths_yaml),
            "--train_index",
            str(train_index),
            "--val_index",
            str(val_index),
            "--test_index",
            str(test_index),
            "--output_root",
            str(args.export_root),
            "--max_samples",
            str(args.max_export_samples),
        ]
        run(cmd)

    if not args.skip_train:
        run(
            [
                py,
                str(PKG / "train_from_1600.py"),
                "--export_root",
                str(args.export_root),
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

    if not args.skip_w8:
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
                str(args.export_root / "test_txt"),
                "--output_dir",
                str(args.deploy_dir),
                "--max_samples",
                str(args.smoke_package_samples),
            ]
        )

    if not args.skip_build:
        run(["make", "clean"], cwd=args.deploy_dir)
        run(["make"], cwd=args.deploy_dir)

    if not args.skip_test_infer:
        run(
            [
                py,
                str(PKG / "run_c_test_infer.py"),
                "--deploy_dir",
                str(args.deploy_dir),
                "--test_bin_dir",
                str(args.export_root / "test_bin"),
                "--test_y",
                str(args.export_root / "test_y.npy"),
                "--test_names",
                str(args.export_root / "test_names.txt"),
                "--output_json",
                str(args.artifacts_dir / "c_test_report.json"),
                "--max_samples",
                str(args.max_export_samples),
            ]
        )

    print("\nACIDS pipeline complete.")
    print(f"  export_root: {args.export_root}")
    print(f"  checkpoint:  {ckpt}")
    print(f"  deploy/:     {args.deploy_dir}  (self-contained C packet)")
    print(f"  test report: {args.artifacts_dir / 'c_test_report.json'}")


if __name__ == "__main__":
    main()
