#!/usr/bin/env python3
"""
Full train → W8 export → package samples → make → smoke infer.

Expects a train / val / test layout under --data_root when possible:

  data/train/*.txt
  data/val/*.txt
  data/test/*.txt   # optional but recommended; used for held-out C scoring

Usage:
  python run_pipeline.py
  python3 run_pipeline.py --data_root data --epochs 20
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))

from load_1600_txt import load_dir  # noqa: E402


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd or PKG), check=True)


def write_test_bins(test_dir: Path, out_root: Path) -> tuple[Path, Path, Path]:
    """Write test_bin/*.bin + test_y.npy + test_names.txt for run_c_test_infer."""
    names, streams, labels = load_dir(test_dir)
    bin_dir = out_root / "test_bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for old in bin_dir.glob("*.bin"):
        old.unlink()
    for name, stream in zip(names, streams):
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        with (bin_dir / f"{stem}.bin").open("wb") as f:
            f.write(struct.pack(f"{len(stream)}f", *stream.astype(np.float32).tolist()))
    y_path = out_root / "test_y.npy"
    names_path = out_root / "test_names.txt"
    np.save(y_path, labels.astype(np.int64))
    names_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"wrote {len(names)} test bins under {bin_dir}")
    return bin_dir, y_path, names_path


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
    parser.add_argument(
        "--sample_dir",
        type=Path,
        default=None,
        help="Smoke samples for deploy/. Defaults to data_root/test if present else val.",
    )
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Force MLP head size. Default: max(train/val label)+1.",
    )
    parser.add_argument(
        "--class_names",
        type=Path,
        default=None,
        help="Text file with one class name per line (length must equal num_classes).",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--skip_package", action="store_true")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--skip_test_infer", action="store_true")
    args = parser.parse_args()

    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    test_dir = args.data_root / "test"
    has_test = test_dir.is_dir() and any(test_dir.glob("*.txt"))
    if not train_dir.is_dir() or not any(train_dir.glob("*.txt")):
        raise FileNotFoundError(f"missing train txts under {train_dir}")
    if not val_dir.is_dir() or not any(val_dir.glob("*.txt")):
        raise FileNotFoundError(f"missing val txts under {val_dir}")
    if not has_test:
        print(
            f"WARNING: no {test_dir}/*.txt — training will use train/val only; "
            "add a held-out test/ for true train/val/test evaluation."
        )

    sample_dir = args.sample_dir or (test_dir if has_test else val_dir)
    py = sys.executable
    ckpt = args.artifacts_dir / "spectral_mlp.pth"

    if not args.skip_train:
        train_cmd = [
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
        if args.num_classes is not None:
            train_cmd.extend(["--num_classes", str(args.num_classes)])
        if args.class_names is not None:
            train_cmd.extend(["--class_names", str(args.class_names)])
        run(train_cmd)

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
        package_cmd = [
            py,
            str(PKG / "package_deploy_samples.py"),
            "--sample_dir",
            str(sample_dir),
            "--output_dir",
            str(args.deploy_dir),
            "--max_samples",
            str(args.max_samples),
            "--ckpt_path",
            str(ckpt),
        ]
        if args.class_names is not None:
            package_cmd.extend(["--class_names", str(args.class_names)])
        run(package_cmd)

    if not args.skip_build:
        run(["make", "clean"], cwd=args.deploy_dir)
        run(["make"], cwd=args.deploy_dir)

    if not args.skip_smoke:
        infer = args.deploy_dir / "spectral_infer"
        if not infer.exists():
            raise FileNotFoundError(f"missing {infer}; build failed?")
        run([str(infer)], cwd=args.deploy_dir)

    if has_test and not args.skip_test_infer:
        bin_dir, y_path, names_path = write_test_bins(test_dir, args.artifacts_dir)
        score_cmd = [
            py,
            str(PKG / "run_c_test_infer.py"),
            "--deploy_dir",
            str(args.deploy_dir),
            "--test_bin_dir",
            str(bin_dir),
            "--test_y",
            str(y_path),
            "--test_names",
            str(names_path),
            "--output_json",
            str(args.artifacts_dir / "c_test_report.json"),
            "--ckpt_path",
            str(ckpt),
        ]
        if args.class_names is not None:
            score_cmd.extend(["--class_names", str(args.class_names)])
        run(score_cmd)

    print("\nPipeline complete.")
    print(f"  checkpoint: {ckpt}")
    print(f"  deploy packet (handoff): {args.deploy_dir}")
    if has_test:
        print(f"  test report: {args.artifacts_dir / 'c_test_report.json'}")
    print("  Recipient: cd deploy && make && ./spectral_infer")


if __name__ == "__main__":
    main()
