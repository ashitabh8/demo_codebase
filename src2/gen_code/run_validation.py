#!/usr/bin/env python3
"""Run DeepSense validation workflows (quick C-only or full checkpoint flow)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from compile_deepsense import (
    DEFAULT_CKPT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT,
    DEFAULT_YAML,
)
from compare_outputs import DEFAULT_C_OUTPUT, DEFAULT_EXPECTED_OUTPUTS, DEFAULT_REPORT_CSV
from export_test_data import DEFAULT_DATA_DIR, DEFAULT_HEADER


GEN_ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _build_and_run_c(generated_dir: Path, c_output_path: Path) -> None:
    _run(
        [
            "gcc",
            "-O2",
            "-I",
            str(generated_dir),
            "-o",
            "test_inference",
            str(generated_dir / "model.c"),
            "test_main.c",
            "-lm",
        ],
        GEN_ROOT,
    )
    with c_output_path.open("w", encoding="utf-8") as handle:
        subprocess.run(["./test_inference"], cwd=str(GEN_ROOT), check=True, stdout=handle)


def quick_c_check(args: argparse.Namespace) -> None:
    _build_and_run_c(args.generated_dir, args.c_output_path)
    _run(
        [
            sys.executable,
            "compare_outputs.py",
            "--c_output_path",
            str(args.c_output_path),
            "--reference_file",
            str(args.reference_file),
            "--report_csv_path",
            str(args.report_csv_path),
            "--tolerance",
            str(args.tolerance),
        ],
        GEN_ROOT,
    )


def full_checkpoint_check(args: argparse.Namespace) -> None:
    _run(
        [
            sys.executable,
            "compile_deepsense.py",
            "--yaml_path",
            str(args.yaml_path),
            "--checkpoint_path",
            str(args.checkpoint_path),
            "--model_name",
            str(args.model_name),
            "--output_dir",
            str(args.generated_dir),
        ],
        GEN_ROOT,
    )
    _run(
        [
            sys.executable,
            "export_test_data.py",
            "--yaml_path",
            str(args.yaml_path),
            "--checkpoint_path",
            str(args.checkpoint_path),
            "--model_name",
            str(args.model_name),
            "--num_samples",
            str(args.num_samples),
            "--seed",
            str(args.seed),
            "--data_dir",
            str(args.data_dir),
            "--header_path",
            str(args.header_path),
            "--expected_outputs_path",
            str(args.reference_file),
        ],
        GEN_ROOT,
    )
    _build_and_run_c(args.generated_dir, args.c_output_path)
    _run(
        [
            sys.executable,
            "compare_outputs.py",
            "--c_output_path",
            str(args.c_output_path),
            "--reference_dir",
            str(args.data_dir),
            "--report_csv_path",
            str(args.report_csv_path),
            "--tolerance",
            str(args.tolerance),
        ],
        GEN_ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSense validation workflow runner.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subp: argparse.ArgumentParser) -> None:
        subp.add_argument("--generated_dir", type=Path, default=DEFAULT_OUTPUT)
        subp.add_argument("--c_output_path", type=Path, default=DEFAULT_C_OUTPUT)
        subp.add_argument("--reference_file", type=Path, default=DEFAULT_EXPECTED_OUTPUTS)
        subp.add_argument("--report_csv_path", type=Path, default=DEFAULT_REPORT_CSV)
        subp.add_argument("--tolerance", type=float, default=1e-3)

    quick = subparsers.add_parser("quick-c-check", help="Run C model on saved samples only.")
    add_common(quick)

    full = subparsers.add_parser("full-checkpoint-check", help="Compile + export + run + compare.")
    add_common(full)
    full.add_argument("--yaml_path", type=Path, default=DEFAULT_YAML)
    full.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CKPT)
    full.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    full.add_argument("--num_samples", type=int, default=50)
    full.add_argument("--seed", type=int, default=1234)
    full.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    full.add_argument("--header_path", type=Path, default=DEFAULT_HEADER)

    args = parser.parse_args()
    if args.mode == "quick-c-check":
        quick_c_check(args)
    else:
        full_checkpoint_check(args)


if __name__ == "__main__":
    main()
