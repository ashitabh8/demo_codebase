#!/usr/bin/env python3
"""Run W8A8 validation workflows (quick C-only or full compile flow)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from compare_outputs import DEFAULT_C_OUTPUT, DEFAULT_EXPECTED_OUTPUTS, DEFAULT_REPORT_CSV
from compile_w8a8 import DEFAULT_OUTPUT_DIR, DEFAULT_YAML
from export_test_data_w8a8 import DEFAULT_DATA_DIR, DEFAULT_HEADER


ROOT = Path(__file__).resolve().parent


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
        ROOT,
    )
    with c_output_path.open("w", encoding="utf-8") as handle:
        subprocess.run(["./test_inference"], cwd=str(ROOT), check=True, stdout=handle)


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
        ROOT,
    )


def full_checkpoint_check(args: argparse.Namespace) -> None:
    compile_cmd = [
        sys.executable,
        "compile_w8a8.py",
        "--yaml_path",
        str(args.yaml_path),
        "--experiment_name",
        str(args.experiment_name),
        "--model_name",
        str(args.model_name),
        "--float_model_name",
        str(args.float_model_name),
        "--output_dir",
        str(args.generated_dir),
        "--calib_batches",
        str(args.calib_batches),
    ]
    export_cmd = [
        sys.executable,
        "export_test_data_w8a8.py",
        "--yaml_path",
        str(args.yaml_path),
        "--experiment_name",
        str(args.experiment_name),
        "--model_name",
        str(args.model_name),
        "--split",
        str(args.split),
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
    ]
    if args.checkpoint_path is not None:
        compile_cmd += ["--checkpoint_path", str(args.checkpoint_path)]
        export_cmd += ["--checkpoint_path", str(args.checkpoint_path)]

    _run(compile_cmd, ROOT)
    _run(export_cmd, ROOT)
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
        ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="W8A8 validation workflow runner.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subp: argparse.ArgumentParser) -> None:
        subp.add_argument("--generated_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        subp.add_argument("--c_output_path", type=Path, default=DEFAULT_C_OUTPUT)
        subp.add_argument("--reference_file", type=Path, default=DEFAULT_EXPECTED_OUTPUTS)
        subp.add_argument("--report_csv_path", type=Path, default=DEFAULT_REPORT_CSV)
        subp.add_argument("--tolerance", type=float, default=0.5)

    quick = subparsers.add_parser("quick-c-check", help="Run C model on saved samples only.")
    add_common(quick)

    full = subparsers.add_parser("full-checkpoint-check", help="Compile + export + run + compare.")
    add_common(full)
    full.add_argument("--yaml_path", type=Path, default=DEFAULT_YAML)
    full.add_argument("--checkpoint_path", type=Path, default=None)
    full.add_argument("--experiment_name", type=str, default="finetune_audio_deepsense_dw_simple_tiny_w8a8")
    full.add_argument("--model_name", type=str, default="student_audio_deepsense_dw_simple_tiny_w8a8")
    full.add_argument("--float_model_name", type=str, default="student_audio_deepsense_dw_simple_tiny")
    full.add_argument("--calib_batches", type=int, default=50)
    full.add_argument("--num_samples", type=int, default=50)
    full.add_argument("--split", type=str, choices=("train", "val", "test"), default="val")
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
