#!/usr/bin/env python3
"""Compare C inference logits against PyTorch reference logits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "src2" / "gen_code" / "test_data"
DEFAULT_C_OUTPUT = REPO_ROOT / "src2" / "gen_code" / "c_outputs.txt"
DEFAULT_EXPECTED_OUTPUTS = REPO_ROOT / "src2" / "gen_code" / "expected_outputs.txt"
DEFAULT_REPORT_CSV = REPO_ROOT / "src2" / "gen_code" / "comparison_report.csv"


def _parse_c_outputs(path: Path) -> dict[int, np.ndarray]:
    parsed: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            tag = parts[0]
            if not tag.startswith("sample_"):
                continue
            idx = int(tag.replace("sample_", ""))
            values = np.array([float(v) for v in parts[1:]], dtype=np.float64)
            parsed[idx] = values
    return parsed


def _parse_expected_outputs_file(path: Path) -> dict[int, np.ndarray]:
    return _parse_c_outputs(path)


def _load_reference_from_dir(data_dir: Path) -> dict[int, np.ndarray]:
    references: dict[int, np.ndarray] = {}
    pytorch_files = sorted(data_dir.glob("pytorch_output_*.txt"))
    if not pytorch_files:
        raise FileNotFoundError(f"No pytorch_output_*.txt files found in {data_dir}")
    for path in pytorch_files:
        idx = int(path.stem.replace("pytorch_output_", ""))
        references[idx] = np.loadtxt(path, dtype=np.float64).reshape(-1)
    return references


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch outputs vs C outputs.")
    parser.add_argument("--c_output_path", type=Path, default=DEFAULT_C_OUTPUT)
    parser.add_argument("--reference_dir", type=Path, default=None)
    parser.add_argument("--reference_file", type=Path, default=None)
    parser.add_argument("--report_csv_path", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    c_outputs = _parse_c_outputs(args.c_output_path)
    if not c_outputs:
        raise FileNotFoundError(f"No sample_* lines found in {args.c_output_path}")

    if args.reference_dir is not None and args.reference_file is not None:
        raise ValueError("Use only one of --reference_dir or --reference_file")
    if args.reference_dir is None and args.reference_file is None:
        args.reference_dir = DEFAULT_DATA_DIR

    if args.reference_file is not None:
        references = _parse_expected_outputs_file(args.reference_file)
    else:
        references = _load_reference_from_dir(args.reference_dir)

    if not references:
        ref_hint = args.reference_file if args.reference_file is not None else args.reference_dir
        raise FileNotFoundError(f"No reference outputs found in {ref_hint}")

    args.report_csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_pass = True
    num_correct = 0
    num_incorrect = 0
    max_err_global = 0.0

    common_indices = sorted(set(references.keys()).intersection(c_outputs.keys()))
    if not common_indices:
        raise ValueError("No overlapping sample IDs between references and C outputs")

    first_ref = references[common_indices[0]]
    num_classes = int(first_ref.shape[0])
    csv_headers = [
        "sample_id",
        "pytorch_pred",
        "c_pred",
        "classification_match",
        "max_abs_err",
    ]
    csv_headers += [f"pt_logit_{i}" for i in range(num_classes)]
    csv_headers += [f"c_logit_{i}" for i in range(num_classes)]

    with args.report_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_headers)
        writer.writeheader()

        for idx in common_indices:
            pt = references[idx].reshape(-1)
            c = c_outputs[idx].reshape(-1)
            if pt.shape != c.shape:
                raise ValueError(
                    f"Shape mismatch for sample_{idx}: pytorch={pt.shape}, c={c.shape}"
                )

            pt_pred = int(np.argmax(pt))
            c_pred = int(np.argmax(c))
            is_correct = pt_pred == c_pred
            classification_match = "CORRECT" if is_correct else "INCORRECT"

            max_abs = float(np.max(np.abs(pt - c)))
            max_err_global = max(max_err_global, max_abs)
            ok = max_abs <= args.tolerance
            all_pass = all_pass and ok

            if is_correct:
                num_correct += 1
            else:
                num_incorrect += 1

            row = {
                "sample_id": idx,
                "pytorch_pred": pt_pred,
                "c_pred": c_pred,
                "classification_match": classification_match,
                "max_abs_err": f"{max_abs:.9g}",
            }
            for i in range(num_classes):
                row[f"pt_logit_{i}"] = f"{float(pt[i]):.9g}"
                row[f"c_logit_{i}"] = f"{float(c[i]):.9g}"
            writer.writerow(row)

            verdict = "PASS" if ok else "FAIL"
            print(
                f"sample_{idx}: max_abs_err={max_abs:.6e} [{verdict}] "
                f"classification={classification_match} (pt={pt_pred}, c={c_pred})"
            )

    missing_in_c = sorted(set(references.keys()) - set(c_outputs.keys()))
    if missing_in_c:
        print(f"Warning: missing C outputs for samples: {missing_in_c}")
    extra_in_c = sorted(set(c_outputs.keys()) - set(references.keys()))
    if extra_in_c:
        print(f"Warning: extra C outputs without references: {extra_in_c}")

    print(
        f"Classification matches: {num_correct}/{len(common_indices)}; "
        f"mismatches: {num_incorrect}"
    )
    print(f"Global max abs error: {max_err_global:.6e}")
    print(f"CSV report: {args.report_csv_path}")

    if not all_pass:
        raise SystemExit(1)
    print(f"All compared samples PASS tolerance={args.tolerance:g}")


if __name__ == "__main__":
    main()
