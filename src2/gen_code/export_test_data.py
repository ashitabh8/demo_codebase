#!/usr/bin/env python3
"""Export deterministic test inputs and PyTorch reference outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from compile_deepsense import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_MODEL_NAME,
    DEFAULT_YAML,
    INPUT_SHAPE,
    load_model,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_ROOT = REPO_ROOT / "src2" / "gen_code"
DEFAULT_DATA_DIR = GEN_ROOT / "test_data"
DEFAULT_HEADER = GEN_ROOT / "test_inputs.h"
DEFAULT_EXPECTED_OUTPUTS = GEN_ROOT / "expected_outputs.txt"


def _write_test_header(
    header_path: Path,
    samples_nhwc: list[np.ndarray],
    output_size: int,
) -> None:
    input_size = int(samples_nhwc[0].size)
    with header_path.open("w", encoding="utf-8") as handle:
        handle.write("// Auto-generated test input arrays\n")
        handle.write("#pragma once\n\n")
        handle.write(f"#define NUM_TEST_SAMPLES {len(samples_nhwc)}\n")
        handle.write(f"#define TEST_INPUT_SIZE {input_size}\n")
        handle.write(f"#define TEST_OUTPUT_SIZE {output_size}\n\n")

        for idx, sample in enumerate(samples_nhwc):
            values = ", ".join(f"{float(v):.9g}f" for v in sample.reshape(-1))
            handle.write(f"static const float sample_{idx}[TEST_INPUT_SIZE] = {{{values}}};\n")
        handle.write("\n")

        ptrs = ", ".join(f"sample_{idx}" for idx in range(len(samples_nhwc)))
        handle.write(
            f"static const float* const TEST_INPUTS[NUM_TEST_SAMPLES] = {{{ptrs}}};\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export test inputs for generated C model.")
    parser.add_argument("--yaml_path", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--header_path", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--expected_outputs_path", type=Path, default=DEFAULT_EXPECTED_OUTPUTS)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.header_path.parent.mkdir(parents=True, exist_ok=True)
    args.expected_outputs_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.yaml_path, args.checkpoint_path, args.model_name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    nhwc_samples: list[np.ndarray] = []
    expected_lines: list[str] = []
    output_size = 0

    for idx in range(args.num_samples):
        x_nchw = torch.randn(*INPUT_SHAPE)
        with torch.no_grad():
            y = model(x_nchw).detach().cpu().numpy().reshape(-1)
        output_size = int(y.size)

        x_nhwc = x_nchw[0].permute(1, 2, 0).contiguous().detach().cpu().numpy().reshape(-1)
        nhwc_samples.append(x_nhwc)

        np.savetxt(args.data_dir / f"test_input_{idx}.txt", x_nhwc, fmt="%.9g")
        np.savetxt(args.data_dir / f"pytorch_output_{idx}.txt", y, fmt="%.9g")
        expected_lines.append(
            "sample_"
            + str(idx)
            + " "
            + " ".join(f"{float(v):.9g}" for v in y.tolist())
        )

    _write_test_header(args.header_path, nhwc_samples, output_size)
    args.expected_outputs_path.write_text("\n".join(expected_lines) + "\n", encoding="utf-8")
    print(f"Saved {args.num_samples} inputs/outputs to {args.data_dir}")
    print(f"Wrote header: {args.header_path}")
    print(f"Wrote expected outputs: {args.expected_outputs_path}")


if __name__ == "__main__":
    main()
