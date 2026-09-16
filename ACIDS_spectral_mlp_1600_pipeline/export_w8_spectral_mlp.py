#!/usr/bin/env python3
"""
Export W8 Spectral MLP to C via Tiny-NN-in-C (linear weight-only int8).

Also writes spectral_scaler.h from checkpoint scaler_mean / scaler_scale
(not folded into weights).

Usage:
  python export_w8_spectral_mlp.py \\
    --ckpt_path artifacts/spectral_mlp.pth \\
    --output_dir deploy \\
    --tiny_nn_root Tiny-NN-in-C   # default: vendored copy in this folder
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parent
DEFAULT_TINY_NN = PKG / "Tiny-NN-in-C"
DEFAULT_CKPT = PKG / "artifacts" / "spectral_mlp.pth"
DEFAULT_OUTPUT = PKG / "deploy"

sys.path.insert(0, str(PKG))
from train_from_1600 import SpectralMLP  # noqa: E402


def build_linear_w8_rules(ir_graph):
    from src.pytorch_to_c.quantization import Int8WeightOnlyLinearRule

    rules = []
    for node in ir_graph.nodes:
        if node.op_type != "linear":
            continue
        pattern = f"^{re.escape(node.name)}$"
        rules.append(Int8WeightOnlyLinearRule(pattern=pattern))
    if not rules:
        raise RuntimeError("No linear nodes found in IR graph")
    return rules


def write_spectral_scaler_h(output_dir: Path, mean: np.ndarray, scale: np.ndarray) -> None:
    mean = np.asarray(mean, dtype=np.float64).ravel()
    scale = np.asarray(scale, dtype=np.float64).ravel()
    if mean.shape != scale.shape:
        raise ValueError(f"scaler mean/scale shape mismatch: {mean.shape} vs {scale.shape}")
    dim = int(mean.shape[0])

    def fmt_float(v: float) -> str:
        # Always emit a decimal form so "11580956f" is never an invalid C token.
        s = f"{float(v):.9g}"
        if "e" in s or "E" in s:
            return s + "f"
        if "." not in s:
            s = s + ".0"
        return s + "f"

    def fmt_arr(name: str, arr: np.ndarray) -> str:
        lines = [f"static const float {name}[{dim}] = {{"]
        row = []
        for i, v in enumerate(arr):
            row.append(fmt_float(float(v)))
            if len(row) == 4 or i == dim - 1:
                lines.append("    " + ", ".join(row) + ("," if i < dim - 1 else ""))
                row = []
        lines.append("};")
        return "\n".join(lines)

    body = f"""/* Auto-generated StandardScaler stats for spectral MLP input.
 * Apply in C before model_forward(); do not fold into model weights.
 * Frontend: Welch n_fft=160 hop=80 over 1600 samples -> 83 features.
 */
#ifndef SPECTRAL_SCALER_H
#define SPECTRAL_SCALER_H

#define SPECTRAL_FEATURE_DIM {dim}

{fmt_arr("SPECTRAL_SCALER_MEAN", mean)}

{fmt_arr("SPECTRAL_SCALER_SCALE", scale)}

void spectral_apply_standard_scaler(
    const float *raw,
    float *scaled,
    int dim
);

#endif /* SPECTRAL_SCALER_H */
"""
    (output_dir / "spectral_scaler.h").write_text(body, encoding="utf-8")
    print(f"  wrote spectral_scaler.h ({dim} dims)")


def patch_model_c_includes(model_c: Path) -> None:
    """Tiny-NN may omit nn_ops_int8.h; ensure both float and int8 ops headers are present."""
    text = model_c.read_text(encoding="utf-8")
    if '#include "nn_ops_int8.h"' in text:
        return
    needle = '#include "nn_ops_float.h"\n'
    if needle in text:
        text = text.replace(needle, needle + '#include "nn_ops_int8.h"\n', 1)
    else:
        text = '#include "nn_ops_int8.h"\n' + text
    model_c.write_text(text, encoding="utf-8")
    print("  patched model.c to include nn_ops_int8.h")


def ensure_minimal_nn_ops(output_dir: Path) -> None:
    """Keep lean hand-written ops used by spectral MLP deploy (override bulky Tiny-NN copies)."""
    float_h = """// Minimal float ops for spectral MLP deploy.
#ifndef NN_OPS_FLOAT_H
#define NN_OPS_FLOAT_H

static inline void relu(float *x, int n)
{
    for (int i = 0; i < n; ++i) {
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

#endif /* NN_OPS_FLOAT_H */
"""
    int8_h = """// Minimal int8 ops for spectral MLP deploy.
#ifndef NN_OPS_INT8_H
#define NN_OPS_INT8_H

#include <stdint.h>

static inline void dense_float_input_int8_weight_per_channel(
    const float *x,
    int in_features,
    const int8_t *W,
    const float *b,
    int out_features,
    const float *weight_scales,
    float *y)
{
    for (int o = 0; o < out_features; ++o) {
        float acc = (b != NULL) ? b[o] : 0.0f;
        float scale_o = weight_scales[o];
        const int8_t *w_col = W + o;
        for (int i = 0; i < in_features; ++i) {
            acc += x[i] * ((float)w_col[i * out_features] * scale_o);
        }
        y[o] = acc;
    }
}

#endif /* NN_OPS_INT8_H */
"""
    (output_dir / "nn_ops_float.h").write_text(float_h, encoding="utf-8")
    (output_dir / "nn_ops_int8.h").write_text(int8_h, encoding="utf-8")
    print("  wrote nn_ops_float.h / nn_ops_int8.h (minimal)")


def export_w8(
    ckpt_path: Path,
    output_dir: Path,
    tiny_nn_root: Path,
) -> None:
    sys.path.insert(0, str(tiny_nn_root))
    from src.pytorch_to_c.codegen.c_printer import CPrinter
    from src.pytorch_to_c.compiler import compile_model
    from src.pytorch_to_c.quantization import QuantizationTransform

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    in_dim = int(ckpt["in_dim"])
    num_classes = int(ckpt["num_classes"])
    model = SpectralMLP(in_dim, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    example_input = torch.randn(1, in_dim)
    print(f"Compiling IR (example input {list(example_input.shape)})...")
    ir_graph = compile_model(model, example_input, return_ir=True, verbose=False)

    rules = build_linear_w8_rules(ir_graph)
    print(f"Applying {len(rules)} Int8WeightOnlyLinearRule(s) (no calibrate)...")
    ir_graph = QuantizationTransform(rules).apply(ir_graph)

    output_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = output_dir / "_tiny_nn_gen"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    print(f"Generating C -> {gen_dir}")
    CPrinter(ir_graph).generate_all(str(gen_dir))

    for name in ("model.c", "model.h", "weights.h"):
        src = gen_dir / name
        if not src.exists():
            raise FileNotFoundError(f"expected generated file missing: {src}")
        shutil.copy2(src, output_dir / name)
        print(f"  copied {name}")

    shutil.rmtree(gen_dir)
    ensure_minimal_nn_ops(output_dir)
    patch_model_c_includes(output_dir / "model.c")
    write_spectral_scaler_h(
        output_dir,
        np.asarray(ckpt["scaler_mean"]),
        np.asarray(ckpt["scaler_scale"]),
    )
    print(f"Done. W8 C artifacts in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export W8 spectral MLP + scaler header")
    parser.add_argument("--ckpt_path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tiny_nn_root", type=Path, default=DEFAULT_TINY_NN)
    args = parser.parse_args()

    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    if not args.tiny_nn_root.exists():
        raise FileNotFoundError(
            f"Tiny-NN-in-C not found at {args.tiny_nn_root}. Set --tiny_nn_root."
        )

    export_w8(args.ckpt_path, args.output_dir, args.tiny_nn_root)


if __name__ == "__main__":
    main()
