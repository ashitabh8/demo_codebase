#!/usr/bin/env python3
"""
Parity test: C acids_mel_preprocess vs PyTorch MelPreprocessor.

Builds [1,1,7,228] from 1600-sample @ 1.6 kHz input (trim last 4 samples),
compiles the C shared library, and asserts max abs diff < 1e-4.
"""

import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

INFERENCE_C = Path(__file__).resolve().parent
SRC2 = INFERENCE_C.parent
sys.path.insert(0, str(SRC2))

from data_augmenter.mel_preprocess import MelPreprocessor  # noqa: E402

INPUT_SAMPLES = 1600
NUM_SEGMENTS = 7
SEG_SAMPLES = 228
USED_SAMPLES = NUM_SEGMENTS * SEG_SAMPLES
MEL_BINS = 80
OUTPUT_SIZE = NUM_SEGMENTS * MEL_BINS
TOLERANCE = 1e-4


def split_input_1600(input_1600: np.ndarray) -> np.ndarray:
    """Trim to 1596 samples and reshape to [7, 228]."""
    if input_1600.shape != (INPUT_SAMPLES,):
        raise ValueError(f"expected shape ({INPUT_SAMPLES},), got {input_1600.shape}")
    trimmed = input_1600[:USED_SAMPLES]
    return trimmed.reshape(NUM_SEGMENTS, SEG_SAMPLES)


def pytorch_mel_chw(input_1600: np.ndarray) -> np.ndarray:
    """Reference mel output [1, 7, 80] CHW."""
    segments = split_input_1600(input_1600.astype(np.float32))
    x = torch.from_numpy(segments).unsqueeze(0).unsqueeze(0)
    mel_pp = MelPreprocessor(
        n_fft=160, n_mel=80, fmin=20.0, fmax=800.0, sample_rate=1600, device="cpu"
    )
    out = mel_pp.preprocess({"shake": {"audio": x}})
    return out["shake"]["audio"].squeeze(0).numpy()


def compile_shared_lib(build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    lib_path = build_dir / "libacids_mel_preprocess.so"
    cmd = [
        "gcc",
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c99",
        str(INFERENCE_C / "acids_mel_preprocess.c"),
        "-o",
        str(lib_path),
        "-lm",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return lib_path


def c_mel_chw(lib_path: Path, input_1600: np.ndarray) -> np.ndarray:
    lib = ctypes.CDLL(str(lib_path))
    lib.acids_mel_preprocess_chw.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.acids_mel_preprocess_chw.restype = ctypes.c_int

    in_arr = np.ascontiguousarray(input_1600.astype(np.float32))
    out_arr = np.zeros(OUTPUT_SIZE, dtype=np.float32)
    rc = lib.acids_mel_preprocess_chw(
        in_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"acids_mel_preprocess_chw returned {rc}")
    return out_arr.reshape(1, NUM_SEGMENTS, MEL_BINS)


def c_mel_hwc(lib_path: Path, input_1600: np.ndarray) -> np.ndarray:
    lib = ctypes.CDLL(str(lib_path))
    lib.acids_mel_preprocess_hwc.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.acids_mel_preprocess_hwc.restype = ctypes.c_int

    in_arr = np.ascontiguousarray(input_1600.astype(np.float32))
    out_arr = np.zeros(OUTPUT_SIZE, dtype=np.float32)
    rc = lib.acids_mel_preprocess_hwc(
        in_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"acids_mel_preprocess_hwc returned {rc}")
    return out_arr.reshape(NUM_SEGMENTS, MEL_BINS, 1)


def assert_close(ref: np.ndarray, got: np.ndarray, label: str) -> float:
    diff = float(np.max(np.abs(ref - got)))
    if diff >= TOLERANCE:
        idx = np.unravel_index(np.argmax(np.abs(ref - got)), ref.shape)
        raise AssertionError(
            f"{label}: max abs diff {diff:.6e} >= {TOLERANCE} at {idx} "
            f"(ref={ref[idx]:.6f}, got={got[idx]:.6f})"
        )
    print(f"  {label}: max abs diff {diff:.6e} OK")
    return diff


def main():
    print("Compiling C library...")
    with tempfile.TemporaryDirectory() as tmp:
        lib_path = compile_shared_lib(Path(tmp))

        rng = np.random.default_rng(42)
        cases = [
            ("random", rng.standard_normal(INPUT_SAMPLES).astype(np.float32)),
            ("zeros", np.zeros(INPUT_SAMPLES, dtype=np.float32)),
            ("ramp", np.linspace(-1.0, 1.0, INPUT_SAMPLES, dtype=np.float32)),
        ]

        for name, inp in cases:
            print(f"Case: {name}")
            ref = pytorch_mel_chw(inp)
            got_chw = c_mel_chw(lib_path, inp)
            got_hwc = c_mel_hwc(lib_path, inp)

            assert_close(ref, got_chw, "CHW")
            assert_close(ref.reshape(NUM_SEGMENTS, MEL_BINS, 1), got_hwc, "HWC")

    print("All parity checks passed.")


if __name__ == "__main__":
    main()
