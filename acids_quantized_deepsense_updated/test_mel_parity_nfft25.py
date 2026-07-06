#!/usr/bin/env python3
"""Parity test: C acids_mel_preprocess vs PyTorch MelPreprocessor (n_fft=25)."""

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parent
SRC2 = PKG.parent / "src2"
sys.path.insert(0, str(SRC2))

from data_augmenter.mel_preprocess import MelPreprocessor

NUM_SEGMENTS = 7
SAMPLES_PER_SEGMENT = 25
MEL_BINS = 80
OUTPUT_SIZE = NUM_SEGMENTS * MEL_BINS
TOLERANCE = 1e-4


def compile_shared_lib() -> Path:
    lib_path = PKG / "libacids_mel_preprocess.so"
    cmd = [
        "gcc",
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c99",
        str(PKG / "acids_mel_preprocess.c"),
        "-o",
        str(lib_path),
        "-lm",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return lib_path


def pytorch_mel(chw_25: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(chw_25.astype(np.float32)).unsqueeze(0)
    mel_pp = MelPreprocessor(
        n_fft=25, n_mel=80, fmin=20.0, fmax=800.0, sample_rate=1600, device="cpu"
    )
    out = mel_pp.preprocess({"shake": {"audio": x}})
    return out["shake"]["audio"].squeeze(0).numpy()


def c_mel(chw_25: np.ndarray, lib_path: Path) -> np.ndarray:
    lib = ctypes.CDLL(str(lib_path))
    lib.acids_audio_preprocess_ch0_segments_chw.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.acids_audio_preprocess_ch0_segments_chw.restype = ctypes.c_int

    flat_in = np.ascontiguousarray(chw_25.astype(np.float32).reshape(-1))
    out_arr = np.zeros(OUTPUT_SIZE, dtype=np.float32)
    rc = lib.acids_audio_preprocess_ch0_segments_chw(
        flat_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"acids_audio_preprocess_ch0_segments_chw returned {rc}")
    return out_arr.reshape(1, NUM_SEGMENTS, MEL_BINS)


def main() -> None:
    lib_path = compile_shared_lib()
    rng = np.random.default_rng(42)
    chw = rng.standard_normal((3, NUM_SEGMENTS, SAMPLES_PER_SEGMENT)).astype(np.float32)
    ref = pytorch_mel(chw[0:1])
    got = c_mel(chw, lib_path)
    diff = float(np.max(np.abs(ref - got)))
    if diff >= TOLERANCE:
        raise AssertionError(f"max abs diff {diff:.6e} >= {TOLERANCE}")
    print(f"mel parity OK (max abs diff {diff:.6e})")


if __name__ == "__main__":
    main()
