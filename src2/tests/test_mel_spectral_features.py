"""Unit tests for analyze_mel_spectral_features.py"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_analysis.analyze_mel_spectral_features import (
    build_combined_matrix,
    compute_spectral_features,
)


def test_spectral_features_shapes_sine():
    sr = 1600
    n_fft = 160
    t = np.arange(1600, dtype=np.float32) / sr
    freq = 200.0
    sine = np.sin(2 * np.pi * freq * t)
    streams = np.stack([sine, sine * 0.5], axis=0)

    feats = compute_spectral_features(streams, sample_rate=sr, n_fft=n_fft, use_hann=False)
    assert feats["centroid"].shape == (2,)
    assert feats["mean_energy"].shape == (2,)
    assert feats["psd"].shape == (2, n_fft // 2 + 1)
    assert np.allclose(feats["psd"].sum(axis=-1), 1.0, atol=1e-5)

    combined = build_combined_matrix(feats)
    assert combined.shape == (2, 2 + n_fft // 2 + 1)


def test_hann_changes_centroid():
    sr = 1600
    n_fft = 160
    rng = np.random.default_rng(0)
    streams = rng.standard_normal((4, 1600)).astype(np.float32)
    a = compute_spectral_features(streams, sample_rate=sr, n_fft=n_fft, use_hann=False)
    b = compute_spectral_features(streams, sample_rate=sr, n_fft=n_fft, use_hann=True)
    assert not np.allclose(a["centroid"], b["centroid"])
