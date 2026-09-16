"""Welch spectral features matching deploy/acids_spectral_preprocess.c."""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 1600
N_FFT = 160
HOP = 80
FEATURE_DIM = 83
EPS = 1e-12


def compute_welch_spectral_features(
    streams: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop: int = HOP,
    use_hann: bool = False,
) -> dict[str, np.ndarray]:
    """Average periodogram over overlapping windows covering each full stream.

    Args:
        streams: [N, T] float32 (T should be 1600 for the deploy contract)
    Returns:
        dict with centroid [N], mean_energy [N], psd [N, n_fft//2+1]
    """
    if hop <= 0:
        raise ValueError("welch hop must be positive")
    n_samples, t_len = streams.shape
    n_bins = n_fft // 2 + 1
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate).astype(np.float64)
    avg_power = np.zeros((n_samples, n_bins), dtype=np.float64)

    for i in range(n_samples):
        stream = streams[i].astype(np.float32)
        frame_powers = []
        last_start = t_len - n_fft
        if last_start < 0:
            starts = [0]
        else:
            starts = list(range(0, last_start + 1, hop))
            if starts[-1] != last_start:
                starts.append(last_start)
        for start in starts:
            frame = stream[start : start + n_fft]
            if frame.shape[0] < n_fft:
                pad = np.zeros(n_fft - frame.shape[0], dtype=np.float32)
                frame = np.concatenate([frame, pad], axis=0)
            if use_hann:
                win = np.hanning(n_fft).astype(np.float32)
                frame = frame * win
            fft = np.fft.rfft(frame, n=n_fft)
            frame_powers.append((fft.real ** 2 + fft.imag ** 2).astype(np.float64))
        avg_power[i] = np.mean(np.stack(frame_powers, axis=0), axis=0)

    power = avg_power
    psd = power / (power.sum(axis=-1, keepdims=True) + EPS)
    centroid = (power * freqs).sum(axis=-1) / (power.sum(axis=-1) + EPS)
    mean_energy = power.mean(axis=-1)
    return {
        "centroid": centroid.astype(np.float32),
        "mean_energy": mean_energy.astype(np.float32),
        "psd": psd.astype(np.float32),
        "freqs": freqs.astype(np.float32),
    }


def build_combined_matrix(features: dict[str, np.ndarray]) -> np.ndarray:
    """Stack centroid, mean_energy, and PSD into [N, 2 + n_freqs] (=83)."""
    return np.concatenate(
        [
            features["centroid"].reshape(-1, 1),
            features["mean_energy"].reshape(-1, 1),
            features["psd"],
        ],
        axis=1,
    )


def streams_to_features(streams: np.ndarray) -> np.ndarray:
    """[N, 1600] -> [N, 83] Welch features."""
    return build_combined_matrix(compute_welch_spectral_features(streams))
