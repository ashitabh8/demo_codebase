"""
Test suite for src2/data_augmenter/mel_preprocess.py

Tests:
  1.  Filterbank shape and non-negativity
  2.  Mel center frequencies are strictly increasing
  3.  Filter coverage between fmin and fmax
  4.  Shape correctness of preprocess output
  5.  Numerical match vs torchaudio MelScale reference
  6.  Log stability (no -inf or nan)
  7.  Hz-Mel round-trip accuracy
  8.  DC bin handling with fmin > 0
  9.  Energy ordering for a pure low-frequency tone
  10. Batch consistency
"""

import sys
import math
import torch
import pytest
from pathlib import Path

# Make src2 importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_augmenter.mel_preprocess import (
    hz_to_mel,
    mel_to_hz,
    build_mel_filterbank,
    MelPreprocessor,
)

# ── Default test parameters ────────────────────────────────────────────────────
N_FFT = 512
N_MEL = 64
FMIN = 20.0
FMAX = 8000.0
SR = 16000
N_FREQS = N_FFT // 2 + 1  # 257


# ── Test 1: Filterbank shape is [n_mel, n_fft//2+1], all values >= 0 ──────────
def test_filterbank_shape_and_nonnegative():
    fb = build_mel_filterbank(N_FFT, N_MEL, FMIN, FMAX, SR)
    assert fb.shape == (N_MEL, N_FREQS), (
        f"Expected shape ({N_MEL}, {N_FREQS}), got {fb.shape}"
    )
    assert (fb >= 0).all(), "Filterbank contains negative values"


# ── Test 2: Mel center frequencies are strictly increasing ────────────────────
def test_mel_center_frequencies_increasing():
    mel_min = hz_to_mel(FMIN)
    mel_max = hz_to_mel(FMAX)
    mel_points = [mel_min + i * (mel_max - mel_min) / (N_MEL + 1) for i in range(N_MEL + 2)]
    hz_centers = [mel_to_hz(m) for m in mel_points[1 : N_MEL + 1]]  # peak of each filter

    for i in range(len(hz_centers) - 1):
        assert hz_centers[i] < hz_centers[i + 1], (
            f"Center frequencies not strictly increasing at index {i}: "
            f"{hz_centers[i]:.2f} >= {hz_centers[i+1]:.2f}"
        )


# ── Test 3: Filter coverage between fmin and fmax ─────────────────────────────
def test_filter_coverage():
    fb = build_mel_filterbank(N_FFT, N_MEL, FMIN, FMAX, SR)
    fft_bin_freqs = torch.tensor(
        [k * SR / N_FFT for k in range(N_FREQS)], dtype=torch.float32
    )
    # For each FFT bin strictly inside (fmin, fmax), at least one filter should be nonzero
    in_range = (fft_bin_freqs > FMIN) & (fft_bin_freqs < FMAX)
    covered = fb.sum(dim=0) > 0  # [N_FREQS] – True if any filter is non-zero

    uncovered_freqs = fft_bin_freqs[in_range & ~covered]
    assert len(uncovered_freqs) == 0, (
        f"{len(uncovered_freqs)} FFT bins between fmin/fmax have zero coverage: "
        f"{uncovered_freqs.tolist()[:5]}"
    )


# ── Test 4: Shape correctness ─────────────────────────────────────────────────
def test_output_shape():
    B, C, SEG = 3, 2, 5
    preprocessor = MelPreprocessor(N_FFT, N_MEL, FMIN, FMAX, SR, device="cpu")
    x = torch.randn(B, C, SEG, N_FFT)
    inputs = {"loc1": {"mod1": x}}
    outputs = preprocessor.preprocess(inputs)
    out = outputs["loc1"]["mod1"]
    assert out.shape == (B, C, SEG, N_MEL), (
        f"Expected shape ({B}, {C}, {SEG}, {N_MEL}), got {out.shape}"
    )


# ── Test 5: vs torchaudio MelScale reference ──────────────────────────────────
def test_vs_torchaudio_reference():
    """
    Compare our log-mel output against torchaudio MelScale applied to the same
    power spectrum.  We use NO window so that both implementations see identical
    power spectra.

    Reference pipeline:
        power = rfft(x).abs()**2
        mel   = torchaudio.transforms.MelScale(...)(power)
        ref   = log(mel + 1e-6)
    """
    import torchaudio  # noqa: F401 – guarded import so the test is skippable

    B, C, SEG = 2, 1, 4
    torch.manual_seed(42)
    x = torch.randn(B, C, SEG, N_FFT)

    # ── Our implementation ────────────────────────────────────────────────────
    preprocessor = MelPreprocessor(N_FFT, N_MEL, FMIN, FMAX, SR, device="cpu")
    our_out = preprocessor.preprocess({"loc": {"mod": x}})["loc"]["mod"]
    # our_out: [B, C, SEG, N_MEL]

    # ── torchaudio reference (no window, raw power spectrum) ─────────────────
    mel_scale = torchaudio.transforms.MelScale(
        n_mels=N_MEL,
        sample_rate=SR,
        f_min=FMIN,
        f_max=FMAX,
        n_stft=N_FREQS,
        norm=None,
        mel_scale="htk",
    )
    # Compute raw power spectrum (same as our pipeline)
    power = torch.fft.rfft(x, n=N_FFT, dim=-1).abs() ** 2  # [B, C, SEG, N_FREQS]
    # MelScale expects [..., freq, time]; our last two dims are (SEG, N_FREQS)
    # Reshape to [B*C, N_FREQS, SEG], apply, then reshape back
    bc = B * C
    power_t = power.view(bc, SEG, N_FREQS).permute(0, 2, 1)  # [B*C, N_FREQS, SEG]
    mel_t = mel_scale(power_t)                                 # [B*C, N_MEL, SEG]
    mel_t = mel_t.permute(0, 2, 1).view(B, C, SEG, N_MEL)    # [B, C, SEG, N_MEL]
    ref_out = torch.log(mel_t + 1e-6)

    assert torch.allclose(our_out, ref_out, atol=1e-3), (
        f"Max absolute difference: {(our_out - ref_out).abs().max().item():.6f}"
    )


# ── Test 6: Log stability – no -inf or nan ────────────────────────────────────
def test_log_stability():
    B, C, SEG = 4, 2, 8
    torch.manual_seed(0)
    x = torch.randn(B, C, SEG, N_FFT)
    preprocessor = MelPreprocessor(N_FFT, N_MEL, FMIN, FMAX, SR, device="cpu")
    out = preprocessor.preprocess({"loc": {"mod": x}})["loc"]["mod"]
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"


# ── Test 7: Hz-Mel round-trip ─────────────────────────────────────────────────
def test_hz_mel_round_trip():
    freqs = [20.0, 100.0, 440.0, 1000.0, 4000.0, 8000.0]
    for f in freqs:
        recovered = mel_to_hz(hz_to_mel(f))
        assert abs(recovered - f) < 1e-4, (
            f"Round-trip failed for {f} Hz: recovered {recovered:.6f} Hz, "
            f"error {abs(recovered - f):.2e}"
        )


# ── Test 8: DC bin handling with fmin > 0 ────────────────────────────────────
def test_dc_bin_near_zero_with_fmin_above_zero():
    fmin_above_zero = 50.0
    fb = build_mel_filterbank(N_FFT, N_MEL, fmin_above_zero, FMAX, SR)
    # Bin 0 is DC (0 Hz); all filters should have essentially zero weight there
    assert (fb[:, 0] < 0.01).all(), (
        f"DC bin weights unexpectedly large: {fb[:, 0].tolist()}"
    )


# ── Test 9: Energy ordering for a pure 200 Hz tone ───────────────────────────
def test_energy_ordering_pure_tone():
    # Create a batch of 200 Hz pure tones
    t = torch.arange(N_FFT, dtype=torch.float32) / SR
    tone = torch.sin(2 * math.pi * 200.0 * t)  # [N_FFT]
    # Shape to [B=1, C=1, SEG=1, N_FFT]
    x = tone.view(1, 1, 1, N_FFT)

    preprocessor = MelPreprocessor(N_FFT, N_MEL, FMIN, FMAX, SR, device="cpu")
    out = preprocessor.preprocess({"loc": {"mod": x}})["loc"]["mod"]
    log_mel = out[0, 0, 0, :]  # [N_MEL]

    # 200 Hz falls in the low mel bins; compare average energy of lower vs upper half
    low_half_mean = log_mel[: N_MEL // 2].mean().item()
    high_half_mean = log_mel[N_MEL // 2 :].mean().item()
    assert low_half_mean > high_half_mean, (
        f"Expected low bins ({low_half_mean:.4f}) > high bins ({high_half_mean:.4f}) "
        "for a 200 Hz tone"
    )


# ── Test 10: Batch consistency ────────────────────────────────────────────────
def test_batch_consistency():
    torch.manual_seed(7)
    x_single = torch.randn(1, 2, 4, N_FFT)
    x_batch = x_single.repeat(2, 1, 1, 1)  # identical rows

    preprocessor = MelPreprocessor(N_FFT, N_MEL, FMIN, FMAX, SR, device="cpu")
    out = preprocessor.preprocess({"loc": {"mod": x_batch}})["loc"]["mod"]
    # Both batch items should produce the same output
    assert torch.allclose(out[0], out[1], atol=1e-6), (
        f"Batch items differ; max diff: {(out[0] - out[1]).abs().max().item():.2e}"
    )
