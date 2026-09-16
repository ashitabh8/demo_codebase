"""
Tests for ACIDS continuous tail-cut truncation and mel windowing.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_augmenter.acids_continuous import AcidsContinuousTruncator
from data_augmenter.mel_preprocess import MelPreprocessor
from data_augmenter.Augmenter import Augmenter

NUM_SEGMENTS = 7
PADDED_SEG_LEN = 256
USED_SAMPLES = 1600
MEL_WINDOW = 228
MEL_HOP = 228
MEL_ROWS = 7
N_FFT = 160
N_MEL = 80
SR = 1600


def _make_segmented(batch=1, channels=1, fill=1.0):
    return torch.full((batch, channels, NUM_SEGMENTS, PADDED_SEG_LEN), fill)


def _truncator():
    return AcidsContinuousTruncator(
        num_segments=NUM_SEGMENTS,
        padded_seg_len=PADDED_SEG_LEN,
        used_samples=USED_SAMPLES,
        target_modalities=["audio"],
    )


def test_truncate_output_shape():
    x = _make_segmented()
    out = _truncator().truncate({"shake": {"audio": x}})["shake"]["audio"]
    assert out.shape == (1, 1, USED_SAMPLES)


def test_tail_cut_only_discards_end():
    x = _make_segmented()
    x[..., 1600:] = 999.0
    out = _truncator().truncate({"shake": {"audio": x}})["shake"]["audio"]
    assert not torch.any(out == 999.0)
    assert torch.all(out == 1.0)


def test_tail_cut_sensitive_to_last_kept_sample():
    x = _make_segmented()
    flat_idx = USED_SAMPLES - 1
    seg = flat_idx // PADDED_SEG_LEN
    t = flat_idx % PADDED_SEG_LEN
    x[0, 0, seg, t] = 42.0
    out = _truncator().truncate({"shake": {"audio": x}})["shake"]["audio"]
    assert out[0, 0, -1].item() == 42.0


def test_segments_zero_through_five_unchanged():
    x = torch.arange(NUM_SEGMENTS * PADDED_SEG_LEN, dtype=torch.float32).view(
        1, 1, NUM_SEGMENTS, PADDED_SEG_LEN
    )
    out = _truncator().truncate({"shake": {"audio": x}})["shake"]["audio"]
    expected = x.reshape(1, 1, -1)[..., :1536]
    assert torch.equal(out[..., :1536], expected)


def test_preprocess_continuous_mel_shape():
    stream = torch.randn(2, 1, USED_SAMPLES)
    mel_pp = MelPreprocessor(
        N_FFT,
        N_MEL,
        20.0,
        800.0,
        SR,
        device="cpu",
        mel_window=MEL_WINDOW,
        mel_hop=MEL_HOP,
        mel_rows=MEL_ROWS,
    )
    out = mel_pp.preprocess_continuous({"shake": {"audio": stream}})
    assert out["shake"]["audio"].shape == (2, 1, MEL_ROWS, N_MEL)
    assert torch.isfinite(out["shake"]["audio"]).all()


def _continuous_augmenter_args():
    return SimpleNamespace(
        device="cpu",
        train_mode="supervised",
        stage="train",
        learn_framework=None,
        model="student_audio_deepsense_dw_large_mel_1ch",
        dataset_config={
            "modality_names": ["audio"],
            "location_names": ["shake"],
            "num_segments": NUM_SEGMENTS,
            "preprocess_mode": "mel",
            "downsample_mode": "acids_continuous",
            "sample_rate": SR,
            "n_fft": N_FFT,
            "mel_bins": N_MEL,
            "mel_fmin": 20.0,
            "mel_fmax": 800.0,
            "acids_used_samples": USED_SAMPLES,
            "acids_padded_seg_len": PADDED_SEG_LEN,
            "acids_mel_window": MEL_WINDOW,
            "acids_mel_hop": MEL_HOP,
            "acids_mel_rows": MEL_ROWS,
            "downsample_modalities": ["audio"],
            "fixed_augmenters": {
                "time_augmenters": ["no"],
                "freq_augmenters": ["no"],
            },
            "time_mask": {"prob": 1.0, "mask_ratio": 0.1},
            "freq_mask": {"prob": 0.0, "mask_ratio": 0.1},
        },
    )


def test_augmenter_continuous_no_audio_downsampler():
    augmenter = Augmenter(_continuous_augmenter_args())
    assert not hasattr(augmenter, "audio_downsampler")
    assert hasattr(augmenter, "acids_truncator")


def test_augmenter_preprocess_continuous_end_to_end():
    augmenter = Augmenter(_continuous_augmenter_args())
    x_in = {"shake": {"audio": torch.randn(1, 1, NUM_SEGMENTS, PADDED_SEG_LEN)}}
    mel_out = augmenter.preprocess(x_in)
    assert mel_out["shake"]["audio"].shape == (1, 1, MEL_ROWS, N_MEL)


def test_mel_continuous_matches_windowed_reference():
    torch.manual_seed(0)
    stream = torch.randn(1, 1, USED_SAMPLES)
    mel_pp = MelPreprocessor(
        N_FFT,
        N_MEL,
        20.0,
        800.0,
        SR,
        device="cpu",
        mel_window=MEL_WINDOW,
        mel_hop=MEL_HOP,
        mel_rows=MEL_ROWS,
    )
    got = mel_pp.preprocess_continuous({"shake": {"audio": stream}})["shake"]["audio"]

    rows = []
    for i in range(MEL_ROWS):
        start = i * MEL_HOP
        window = stream[:, :, start : start + MEL_WINDOW]
        seg = window.unsqueeze(2)
        ref_row = mel_pp.preprocess({"shake": {"audio": seg}})["shake"]["audio"]
        rows.append(ref_row.squeeze(2))
    expected = torch.stack(rows, dim=2)
    assert torch.allclose(got, expected, atol=1e-5)
