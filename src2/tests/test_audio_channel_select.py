"""Tests for single-mic audio channel selection."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_augmenter.audio_channel_select import select_audio_channel
from data_augmenter.Augmenter import Augmenter


def _make_inputs(batch=2, audio_c=3, segments=7, audio_t=160):
    return {
        "shake": {
            "audio": torch.randn(batch, audio_c, segments, audio_t),
            "seismic": torch.randn(batch, 2, segments, audio_t),
        }
    }


def test_select_audio_channel_dim1():
    x = _make_inputs()
    out = select_audio_channel(x, channel_index=1, num_channels=3)
    assert out["shake"]["audio"].shape == (2, 1, 7, 160)
    assert torch.equal(out["shake"]["audio"][:, 0], x["shake"]["audio"][:, 1])
    assert torch.equal(out["shake"]["seismic"], x["shake"]["seismic"])


def test_select_audio_channel_legacy_dim2():
    x = {"shake": {"audio": torch.randn(2, 1, 3, 160)}}
    out = select_audio_channel(x, channel_index=2, num_channels=3)
    assert out["shake"]["audio"].shape == (2, 1, 1, 160)
    assert torch.equal(out["shake"]["audio"][:, 0, 0], x["shake"]["audio"][:, 0, 2])


def test_select_audio_channel_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        select_audio_channel(_make_inputs(), channel_index=3, num_channels=3)


def test_augmenter_mel_1ch_output_shape():
    args = SimpleNamespace(
        device="cpu",
        train_mode="supervised",
        stage="train",
        learn_framework=None,
        model="student_audio_deepsense_dw_large_mel_1ch",
        dataset_config={
            "modality_names": ["seismic", "audio"],
            "location_names": ["shake"],
            "num_segments": 7,
            "preprocess_mode": "mel",
            "input_sample_rate": 16000,
            "sample_rate": 1600,
            "n_fft": 160,
            "mel_bins": 80,
            "mel_fmin": 20.0,
            "mel_fmax": 800.0,
            "downsample_modalities": ["audio"],
            "loc_mod_in_time_channels": {"shake": {"audio": 3, "seismic": 2}},
            "audio_channel_index": 0,
            "fixed_augmenters": {
                "time_augmenters": ["no"],
                "freq_augmenters": ["no"],
            },
            "time_mask": {"prob": 1.0, "mask_ratio": 0.1},
            "freq_mask": {"prob": 0.0, "mask_ratio": 0.1},
        },
    )
    augmenter = Augmenter(args)
    x_in = _make_inputs(audio_t=1600)
    out, _ = augmenter.forward("no", x_in, labels=torch.zeros(2, dtype=torch.long))
    assert out["shake"]["audio"].shape == (2, 1, 7, 80)
