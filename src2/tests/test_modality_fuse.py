"""Tests for audio+seismic channel concat fusion."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_augmenter.modality_fuse import fuse_channel_concat
from data_augmenter.Augmenter import Augmenter


def _mel_inputs(batch=2, audio_c=1, seismic_c=1, segments=7, mel=80):
    return {
        "shake": {
            "audio": torch.randn(batch, audio_c, segments, mel),
            "seismic": torch.randn(batch, seismic_c, segments, mel),
        }
    }


def test_fuse_channel_concat_pad0():
    x = _mel_inputs()
    out = fuse_channel_concat(x, location="shake", pad_channels=0)
    assert out["shake"]["audio"].shape == (2, 2, 7, 80)
    assert torch.equal(out["shake"]["audio"][:, 0], x["shake"]["audio"][:, 0])
    assert torch.equal(out["shake"]["audio"][:, 1], x["shake"]["seismic"][:, 0])


def test_fuse_channel_concat_pad1():
    x = _mel_inputs()
    out = fuse_channel_concat(x, location="shake", pad_channels=1)
    assert out["shake"]["audio"].shape == (2, 3, 7, 80)
    assert torch.all(out["shake"]["audio"][:, 1] == 0)


def test_fuse_channel_concat_bad_shape():
    x = {
        "shake": {
            "audio": torch.randn(2, 1, 7, 80),
            "seismic": torch.randn(2, 1, 6, 80),
        }
    }
    with pytest.raises(ValueError, match="segment/mel shapes"):
        fuse_channel_concat(x, location="shake")


def test_augmenter_fused_mel_shape():
    args = SimpleNamespace(
        device="cpu",
        train_mode="supervised",
        stage="train",
        learn_framework=None,
        model="student_audio_seismic_fused_mel_2ch",
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
            "seismic_channel_index": 0,
            "modality_fusion": {
                "mode": "channel_concat",
                "pad_channels": 0,
                "output_modality": "audio",
            },
            "fixed_augmenters": {
                "time_augmenters": ["no"],
                "freq_augmenters": ["no"],
            },
            "time_mask": {"prob": 1.0, "mask_ratio": 0.1},
            "freq_mask": {"prob": 0.0, "mask_ratio": 0.1},
        },
    )
    augmenter = Augmenter(args)
    x_in = {
        "shake": {
            "audio": torch.randn(2, 3, 7, 1600),
            "seismic": torch.randn(2, 2, 7, 1600),
        }
    }
    out, _ = augmenter.forward("no", x_in, labels=torch.zeros(2, dtype=torch.long))
    assert out["shake"]["audio"].shape == (2, 2, 7, 80)
