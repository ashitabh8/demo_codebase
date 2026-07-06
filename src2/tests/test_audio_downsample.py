"""
Tests for anti-aliased audio downsampling (16 kHz -> 1.6 kHz).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_augmenter.audio_downsample import AudioDownsampler
from data_augmenter.mel_preprocess import MelPreprocessor
from data_augmenter.Augmenter import Augmenter


INPUT_SR = 16000
TARGET_SR = 1600
FACTOR = INPUT_SR // TARGET_SR


def _make_time_inputs(audio_t=1600, seismic_t=1600, batch=2, audio_c=3, seismic_c=2, segments=7):
    return {
        "shake": {
            "audio": torch.randn(batch, audio_c, segments, audio_t),
            "seismic": torch.randn(batch, seismic_c, segments, seismic_t),
        }
    }


def test_downsample_output_shape():
    ds = AudioDownsampler(INPUT_SR, TARGET_SR, target_modalities=["audio"])
    out = ds.downsample(_make_time_inputs())
    assert out["shake"]["audio"].shape == (2, 3, 7, 160)
    assert out["shake"]["seismic"].shape == (2, 2, 7, 1600)


def test_downsample_rejects_non_integer_factor():
    with pytest.raises(ValueError, match="integer multiple"):
        AudioDownsampler(16000, 1500)


def test_downsample_noop_when_rates_equal():
    ds = AudioDownsampler(16000, 16000)
    x_in = _make_time_inputs()
    out = ds.downsample(x_in)
    assert torch.equal(out["shake"]["audio"], x_in["shake"]["audio"])


def test_anti_aliasing_attenuates_out_of_band():
    """2 kHz tone at 16 kHz: FIR+decimate attenuates more than naive ::10."""
    t_len = 1600
    t = torch.arange(t_len, dtype=torch.float32) / INPUT_SR
    tone = torch.sin(2 * torch.pi * 2000.0 * t).view(1, 1, 1, t_len)

    ds = AudioDownsampler(INPUT_SR, TARGET_SR, target_modalities=["audio"])
    filtered = ds._decimate(tone)
    naive = tone[..., ::FACTOR]

    assert not torch.allclose(filtered, naive, atol=1e-3)
    assert filtered.pow(2).mean() < naive.pow(2).mean()


def test_augmenter_preprocess_mel_output_shape():
    args = SimpleNamespace(
        device="cpu",
        train_mode="supervised",
        stage="train",
        learn_framework=None,
        model="student_audio_deepsense_dw_large_mel",
        dataset_config={
            "modality_names": ["seismic", "audio"],
            "location_names": ["shake"],
            "num_segments": 7,
            "preprocess_mode": "mel",
            "input_sample_rate": INPUT_SR,
            "sample_rate": TARGET_SR,
            "n_fft": 160,
            "mel_bins": 80,
            "mel_fmin": 20.0,
            "mel_fmax": 800.0,
            "downsample_modalities": ["audio"],
            "fixed_augmenters": {
                "time_augmenters": ["no"],
                "freq_augmenters": ["no"],
            },
            "time_mask": {"prob": 1.0, "mask_ratio": 0.1},
            "freq_mask": {"prob": 0.0, "mask_ratio": 0.1},
        },
    )
    augmenter = Augmenter(args)
    mel_out = augmenter.preprocess(_make_time_inputs())
    assert mel_out["shake"]["audio"].shape == (2, 3, 7, 80)
    assert mel_out["shake"]["seismic"].shape[0:3] == (2, 2, 7)


def test_forward_noaug_downsamples_audio():
    args = SimpleNamespace(
        device="cpu",
        train_mode="supervised",
        stage="train",
        learn_framework=None,
        model="student_audio_deepsense_dw_large_mel",
        dataset_config={
            "modality_names": ["seismic", "audio"],
            "location_names": ["shake"],
            "num_segments": 7,
            "preprocess_mode": "mel",
            "input_sample_rate": INPUT_SR,
            "sample_rate": TARGET_SR,
            "n_fft": 160,
            "mel_bins": 80,
            "mel_fmin": 20.0,
            "mel_fmax": 800.0,
            "downsample_modalities": ["audio"],
            "fixed_augmenters": {
                "time_augmenters": ["no"],
                "freq_augmenters": ["no"],
            },
            "time_mask": {"prob": 1.0, "mask_ratio": 0.1},
            "freq_mask": {"prob": 0.0, "mask_ratio": 0.1},
        },
    )
    augmenter = Augmenter(args)
    x_in = _make_time_inputs()
    out, _ = augmenter.forward_noaug(x_in, labels=torch.zeros(2, dtype=torch.long))
    assert out["shake"]["audio"].shape == (2, 3, 7, 80)


def test_forward_fixed_time_mask_before_downsample():
    """Time mask zeros segments at 16 kHz; downsample runs in preprocess afterward."""
    args = SimpleNamespace(
        device="cpu",
        train_mode="supervised",
        stage="train",
        learn_framework=None,
        model="student_audio_deepsense_dw_large_mel",
        dataset_config={
            "modality_names": ["audio"],
            "location_names": ["shake"],
            "num_segments": 7,
            "preprocess_mode": "mel",
            "input_sample_rate": INPUT_SR,
            "sample_rate": TARGET_SR,
            "n_fft": 160,
            "mel_bins": 80,
            "mel_fmin": 20.0,
            "mel_fmax": 800.0,
            "downsample_modalities": ["audio"],
            "fixed_augmenters": {
                "time_augmenters": ["time_mask"],
                "freq_augmenters": ["no"],
            },
            "time_mask": {"prob": 1.0, "mask_ratio": 0.5},
            "freq_mask": {"prob": 0.0, "mask_ratio": 0.1},
        },
    )
    torch.manual_seed(0)
    augmenter = Augmenter(args)
    x_in = _make_time_inputs(batch=1, audio_c=1, segments=4, audio_t=1600)
    # Run time aug only (simulate forward_fixed pre-preprocess)
    aug_x = x_in
    for time_aug in augmenter.time_augmenters:
        aug_x, _, _ = time_aug(aug_x, labels=torch.zeros(1, dtype=torch.long))
    assert aug_x["shake"]["audio"].shape[-1] == 1600

    mel_out = augmenter.preprocess(aug_x)
    assert mel_out["shake"]["audio"].shape == (1, 1, 4, 80)


def _make_acids_time_inputs(batch=1, audio_c=3, segments=7, audio_t=256):
    return {
        "shake": {
            "audio": torch.randn(batch, audio_c, segments, audio_t),
        }
    }


def test_nfft25_mel_after_acids_downsample():
    """ACIDS-shaped 256-sample segments -> 25 @ 1600 Hz -> mel with n_fft=25."""
    ds = AudioDownsampler(INPUT_SR, TARGET_SR, target_modalities=["audio"])
    decimated = ds.downsample(_make_acids_time_inputs())
    assert decimated["shake"]["audio"].shape == (1, 3, 7, 25)

    mel = MelPreprocessor(
        n_fft=25,
        n_mel=80,
        fmin=20.0,
        fmax=800.0,
        sample_rate=TARGET_SR,
        device="cpu",
    )
    mel_out = mel.preprocess(decimated)
    out = mel_out["shake"]["audio"]
    assert out.shape == (1, 3, 7, 80)
    assert torch.isfinite(out).all()


def test_nfft25_mel_differs_from_nfft160_padded():
    """n_fft=25 on 25 samples is not equivalent to n_fft=160 with zero-padding."""
    x = torch.randn(1, 1, 1, 25)
    mel25 = MelPreprocessor(25, 80, 20.0, 800.0, TARGET_SR, device="cpu")
    mel160 = MelPreprocessor(160, 80, 20.0, 800.0, TARGET_SR, device="cpu")

    out25 = mel25.preprocess({"shake": {"audio": x}})
    padded = torch.cat([x, torch.zeros(1, 1, 1, 135)], dim=-1)
    out160 = mel160.preprocess({"shake": {"audio": padded}})

    assert not torch.allclose(
        out25["shake"]["audio"],
        out160["shake"]["audio"],
        atol=1e-4,
    )


def test_augmenter_config_nfft_experiment_override():
    from data_augmenter.augmenter_utils import AugmenterConfig

    config = {
        "modality_names": ["audio"],
        "location_names": ["shake"],
        "num_segments": 7,
        "n_fft": 160,
        "mel_bins": 80,
        "preprocess_mode": "fft",
        "loc_mod_spectrum_len": {"shake": {"audio": 256}},
    }
    experiment_config = {
        "model": "student_audio_deepsense_dw_large_mel",
        "preprocess_mode": "mel",
        "n_fft": 25,
        "fixed_augmenters": {
            "time_augmenters": ["no"],
            "freq_augmenters": ["no"],
        },
    }
    args = AugmenterConfig(config, experiment_config)
    assert args.dataset_config["n_fft"] == 25
    assert args.dataset_config["preprocess_mode"] == "mel"
