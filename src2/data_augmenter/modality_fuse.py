"""Fuse mel tensors from multiple modalities (e.g. audio + seismic)."""

import logging

import torch


def fuse_channel_concat(
    mel_loc_inputs: dict,
    location: str,
    audio_modality: str = "audio",
    seismic_modality: str = "seismic",
    pad_channels: int = 0,
    output_modality: str = "audio",
) -> dict:
    """
    Concatenate single-channel audio and seismic mel along channel dim.

    Expects mel tensors [B, 1, segments, mel_bins] per modality.
    Output: [B, 2 + pad_channels, segments, mel_bins] under output_modality.
    """
    if pad_channels < 0:
        raise ValueError(f"pad_channels must be >= 0, got {pad_channels}")

    if location not in mel_loc_inputs:
        raise ValueError(f"location {location!r} not in mel_loc_inputs")

    loc_dict = mel_loc_inputs[location]
    if audio_modality not in loc_dict:
        raise ValueError(f"audio modality {audio_modality!r} missing at {location}")
    if seismic_modality not in loc_dict:
        raise ValueError(f"seismic modality {seismic_modality!r} missing at {location}")

    audio = loc_dict[audio_modality]
    seismic = loc_dict[seismic_modality]
    if audio.dim() != 4 or seismic.dim() != 4:
        raise ValueError(
            f"fuse_channel_concat expects 4-D mel tensors, got audio {tuple(audio.shape)}, "
            f"seismic {tuple(seismic.shape)}"
        )
    if audio.shape[0] != seismic.shape[0]:
        raise ValueError("audio and seismic batch sizes must match")
    if audio.shape[2:] != seismic.shape[2:]:
        raise ValueError(
            f"audio and seismic segment/mel shapes must match: "
            f"{tuple(audio.shape)} vs {tuple(seismic.shape)}"
        )

    parts = [audio]
    if pad_channels > 0:
        pad = torch.zeros(
            audio.shape[0],
            pad_channels,
            audio.shape[2],
            audio.shape[3],
            dtype=audio.dtype,
            device=audio.device,
        )
        parts.append(pad)
    parts.append(seismic)
    fused = torch.cat(parts, dim=1)

    out = {loc: dict(mod_dict) for loc, mod_dict in mel_loc_inputs.items()}
    out[location][output_modality] = fused
    logging.debug(
        "Fused %s+%s -> %s shape %s (pad_channels=%d)",
        audio_modality,
        seismic_modality,
        output_modality,
        tuple(fused.shape),
        pad_channels,
    )
    return out
