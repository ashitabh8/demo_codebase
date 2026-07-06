"""Select a single microphone or sensor channel before preprocessing."""

import logging


def select_modality_channel(
    time_loc_inputs: dict,
    channel_index: int,
    target_modality: str,
    num_channels: int,
) -> dict:
    """
    Keep one channel for target_modality; other modalities are unchanged.

    Expects 4-D tensors [B, C, segments, T] (channel in dim 1).
    """
    if channel_index < 0 or channel_index >= num_channels:
        raise ValueError(
            f"channel_index ({channel_index}) out of range for "
            f"modality={target_modality!r} num_channels={num_channels}"
        )

    out = {}
    for loc, mod_dict in time_loc_inputs.items():
        out[loc] = {}
        for mod, x in mod_dict.items():
            if mod != target_modality:
                out[loc][mod] = x
                continue
            if x.dim() != 4:
                raise ValueError(
                    f"select_modality_channel expects 4-D input for {loc}/{mod}, "
                    f"got shape {tuple(x.shape)}"
                )
            c = x.shape[1]
            if c == num_channels:
                out[loc][mod] = x[:, channel_index : channel_index + 1, ...]
            elif c == 1 and x.shape[2] == num_channels:
                out[loc][mod] = x[:, :, channel_index : channel_index + 1, :]
            else:
                raise ValueError(
                    f"Cannot select channel on {loc}/{mod}: shape {tuple(x.shape)}, "
                    f"expected C={num_channels} on dim 1 or dim 2"
                )
    logging.debug(
        "Selected %s channel %d of %d", target_modality, channel_index, num_channels
    )
    return out


def select_audio_channel(
    time_loc_inputs: dict,
    channel_index: int,
    target_modality: str = "audio",
    num_channels: int = 3,
) -> dict:
    """Keep one audio channel; other modalities are unchanged."""
    return select_modality_channel(
        time_loc_inputs, channel_index, target_modality, num_channels
    )


def select_seismic_channel(
    time_loc_inputs: dict,
    channel_index: int,
    target_modality: str = "seismic",
    num_channels: int = 2,
) -> dict:
    """Keep one seismic channel; other modalities are unchanged."""
    return select_modality_channel(
        time_loc_inputs, channel_index, target_modality, num_channels
    )
