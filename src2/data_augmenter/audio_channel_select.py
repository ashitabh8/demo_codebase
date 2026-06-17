"""Select a single audio microphone channel before preprocessing."""

import logging


def select_audio_channel(
    time_loc_inputs: dict,
    channel_index: int,
    target_modality: str = "audio",
    num_channels: int = 3,
) -> dict:
    """
    Keep one audio channel; other modalities are unchanged.

    Expects 4-D tensors [B, C, segments, T] (channel in dim 1). Also accepts
    [B, 1, C, segments, T] is NOT supported — use standard layout only.
    """
    if channel_index < 0 or channel_index >= num_channels:
        raise ValueError(
            f"audio_channel_index ({channel_index}) out of range for "
            f"num_channels={num_channels}"
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
                    f"select_audio_channel expects 4-D input for {loc}/{mod}, "
                    f"got shape {tuple(x.shape)}"
                )
            c = x.shape[1]
            if c == num_channels:
                out[loc][mod] = x[:, channel_index : channel_index + 1, ...]
            elif c == 1 and x.shape[2] == num_channels:
                # [B, 1, C, T] legacy layout
                out[loc][mod] = x[:, :, channel_index : channel_index + 1, :]
            else:
                raise ValueError(
                    f"Cannot select audio channel on {loc}/{mod}: shape {tuple(x.shape)}, "
                    f"expected C={num_channels} on dim 1 or dim 2"
                )
    logging.debug(
        "Selected audio channel %d (modality=%s)", channel_index, target_modality
    )
    return out
