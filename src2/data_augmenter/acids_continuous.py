"""
ACIDS continuous tail-cut: flatten segmented audio, keep flat[:used_samples].

One cut from the end of the concatenated clip — no per-segment trimming.
"""

import logging

import torch


class AcidsContinuousTruncator:
    """Flatten [B, C, segments, padded_seg_len] and truncate to used_samples."""

    def __init__(
        self,
        num_segments: int,
        padded_seg_len: int,
        used_samples: int,
        target_modalities=None,
    ):
        if target_modalities is None:
            target_modalities = ["audio"]
        self.num_segments = int(num_segments)
        self.padded_seg_len = int(padded_seg_len)
        self.used_samples = int(used_samples)
        self.target_modalities = set(target_modalities)
        expected_flat = self.num_segments * self.padded_seg_len
        logging.info(
            "AcidsContinuousTruncator: flat len %d -> stream[:%d], modalities=%s",
            expected_flat,
            self.used_samples,
            sorted(self.target_modalities),
        )

    def truncate(self, time_loc_inputs: dict) -> dict:
        """Return new dict; targeted modalities become [B, C, used_samples]."""
        out = {}
        for loc, mod_dict in time_loc_inputs.items():
            out[loc] = {}
            for mod, x in mod_dict.items():
                if mod in self.target_modalities:
                    out[loc][mod] = self._truncate_tensor(x)
                else:
                    out[loc][mod] = x
        return out

    def _truncate_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, segments, padded_seg_len] -> [B, C, used_samples]."""
        if x.dim() != 4:
            raise ValueError(
                f"AcidsContinuousTruncator expects 4-D input, got shape {tuple(x.shape)}"
            )
        b, c, segments, t = x.shape
        if segments != self.num_segments:
            raise ValueError(
                f"expected num_segments={self.num_segments}, got {segments}"
            )
        if t != self.padded_seg_len:
            raise ValueError(
                f"expected padded_seg_len={self.padded_seg_len}, got {t}"
            )
        flat = x.reshape(b, c, segments * t)
        return flat[..., : self.used_samples]
