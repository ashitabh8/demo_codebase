"""
Anti-aliased audio decimation for the augmenter preprocessing path.

Pipeline per targeted modality: trim to multiple of factor -> FIR low-pass -> subsample.
"""

import logging

import torch
import torch.nn.functional as F
from scipy.signal import firwin


class AudioDownsampler:
    """Decimate time-domain tensors from input_sr to target_sr (integer factor only)."""

    def __init__(
        self,
        input_sr: int,
        target_sr: int,
        target_modalities=None,
        device="cpu",
    ):
        if target_modalities is None:
            target_modalities = ["audio"]
        self.input_sr = int(input_sr)
        self.target_sr = int(target_sr)
        self.target_modalities = set(target_modalities)
        self.device = device

        if self.input_sr % self.target_sr != 0:
            raise ValueError(
                f"input_sample_rate ({self.input_sr}) must be an integer multiple of "
                f"sample_rate ({self.target_sr})"
            )

        self.factor = self.input_sr // self.target_sr
        self.enabled = self.factor > 1
        self.kernel = None
        self.pad = 0
        self._trim_warned = False

        if not self.enabled:
            return

        cutoff_hz = 0.45 * self.target_sr
        numtaps = 101
        if numtaps % 2 == 0:
            numtaps += 1
        coeffs = firwin(numtaps, cutoff_hz, fs=self.input_sr)
        self.pad = numtaps // 2
        self.kernel = (
            torch.tensor(coeffs, dtype=torch.float32).view(1, 1, -1).to(device)
        )
        logging.info(
            "AudioDownsampler: %d Hz -> %d Hz (factor %d), modalities=%s, "
            "FIR cutoff=%.1f Hz, taps=%d",
            self.input_sr,
            self.target_sr,
            self.factor,
            sorted(self.target_modalities),
            cutoff_hz,
            numtaps,
        )

    def to(self, device):
        self.device = device
        if self.kernel is not None:
            self.kernel = self.kernel.to(device)
        return self

    def downsample(self, time_loc_inputs: dict) -> dict:
        """Return a new dict; targeted modalities are decimated, others are unchanged."""
        if not self.enabled:
            return time_loc_inputs

        out = {}
        for loc, mod_dict in time_loc_inputs.items():
            out[loc] = {}
            for mod, x in mod_dict.items():
                if mod in self.target_modalities:
                    out[loc][mod] = self._decimate(x)
                else:
                    out[loc][mod] = x
        return out

    def _decimate(self, x: torch.Tensor) -> torch.Tensor:
        """Decimate [B, C, segments, T] along the time axis."""
        b, c, segments, t = x.shape
        trim = t % self.factor
        if trim and not self._trim_warned:
            logging.warning(
                "AudioDownsampler: trimming last %d sample(s) per segment so T=%d "
                "is divisible by factor %d",
                trim,
                t,
                self.factor,
            )
            self._trim_warned = True
        if trim:
            x = x[..., :-trim]
            t = t - trim

        flat = x.reshape(b * c * segments, 1, t)
        filtered = F.conv1d(flat, self.kernel, padding=self.pad)
        decimated = filtered[:, :, :: self.factor]
        new_t = t // self.factor
        return decimated.reshape(b, c, segments, new_t)
