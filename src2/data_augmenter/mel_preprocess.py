"""
Manual Log-Mel preprocessor implemented from scratch using only PyTorch/NumPy.
No torchaudio or librosa dependencies.

Pipeline: time-domain -> FFT magnitude^2 -> Mel filterbank -> log -> [B, C, segments, n_mel]
"""

import torch
import math


def hz_to_mel(f):
    """Convert Hz to Mel scale (HTK formula).

    mel = 2595 * log10(1 + f/700)
    """
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    """Convert Mel to Hz (HTK formula).

    f = 700 * (10^(m/2595) - 1)
    """
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_mel_filterbank(n_fft: int, n_mel: int, fmin: float, fmax: float, sample_rate: int) -> torch.Tensor:
    """
    Build triangular mel filterbank matrix.

    Returns: shape [n_mel, n_fft//2 + 1], dtype float32, all values >= 0

    Convention:
    - n_mel+2 evenly spaced mel-scale center points from hz_to_mel(fmin) to hz_to_mel(fmax)
    - Create n_mel triangular filters between those points
    - FFT bin frequencies: k * sample_rate / n_fft for k in range(n_fft//2 + 1)
    - Each filter is a triangle peaking at its center mel frequency, going to zero at adjacent centers
    """
    n_freqs = n_fft // 2 + 1

    # Compute mel-scale center points: n_mel+2 points (includes edges)
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = [mel_min + i * (mel_max - mel_min) / (n_mel + 1) for i in range(n_mel + 2)]
    # Convert back to Hz
    hz_points = [mel_to_hz(m) for m in mel_points]

    # FFT bin frequencies
    fft_bin_freqs = [k * sample_rate / n_fft for k in range(n_freqs)]

    # Build filterbank matrix [n_mel, n_freqs]
    filterbank = torch.zeros(n_mel, n_freqs, dtype=torch.float32)

    for m in range(n_mel):
        f_left = hz_points[m]       # left edge (zero crossing)
        f_center = hz_points[m + 1]  # peak
        f_right = hz_points[m + 2]  # right edge (zero crossing)

        for k in range(n_freqs):
            f = fft_bin_freqs[k]
            if f_left <= f <= f_center:
                # Rising slope
                denom = f_center - f_left
                if denom > 0:
                    filterbank[m, k] = (f - f_left) / denom
                else:
                    filterbank[m, k] = 0.0
            elif f_center < f <= f_right:
                # Falling slope
                denom = f_right - f_center
                if denom > 0:
                    filterbank[m, k] = (f_right - f) / denom
                else:
                    filterbank[m, k] = 0.0
            # else: 0 (outside the triangle)

    return filterbank


class MelPreprocessor:
    def __init__(self, n_fft: int, n_mel: int, fmin: float, fmax: float, sample_rate: int, device='cpu'):
        self.n_fft = n_fft
        self.n_mel = n_mel
        self.device = device
        # Build and register filterbank as attribute (torch.Tensor on device)
        self.filterbank = build_mel_filterbank(n_fft, n_mel, fmin, fmax, sample_rate).to(device)

    def to(self, device):
        """Move filterbank to device."""
        self.device = device
        self.filterbank = self.filterbank.to(device)
        return self

    def preprocess(self, time_loc_inputs: dict) -> dict:
        """
        Apply log-mel preprocessing to time-domain inputs.

        Args:
            time_loc_inputs: dict[location][modality] = Tensor [B, C, segments, time_samples]

        Returns:
            mel_loc_inputs: dict[location][modality] = Tensor [B, C, segments, n_mel]

        Pipeline per tensor:
            1. fft = torch.fft.rfft(x, n=self.n_fft, dim=-1)  # [B, C, I, n_fft//2+1]
            2. power = fft.abs() ** 2                          # power spectrum
            3. mel = power @ self.filterbank.T                 # [B, C, I, n_mel]
            4. log_mel = torch.log(mel + 1e-6)                 # log with floor
        """
        mel_loc_inputs = {}
        for location, modality_dict in time_loc_inputs.items():
            mel_loc_inputs[location] = {}
            for modality, x in modality_dict.items():
                # x: [B, C, segments, time_samples]
                # Step 1: one-sided FFT -> [B, C, segments, n_fft//2+1]
                fft = torch.fft.rfft(x, n=self.n_fft, dim=-1)
                # Step 2: power spectrum
                power = fft.abs() ** 2
                # Step 3: apply mel filterbank -> [B, C, segments, n_mel]
                # filterbank: [n_mel, n_fft//2+1], so power @ filterbank.T
                mel = power @ self.filterbank.to(device=power.device, dtype=power.dtype).T
                # Step 4: log with epsilon floor
                log_mel = torch.log(mel + 1e-6)
                mel_loc_inputs[location][modality] = log_mel
        return mel_loc_inputs
