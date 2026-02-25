import torch
import torch.nn as nn
from random import random


class BandLimitedNoiseAugmenter(nn.Module):
    """
    Band-limited Gaussian noise injection in TIME DOMAIN (audio only).
    
    Process:
    1. Generate noise in frequency domain for target band
    2. IFFT to convert noise to time domain (band-limited noise)
    3. Add band-limited noise to original time signal
    
    Noise ranges:
    - no_overlap: 500-800 Hz (outside main signal)
    - some_overlap: 200-500 Hz (partial overlap)
    - full_spectrum: 0-800 Hz (entire range)
    """
    
    NOISE_RANGES = {
        "no_overlap": (500, 800),
        "some_overlap": (200, 500),
        "full_spectrum": (0, 800),
    }
    
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["band_limited_noise"]
        self.p = self.config["prob"]
        self.noise_std = self.config["noise_std"]
        self.noise_range = self.config["noise_range"]  # "no_overlap", "some_overlap", "full_spectrum"
        self.target_modality = self.config.get("target_modality", "audio")  # Only apply to audio
        
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        
        if self.noise_range not in self.NOISE_RANGES:
            raise ValueError(f"Invalid noise_range: {self.noise_range}. Choose from {list(self.NOISE_RANGES.keys())}")
        
        self.freq_start, self.freq_end = self.NOISE_RANGES[self.noise_range]
    
    def forward(self, org_loc_inputs, labels=None):
        """
        Add band-limited noise to audio modality in time domain.
        
        Args:
            org_loc_inputs: Dict {loc: {mod: tensor [b, c, i, s]}}
            labels: Optional labels
            
        Returns:
            aug_loc_inputs: Augmented inputs
            aug_mod_labels: Modality-level augmentation labels
            labels: Original labels
        """
        aug_loc_inputs = {}
        aug_mod_labels = []
        b = None
        
        for loc in self.locations:
            aug_loc_inputs[loc] = {}
            for mod in self.modalities:
                if b is None:
                    b = org_loc_inputs[loc][mod].shape[0]
                
                # Only apply noise to target modality (audio)
                if mod == self.target_modality and random() < self.p:
                    time_data = org_loc_inputs[loc][mod]  # [b, c, i, s]
                    noisy_data = self._add_band_limited_noise(time_data)
                    aug_loc_inputs[loc][mod] = noisy_data
                    aug_mod_labels.append(1)
                else:
                    aug_loc_inputs[loc][mod] = org_loc_inputs[loc][mod]
                    aug_mod_labels.append(0)
        
        aug_mod_labels = torch.Tensor(aug_mod_labels).to(self.args.device)
        aug_mod_labels = aug_mod_labels.unsqueeze(0).tile([b, 1]).float()
        
        return aug_loc_inputs, aug_mod_labels, labels
    
    def _add_band_limited_noise(self, time_data):
        """
        Add band-limited noise to time-domain data.
        
        Args:
            time_data: [b, c, i, s] time domain tensor
            
        Returns:
            noisy_data: [b, c, i, s] time domain tensor with noise added
        """
        b, c, i, s = time_data.shape
        device = time_data.device
        
        # Signal statistics for noise scaling
        signal_std = time_data.std()
        
        # Create band-limited noise in frequency domain
        noise_freq = torch.zeros(b, c, i, s, dtype=torch.complex64, device=device)
        
        start_idx = max(0, min(self.freq_start, s))
        end_idx = max(0, min(self.freq_end, s))
        
        if start_idx < end_idx:
            # Generate complex Gaussian noise for target band
            noise_real = torch.randn(b, c, i, end_idx - start_idx, device=device)
            noise_imag = torch.randn(b, c, i, end_idx - start_idx, device=device)
            band_noise = torch.complex(noise_real, noise_imag)
            
            # Place noise in positive frequencies
            noise_freq[:, :, :, start_idx:end_idx] = band_noise
            
            # Mirror to negative frequencies for real output (conjugate symmetry)
            if end_idx < s:
                neg_start = s - end_idx
                neg_end = s - start_idx
                if neg_start > 0 and neg_end <= s:
                    noise_freq[:, :, :, neg_start:neg_end] = torch.flip(band_noise, dims=[-1]).conj()
        
        # IFFT to get time-domain noise (band-limited)
        time_noise = torch.fft.ifft(noise_freq, dim=-1).real
        
        # Scale noise relative to signal
        noise_std_actual = time_noise.std()
        if noise_std_actual > 1e-8:
            time_noise = time_noise * (self.noise_std * signal_std / noise_std_actual)
        
        # Add noise in time domain
        noisy_data = time_data + time_noise
        
        return noisy_data
