import torch
import torch.nn as nn
import numpy as np
from random import random, choice


class AudioNoiseAugmenter(nn.Module):
    """
    Unified audio noise augmenter supporting both soft and hard noise modes.
    Supports random intensity selection from a list during training.
    
    Soft Noise (Band-Limited Gaussian):
    - no_overlap: 500-800 Hz
    - some_overlap: 200-500 Hz  
    - full_spectrum: 0-800 Hz
    
    Hard Noise (Environmental):
    - wind: Low-frequency dominant with turbulence
    - rain: Mid-high frequency with droplet impacts
    """
    
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.config = args.dataset_config["audio_noise_augmentation"]
        
        # Target modality
        self.target_modality = self.config.get("target_modality", "audio")
        
        # Soft noise config
        self.soft_config = self.config.get("soft_noise", {})
        self.soft_enabled = self.soft_config.get("enabled", False)
        self.soft_prob = self.soft_config.get("prob", 0.5)
        self.soft_mode = self.soft_config.get("mode", "full_spectrum")
        self.soft_noise_std = self.soft_config.get("noise_std", 0.5)  # Can be float or list
        self.soft_modes = self.soft_config.get("modes", {
            "no_overlap": {"freq_range": [500, 800]},
            "some_overlap": {"freq_range": [200, 500]},
            "full_spectrum": {"freq_range": [0, 800]},
        })
        
        # Hard noise config
        self.hard_config = self.config.get("hard_noise", {})
        self.hard_enabled = self.hard_config.get("enabled", False)
        self.hard_prob = self.hard_config.get("prob", 0.5)
        self.hard_mode = self.hard_config.get("mode", "wind")
        self.wind_params = self.hard_config.get("wind", {})
        self.rain_params = self.hard_config.get("rain", {})
        
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
    
    def _sample_intensity(self, value):
        """
        Sample intensity value. If value is a list, randomly pick one.
        If it's a single value, return it directly.
        """
        if isinstance(value, (list, tuple)):
            return choice(value)
        return value
    
    def forward(self, org_loc_inputs, labels=None):
        """
        Apply noise augmentation to audio modality.
        Intensity is randomly sampled per batch if configured as a list.
        """
        aug_loc_inputs = {}
        aug_mod_labels = []
        b = None
        
        # Sample intensities for this batch
        current_soft_std = self._sample_intensity(self.soft_noise_std)
        current_hard_intensity = None
        if self.hard_mode == "wind":
            current_hard_intensity = self._sample_intensity(
                self.wind_params.get("intensity", 0.4)
            )
        elif self.hard_mode == "rain":
            current_hard_intensity = self._sample_intensity(
                self.rain_params.get("intensity", 0.5)
            )
        
        for loc in self.locations:
            aug_loc_inputs[loc] = {}
            for mod in self.modalities:
                if b is None:
                    b = org_loc_inputs[loc][mod].shape[0]
                
                # Only apply to target modality (audio)
                if mod == self.target_modality:
                    time_data = org_loc_inputs[loc][mod]
                    augmented = False
                    
                    # Apply soft noise
                    if self.soft_enabled and random() < self.soft_prob:
                        time_data = self._add_soft_noise(time_data, current_soft_std)
                        augmented = True
                    
                    # Apply hard noise
                    if self.hard_enabled and random() < self.hard_prob:
                        time_data = self._add_hard_noise(time_data, current_hard_intensity)
                        augmented = True
                    
                    aug_loc_inputs[loc][mod] = time_data
                    aug_mod_labels.append(1 if augmented else 0)
                else:
                    aug_loc_inputs[loc][mod] = org_loc_inputs[loc][mod]
                    aug_mod_labels.append(0)
        
        aug_mod_labels = torch.Tensor(aug_mod_labels).to(self.args.device)
        aug_mod_labels = aug_mod_labels.unsqueeze(0).tile([b, 1]).float()
        
        return aug_loc_inputs, aug_mod_labels, labels
    
    # =========================================================================
    # Soft Noise (Band-Limited Gaussian)
    # =========================================================================
    def _add_soft_noise(self, time_data, noise_std):
        """Add band-limited Gaussian noise with given intensity."""
        b, c, i, s = time_data.shape
        device = time_data.device
        
        # Get frequency range for current mode
        mode_config = self.soft_modes.get(self.soft_mode, {"freq_range": [0, 800]})
        freq_range = mode_config.get("freq_range", [0, 800])
        freq_start, freq_end = freq_range[0], freq_range[1]
        
        signal_std = time_data.std()
        
        # Create band-limited noise in frequency domain
        noise_freq = torch.zeros(b, c, i, s, dtype=torch.complex64, device=device)
        
        start_idx = max(0, min(freq_start, s))
        end_idx = max(0, min(freq_end, s))
        
        if start_idx < end_idx:
            noise_real = torch.randn(b, c, i, end_idx - start_idx, device=device)
            noise_imag = torch.randn(b, c, i, end_idx - start_idx, device=device)
            band_noise = torch.complex(noise_real, noise_imag)
            
            noise_freq[:, :, :, start_idx:end_idx] = band_noise
            
            # Conjugate symmetry for real output
            if end_idx < s:
                neg_start = s - end_idx
                neg_end = s - start_idx
                if neg_start > 0 and neg_end <= s:
                    noise_freq[:, :, :, neg_start:neg_end] = torch.flip(band_noise, dims=[-1]).conj()
        
        # IFFT to time domain
        time_noise = torch.fft.ifft(noise_freq, dim=-1).real
        
        # Scale noise with sampled intensity
        noise_std_actual = time_noise.std()
        if noise_std_actual > 1e-8:
            time_noise = time_noise * (noise_std * signal_std / noise_std_actual)
        
        return time_data + time_noise
    
    # =========================================================================
    # Hard Noise (Environmental)
    # =========================================================================
    def _add_hard_noise(self, time_data, intensity):
        """Add environmental noise (wind or rain) with given intensity."""
        if self.hard_mode == "wind":
            return self._add_wind_noise(time_data, intensity)
        elif self.hard_mode == "rain":
            return self._add_rain_noise(time_data, intensity)
        else:
            return time_data
    
    def _add_wind_noise(self, time_data, intensity):
        """Generate and add wind noise with low-frequency characteristics."""
        b, c, i, s = time_data.shape
        device = time_data.device
        
        # Get parameters (intensity is passed in, others from config)
        low_freq_weight = self.wind_params.get("low_freq_weight", 1.5)
        high_freq_cutoff = self.wind_params.get("high_freq_cutoff", 0.25)
        turbulence_factor = self.wind_params.get("turbulence_factor", 0.4)
        
        # Generate white noise
        noise = torch.randn(b, c, i, s, device=device)
        
        # Shape spectrum via FFT
        noise_fft = torch.fft.rfft(noise, dim=-1)
        num_freqs = noise_fft.shape[-1]
        
        # Create frequency-dependent weighting (low-freq dominant with rolloff)
        freqs = torch.linspace(0, 1, num_freqs, device=device)
        freq_weights = 1.0 / (1.0 + (freqs / (high_freq_cutoff + 0.01)) ** low_freq_weight)
        freq_weights = freq_weights.view(1, 1, 1, -1)
        
        noise_fft = noise_fft * freq_weights
        noise = torch.fft.irfft(noise_fft, n=s, dim=-1)
        
        # Add turbulence (amplitude modulation)
        if turbulence_factor > 0:
            t = torch.linspace(0, 2 * np.pi, i * s, device=device).reshape(1, 1, i, s)
            phase = torch.randn(1, 1, 1, 1, device=device) * 2 * np.pi
            modulation = 1.0 + turbulence_factor * torch.sin(0.1 * t + phase)
            noise = noise * modulation
        
        # Normalize and scale with sampled intensity
        noise = noise / (noise.std() + 1e-8)
        signal_std = time_data.std()
        
        return time_data + noise * intensity * signal_std
    
    def _add_rain_noise(self, time_data, intensity):
        """Generate and add rain noise with droplet characteristics."""
        b, c, i, s = time_data.shape
        device = time_data.device
        
        # Get parameters (intensity is passed in, others from config)
        droplet_rate = self.rain_params.get("droplet_rate", 0.3)
        high_freq_emphasis = self.rain_params.get("high_freq_emphasis", 2.0)
        
        # Generate white noise
        noise = torch.randn(b, c, i, s, device=device)
        
        # Shape spectrum via FFT (emphasize high frequencies)
        noise_fft = torch.fft.rfft(noise, dim=-1)
        num_freqs = noise_fft.shape[-1]
        
        freqs = torch.linspace(0, 1, num_freqs, device=device)
        freq_weights = (freqs + 0.1) ** (high_freq_emphasis - 1.0)
        freq_weights = freq_weights / freq_weights.mean()
        freq_weights = freq_weights.view(1, 1, 1, -1)
        
        noise_fft = noise_fft * freq_weights
        noise = torch.fft.irfft(noise_fft, n=s, dim=-1)
        
        # Normalize before applying envelope
        noise = noise / (noise.std() + 1e-8)
        
        # Add temporal sparsity (droplet impacts)
        if droplet_rate > 0:
            envelope = torch.rand(b, c, i, s, device=device)
            envelope = (envelope > droplet_rate).float()
            envelope = envelope * 0.7 + 0.3
            noise = noise * envelope
        
        # Scale with sampled intensity
        signal_std = time_data.std()
        
        return time_data + noise * intensity * signal_std
