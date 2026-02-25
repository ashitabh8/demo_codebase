import torch
import numpy as np
from pathlib import Path


# For running in web browser - run ssh -L 8888:localhost:8888 misra8@acies1 - this will let you visualize
# the generated html


# ============================================================================
# Noise Generation Functions
# ============================================================================

def generate_wind_noise(shape, intensity=0.5, low_freq_weight=2.0, 
                        high_freq_cutoff=0.3, turbulence_factor=0.3):
    """
    Generate wind noise with tunable spectral characteristics.
    Wind noise has dominant low-frequency components with rolloff at higher frequencies.
    
    Args:
        shape: Tuple of (batch, channels, segments, samples_per_segment)
        intensity: Overall noise level (0.0-1.0), controls SNR relative to signal
        low_freq_weight: Emphasis on low frequencies (0.5-5.0), higher = more bass
        high_freq_cutoff: Normalized frequency where rolloff starts (0.0-1.0 of Nyquist)
        turbulence_factor: Amplitude modulation to simulate wind gusts (0.0-1.0)
    
    Returns:
        torch.Tensor: Wind noise with shape matching input
    """
    b, c, i, s = shape
    
    # Generate white noise in time domain
    noise = torch.randn(b, c, i, s)
    
    # Apply FFT to shape spectrum
    noise_fft = torch.fft.rfft(noise, dim=-1)
    
    # Create frequency-dependent weighting for wind characteristics
    num_freqs = noise_fft.shape[-1]
    freqs = torch.linspace(0, 1, num_freqs)  # Normalized frequency (0 to Nyquist)
    
    # Wind noise: strong low frequencies with rolloff
    # weight(f) = 1 / (1 + (f/f_cutoff)^low_freq_weight)
    freq_weights = 1.0 / (1.0 + (freqs / (high_freq_cutoff + 0.01)) ** low_freq_weight)
    freq_weights = freq_weights.view(1, 1, 1, -1)
    
    # Apply spectral shaping
    noise_fft = noise_fft * freq_weights
    
    # Convert back to time domain
    noise = torch.fft.irfft(noise_fft, n=s, dim=-1)
    
    # Add turbulence (amplitude modulation to simulate gusts)
    if turbulence_factor > 0:
        # Create slow modulation envelope
        turbulence_freq = 0.1  # Slow modulation
        t = torch.linspace(0, 2 * np.pi, i * s).reshape(1, 1, i, s)
        modulation = 1.0 + turbulence_factor * torch.sin(turbulence_freq * t + torch.randn(1, 1, 1, 1) * 2 * np.pi)
        noise = noise * modulation
    
    # Normalize to unit variance before scaling
    noise = noise / (noise.std() + 1e-8)
    
    return noise * intensity


def generate_rain_noise(shape, intensity=0.5, droplet_rate=0.3,
                        high_freq_emphasis=1.5, splash_sharpness=0.5):
    """
    Generate rain noise with tunable spectral characteristics.
    Rain noise has broadband characteristics with emphasis on mid-high frequencies
    and temporal sparsity to simulate individual droplet impacts.
    
    Args:
        shape: Tuple of (batch, channels, segments, samples_per_segment)
        intensity: Overall noise level (0.0-1.0), controls SNR relative to signal
        droplet_rate: Controls temporal sparsity (0.0=continuous, 1.0=very sparse)
        high_freq_emphasis: Boost for higher frequencies (0.5-3.0)
        splash_sharpness: Controls transient characteristics (0.0-1.0)
    
    Returns:
        torch.Tensor: Rain noise with shape matching input
    """
    b, c, i, s = shape
    
    # Generate white noise
    noise = torch.randn(b, c, i, s)
    
    # Apply FFT for spectral shaping
    noise_fft = torch.fft.rfft(noise, dim=-1)
    
    # Create frequency-dependent weighting for rain characteristics
    num_freqs = noise_fft.shape[-1]
    freqs = torch.linspace(0, 1, num_freqs)  # Normalized frequency
    
    # Rain noise: emphasis on mid-high frequencies
    # Boost higher frequencies with a power law
    freq_weights = (freqs + 0.1) ** (high_freq_emphasis - 1.0)
    freq_weights = freq_weights / freq_weights.mean()  # Normalize
    freq_weights = freq_weights.view(1, 1, 1, -1)
    
    # Apply spectral shaping
    noise_fft = noise_fft * freq_weights
    
    # Convert back to time domain
    noise = torch.fft.irfft(noise_fft, n=s, dim=-1)
    
    # Normalize to unit variance BEFORE applying envelope
    # This prevents extreme peaks when sparse envelope is applied
    noise = noise / (noise.std() + 1e-8)
    
    # Add temporal sparsity (droplet impacts)
    if droplet_rate > 0:
        # Create sparse temporal envelope
        envelope = torch.rand(b, c, i, s)
        threshold = droplet_rate  # Higher droplet_rate = more sparse
        envelope = (envelope > threshold).float()
        
        # Add transient shaping (short attack, quick decay)
        if splash_sharpness > 0:
            # Create exponential decay for each "droplet"
            decay_length = int(s * 0.05 * (1.0 - splash_sharpness))  # Sharper = shorter decay
            for seg_idx in range(i):
                seg_envelope = envelope[:, :, seg_idx, :]
                # Apply smoothing with exponential decay
                for sample_idx in range(s):
                    if seg_envelope[0, 0, sample_idx] > 0:
                        decay = torch.exp(-torch.arange(min(decay_length, s - sample_idx), dtype=torch.float32) / (decay_length / 3.0))
                        end_idx = min(sample_idx + len(decay), s)
                        envelope[:, :, seg_idx, sample_idx:end_idx] *= decay[:end_idx - sample_idx].view(1, 1, -1)
        
        noise = noise * envelope
    
    return noise * intensity


def add_noise_to_sample(sample, modality, noise_type, **noise_params):
    """
    Add noise to a single sample's time-domain data.
    
    Args:
        sample: Dictionary with structure {'data': {'shake': {modality: tensor}}, ...}
        modality: The modality to augment (e.g., 'audio')
        noise_type: Type of noise ('wind' or 'rain')
        **noise_params: Parameters to pass to the noise generation function
    
    Returns:
        Dictionary: Augmented sample with same structure as input
    """
    # Deep copy to avoid modifying original
    import copy
    augmented_sample = copy.deepcopy(sample)
    
    # Extract time-domain data
    time_data = augmented_sample['data']['shake'][modality]
    
    # Ensure 4D shape: [batch, channels, segments, samples]
    if time_data.dim() < 4:
        original_shape = time_data.shape
        time_data = time_data.reshape(time_data.shape[0], 1, 10, -1)  # Assuming 10 segments
    else:
        original_shape = None
    
    # Compute signal statistics for scaling
    signal_std = time_data.std()
    
    # Generate noise with appropriate spectral characteristics
    if noise_type == 'wind':
        noise = generate_wind_noise(time_data.shape, **noise_params)
    elif noise_type == 'rain':
        noise = generate_rain_noise(time_data.shape, **noise_params)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Use 'wind' or 'rain'.")
    
    # Scale noise relative to signal amplitude
    noise = noise * signal_std
    
    # Add noise to signal
    augmented_data = time_data + noise
    
    # Restore original shape if needed
    if original_shape is not None:
        augmented_data = augmented_data.reshape(original_shape)
    
    # Update the sample
    augmented_sample['data']['shake'][modality] = augmented_data
    
    # Add metadata about augmentation
    if 'augmentation' not in augmented_sample:
        augmented_sample['augmentation'] = {}
    augmented_sample['augmentation'][modality] = {
        'noise_type': noise_type,
        'params': noise_params
    }
    
    return augmented_sample


def augment_class_wise_samples_with_noise(class_wise_samples, modality, noise_type, **noise_params):
    """
    Apply noise augmentation to all samples in class-wise dictionary.
    
    Args:
        class_wise_samples: Dictionary with structure {class_name: [samples]}
        modality: The modality to augment (e.g., 'audio')
        noise_type: Type of noise ('wind' or 'rain')
        **noise_params: Parameters to pass to the noise generation function
    
    Returns:
        Dictionary: Augmented class-wise samples with same structure as input
    """
    augmented_class_wise_samples = {}
    
    for class_name, samples in class_wise_samples.items():
        augmented_samples = []
        for sample in samples:
            augmented_sample = add_noise_to_sample(sample, modality, noise_type, **noise_params)
            augmented_samples.append(augmented_sample)
        augmented_class_wise_samples[class_name] = augmented_samples
    
    print(f"Augmented {sum(len(samples) for samples in class_wise_samples.values())} samples with {noise_type} noise")
    print(f"Noise parameters: {noise_params}")
    
    return augmented_class_wise_samples


def get_time_sample_for_modality(sample, modality):
    sample_data = sample['data']
    sample_modality = sample_data['shake'][modality]
    return sample_modality

def fft_preprocess(time_loc_inputs, modality):
        """Run FFT on the time-domain input.
        time_loc_inputs: [b, c, i, s]
        freq_loc_inputs: [b, c, i, s]
        
        Dimensions:
            b = batch size
            c = channels (becomes c1*c2 after FFT for real/imaginary components)
            i = number of time segments (i.e. number of frames)
            s = number of samples per segment (i.e. number of samples per frame, becomes frequency bins after FFT)
        
        Args:
            time_loc_inputs: dict of time-domain signals
            modality: modality of the data
        """
        freq_loc_inputs = dict()
        # importance_scores = dict() if return_importance_scores else None
        time_loc_inputs = time_loc_inputs['data']
        # breakpoint()

        for loc in time_loc_inputs:
            freq_loc_inputs[loc] = dict()
            # breakpoint()

            if time_loc_inputs[loc][modality].dim() < 4:
                # Assume 10 segments by default (standard for this dataset)
                time_loc_inputs[loc][modality] = time_loc_inputs[loc][modality].reshape(time_loc_inputs[loc][modality].shape[0], 1, 10, -1)
            loc_mod_freq_output = torch.fft.fft(time_loc_inputs[loc][modality], dim=-1)
            loc_mod_freq_output = torch.view_as_real(loc_mod_freq_output)
            loc_mod_freq_output = loc_mod_freq_output.permute(0, 1, 4, 2, 3)
            b, c1, c2, i, s = loc_mod_freq_output.shape
            loc_mod_freq_output = loc_mod_freq_output.reshape(b, c1 * c2, i, s)
            freq_loc_inputs[loc][modality] = loc_mod_freq_output
        return freq_loc_inputs

def load_class_wise_samples(path):
    """
    Load the class-wise samples from the path.
    """
    return torch.load(path)


def print_class_wise_samples(class_wise_samples):
    for class_name, samples in class_wise_samples.items():
        print(f"Class: {class_name}")
        print(f"Number of samples: {len(samples)}")


def get_all_freq_class_wise_for_modality(class_wise_samples, modality):
    class_wise_spectrogram = dict()
    for class_name, samples in class_wise_samples.items():
        class_wise_spectrogram[class_name] = dict()
        class_wise_spectrogram[class_name] = []
        for sample in samples:
            output = fft_preprocess(sample, modality)
            sample = {'data': output, 'labels': sample['label'], 'class_name': sample['class_name'], 'idx': sample['idx']}
            class_wise_spectrogram[class_name].append(sample)
    return class_wise_spectrogram


def plot_all_freq_class_wise_for_modality(freq_class_wise_samples, modality):
    """
    Create an interactive HTML visualization of spectrograms for all classes.
    
    Args:
        freq_class_wise_samples: dict with class_name -> list of samples
                                 Each sample has 'data' -> {'shake' -> {modality -> tensor}}
        modality: the modality to visualize (e.g., 'audio')
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Get all class names
    class_names = list(freq_class_wise_samples.keys())
    
    # Create buttons for each class
    buttons = []
    
    # Prepare all spectrograms
    all_spectrograms = []
    
    for class_idx, class_name in enumerate(class_names):
        samples = freq_class_wise_samples[class_name]
        
        for sample_idx, sample in enumerate(samples):
            # Extract spectrogram: data -> shake -> modality
            spectrogram = sample['data']['shake'][modality]
            
            # Convert to numpy and take magnitude for visualization
            # Shape should be [batch, channels, frames, freq_bins]
            spec_np = spectrogram.squeeze().cpu().numpy()
            
            # If we have multiple channels (e.g., real/imaginary), compute magnitude
            if spec_np.ndim == 3:  # [channels, frames, freq_bins]
                # Average across channels or take magnitude
                spec_np = np.mean(np.abs(spec_np), axis=0)
            elif spec_np.ndim == 2:  # [frames, freq_bins]
                spec_np = np.abs(spec_np)
            else:
                spec_np = np.abs(spec_np.reshape(-1, spec_np.shape[-1]))
            
            # Apply log scaling for better visualization (compresses dynamic range)
            spec_np = np.log10(spec_np + 1.0)  # Add 1 to avoid log(0)
            
            # Store for plotting
            all_spectrograms.append({
                'data': spec_np.T,  # Transpose so freq_bins are on y-axis
                'class_name': class_name,
                'sample_idx': sample_idx,
                'idx': sample.get('idx', sample_idx),
                'class_idx': class_idx
            })
    
    # Determine color scale range using percentiles to avoid outliers
    all_data = np.concatenate([s['data'].flatten() for s in all_spectrograms])
    zmin = 0
    zmax = np.percentile(all_data, 99.9)  # Use 99.9th percentile to keep detail while clipping extremes
    print(f"Color scale range for clean data (log10): [0, {zmax:.2f}] (99.9th percentile, log-scaled for visibility)")
    
    # Create the figure with subplots - 2 rows, 5 columns (10 samples per class)
    fig = make_subplots(
        rows=2, cols=5,
        subplot_titles=[f"Sample {i+1}" for i in range(10)],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Add all traces (hidden initially)
    trace_idx = 0
    for class_idx, class_name in enumerate(class_names):
        class_samples = [s for s in all_spectrograms if s['class_name'] == class_name]
        
        for sample_idx in range(10):
            if sample_idx < len(class_samples):
                spec_data = class_samples[sample_idx]['data']
                
                row = (sample_idx // 5) + 1
                col = (sample_idx % 5) + 1
                
                trace = go.Heatmap(
                    z=spec_data,
                    colorscale='Viridis',
                    visible=(class_idx == 0),  # Only first class visible initially
                    showscale=(sample_idx == 0),  # Show colorbar only for first subplot
                    name=f"{class_name} - Sample {sample_idx + 1}",
                    hovertemplate='Frame: %{x}<br>Freq Bin: %{y}<br>Log Magnitude: %{z:.2f}<extra></extra>',
                    zmin=zmin,
                    zmax=zmax,
                    colorbar=dict(title="log10(Mag)")
                )
                
                fig.add_trace(trace, row=row, col=col)
    
    # Create buttons for class selection
    for class_idx, class_name in enumerate(class_names):
        # Determine visibility for each trace
        visible = [False] * len(fig.data)
        start_idx = class_idx * 10
        end_idx = start_idx + 10
        for i in range(start_idx, min(end_idx, len(fig.data))):
            visible[i] = True
        
        button = dict(
            label=class_name,
            method="update",
            args=[{"visible": visible},
                  {"title": f"Spectrograms for Class: {class_name} ({modality})"}]
        )
        buttons.append(button)
    
    # Update layout
    fig.update_layout(
        title=f"Spectrograms for Class: {class_names[0]} ({modality})",
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.5,
                xanchor="center",
                y=1.15,
                yanchor="top",
                buttons=buttons,
                bgcolor="lightgray",
                bordercolor="gray",
                font=dict(size=12)
            )
        ],
        height=800,
        showlegend=False,
        template="plotly_white"
    )
    
    # Update axes labels
    for i in range(1, 11):
        row = ((i-1) // 5) + 1
        col = ((i-1) % 5) + 1
        fig.update_xaxes(title_text="Time Frame", row=row, col=col)
        fig.update_yaxes(title_text="Freq Bin", row=row, col=col)
    
    # Save to HTML file
    output_path = "/home/misra8/sensing-nn/src2/data_analysis/spectrograms_visualization.html"
    fig.write_html(output_path)
    print(f"Interactive visualization saved to: {output_path}")
    print(f"Open this file in a web browser to explore the spectrograms.")
    
    return fig


def plot_clean_vs_noisy_spectrograms(clean_freq_samples, noisy_freq_samples, modality, noise_type, noise_params=None):
    """
    Create an interactive HTML visualization comparing clean and noisy spectrograms side-by-side.
    
    Args:
        clean_freq_samples: dict with class_name -> list of clean frequency-domain samples
        noisy_freq_samples: dict with class_name -> list of noisy frequency-domain samples
        modality: the modality to visualize (e.g., 'audio')
        noise_type: type of noise applied ('wind' or 'rain')
        noise_params: dictionary of noise parameters for display in title
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Get all class names
    class_names = list(clean_freq_samples.keys())
    
    # Create buttons for each class
    buttons = []
    
    # Prepare all spectrograms (clean and noisy pairs)
    all_spectrograms = []
    
    for class_idx, class_name in enumerate(class_names):
        clean_samples = clean_freq_samples[class_name]
        noisy_samples = noisy_freq_samples[class_name]
        
        for sample_idx in range(min(len(clean_samples), len(noisy_samples), 5)):  # Show 5 samples per class
            # Extract clean spectrogram
            clean_spec = clean_samples[sample_idx]['data']['shake'][modality]
            clean_np = clean_spec.squeeze().cpu().numpy()
            if clean_np.ndim == 3:
                clean_np = np.mean(np.abs(clean_np), axis=0)
            elif clean_np.ndim == 2:
                clean_np = np.abs(clean_np)
            else:
                clean_np = np.abs(clean_np.reshape(-1, clean_np.shape[-1]))
            
            # Apply log scaling for better visualization
            clean_np = np.log10(clean_np + 1.0)
            
            # Extract noisy spectrogram
            noisy_spec = noisy_samples[sample_idx]['data']['shake'][modality]
            noisy_np = noisy_spec.squeeze().cpu().numpy()
            if noisy_np.ndim == 3:
                noisy_np = np.mean(np.abs(noisy_np), axis=0)
            elif noisy_np.ndim == 2:
                noisy_np = np.abs(noisy_np)
            else:
                noisy_np = np.abs(noisy_np.reshape(-1, noisy_np.shape[-1]))
            
            # Apply log scaling for better visualization
            noisy_np = np.log10(noisy_np + 1.0)
            
            # Store for plotting
            all_spectrograms.append({
                'clean_data': clean_np.T,  # Transpose so freq_bins are on y-axis
                'noisy_data': noisy_np.T,
                'class_name': class_name,
                'sample_idx': sample_idx,
                'idx': clean_samples[sample_idx].get('idx', sample_idx),
                'class_idx': class_idx
            })
    
    # Create the figure with subplots - 2 rows, 5 columns (top=clean, bottom=noisy)
    fig = make_subplots(
        rows=2, cols=5,
        subplot_titles=[f"Clean - Sample {i+1}" if i < 5 else f"Noisy - Sample {i-4}" 
                       for i in range(10)],
        vertical_spacing=0.12,
        horizontal_spacing=0.05,
        row_titles=['Clean', 'Noisy']
    )
    
    # Determine global color scale range using percentiles to avoid outliers
    # This prevents extreme values from stretching the color scale
    all_clean = np.concatenate([s['clean_data'].flatten() for s in all_spectrograms])
    all_noisy = np.concatenate([s['noisy_data'].flatten() for s in all_spectrograms])
    all_data = np.concatenate([all_clean, all_noisy])
    
    zmin = 0  # Start from 0 for magnitude
    zmax = np.percentile(all_data, 99.9)  # Use 99.9th percentile to keep detail while clipping extremes
    
    print(f"Color scale range (log10): [0, {zmax:.2f}] (99.9th percentile, log-scaled for visibility)")
    
    # Add all traces (hidden initially)
    for class_idx, class_name in enumerate(class_names):
        class_samples = [s for s in all_spectrograms if s['class_name'] == class_name]
        
        for sample_idx in range(5):
            if sample_idx < len(class_samples):
                clean_data = class_samples[sample_idx]['clean_data']
                noisy_data = class_samples[sample_idx]['noisy_data']
                
                col = sample_idx + 1
                
                # Clean spectrogram (top row)
                clean_trace = go.Heatmap(
                    z=clean_data,
                    colorscale='Viridis',
                    visible=(class_idx == 0),
                    showscale=(sample_idx == 4),  # Show colorbar only for last subplot
                    name=f"{class_name} - Clean {sample_idx + 1}",
                    hovertemplate='Frame: %{x}<br>Freq Bin: %{y}<br>Log Magnitude: %{z:.2f}<extra></extra>',
                    zmin=zmin,
                    zmax=zmax,
                    colorbar=dict(title="log10(Mag)")
                )
                fig.add_trace(clean_trace, row=1, col=col)
                
                # Noisy spectrogram (bottom row)
                noisy_trace = go.Heatmap(
                    z=noisy_data,
                    colorscale='Viridis',
                    visible=(class_idx == 0),
                    showscale=(sample_idx == 4),  # Show colorbar only for last subplot
                    name=f"{class_name} - Noisy {sample_idx + 1}",
                    hovertemplate='Frame: %{x}<br>Freq Bin: %{y}<br>Log Magnitude: %{z:.2f}<extra></extra>',
                    zmin=zmin,
                    zmax=zmax,
                    colorbar=dict(title="log10(Mag)")
                )
                fig.add_trace(noisy_trace, row=2, col=col)
    
    # Create buttons for class selection
    for class_idx, class_name in enumerate(class_names):
        # Determine visibility for each trace (10 traces per class: 5 clean + 5 noisy)
        visible = [False] * len(fig.data)
        start_idx = class_idx * 10
        end_idx = start_idx + 10
        for i in range(start_idx, min(end_idx, len(fig.data))):
            visible[i] = True
        
        button = dict(
            label=class_name,
            method="update",
            args=[{"visible": visible},
                  {"title": f"Clean vs {noise_type.capitalize()} Noise - Class: {class_name} ({modality})"}]
        )
        buttons.append(button)
    
    # Format noise parameters for title
    params_str = ""
    if noise_params:
        params_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                               for k, v in noise_params.items()])
        params_str = f" [{params_str}]"
    
    # Update layout
    fig.update_layout(
        title=f"Clean vs {noise_type.capitalize()} Noise - Class: {class_names[0]} ({modality}){params_str}",
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.5,
                xanchor="center",
                y=1.08,
                yanchor="top",
                buttons=buttons,
                bgcolor="lightgray",
                bordercolor="gray",
                font=dict(size=12)
            )
        ],
        height=900,
        showlegend=False,
        template="plotly_white"
    )
    
    # Update axes labels
    for row in [1, 2]:
        for col in range(1, 6):
            fig.update_xaxes(title_text="Time Frame", row=row, col=col, title_font=dict(size=10))
            fig.update_yaxes(title_text="Freq Bin", row=row, col=col, title_font=dict(size=10))
    
    # Save to HTML file
    output_path = f"/home/misra8/sensing-nn/src2/data_analysis/spectrograms_{noise_type}_comparison.html"
    fig.write_html(output_path)
    print(f"Interactive comparison visualization saved to: {output_path}")
    print(f"Open this file in a web browser to compare clean vs {noise_type} noise spectrograms.")
    
    return fig


if __name__ == "__main__":
    path = "/home/misra8/sensing-nn/src2/data_analysis/class_samples.pt"
    modality = "audio"
    
    print("=" * 80)
    print("Audio Noise Augmentation - Wind and Rain Noise Analysis")
    print("=" * 80)
    
    # Load original samples
    print("\n1. Loading class-wise samples...")
    class_wise_samples = load_class_wise_samples(path)
    print_class_wise_samples(class_wise_samples)
    
    # Generate clean spectrograms
    print("\n2. Generating clean spectrograms...")
    clean_freq = get_all_freq_class_wise_for_modality(class_wise_samples, modality)
    
    # Test Wind Noise with tunable parameters
    print("\n3. Testing WIND NOISE augmentation...")
    print("-" * 80)
    wind_params = {
        'intensity': 0.4,           # Moderate noise level
        'low_freq_weight': 1.5,     # Strong low-frequency emphasis
        'high_freq_cutoff': 0.25,   # Rolloff starts at 25% of Nyquist
        'turbulence_factor': 0.4    # Moderate wind gusts
    }
    wind_noisy_samples = augment_class_wise_samples_with_noise(
        class_wise_samples, modality, "wind", **wind_params
    )
    wind_freq = get_all_freq_class_wise_for_modality(wind_noisy_samples, modality)
    print("Wind noise visualization...")
    plot_clean_vs_noisy_spectrograms(clean_freq, wind_freq, modality, "wind", wind_params)
    
    # Test Rain Noise with tunable parameters
    print("\n4. Testing RAIN NOISE augmentation...")
    print("-" * 80)
    rain_params = {
        'intensity': 0.2,            # Higher noise level
        'droplet_rate': 0.5,         # Medium sparsity
        'high_freq_emphasis': 2.0,   # Strong high-frequency emphasis
        'splash_sharpness': 0.6      # Sharp transients
    }
    rain_noisy_samples = augment_class_wise_samples_with_noise(
        class_wise_samples, modality, "rain", **rain_params
    )
    rain_freq = get_all_freq_class_wise_for_modality(rain_noisy_samples, modality)
    print("Rain noise visualization...")
    plot_clean_vs_noisy_spectrograms(clean_freq, rain_freq, modality, "rain", rain_params)
    
    # Optional: Generate original clean visualization for reference
    print("\n5. Generating original clean spectrograms for reference...")
    plot_all_freq_class_wise_for_modality(clean_freq, modality)
    
    print("\n" + "=" * 80)
    print("COMPLETED! Generated visualizations:")
    print("  - spectrograms_visualization.html (clean baseline)")
    print("  - spectrograms_wind_comparison.html (clean vs wind noise)")
    print("  - spectrograms_rain_comparison.html (clean vs rain noise)")
    print("\nOpen these HTML files in your browser to explore the spectrograms.")
    print("=" * 80)
    print("\nParameter tuning guide:")
    print("  Wind Noise:")
    print("    - intensity: 0.1-0.8 (controls overall noise level)")
    print("    - low_freq_weight: 1.0-4.0 (higher = more bass)")
    print("    - high_freq_cutoff: 0.1-0.5 (where high freqs roll off)")
    print("    - turbulence_factor: 0.0-0.7 (wind gust variation)")
    print("\n  Rain Noise:")
    print("    - intensity: 0.1-0.8 (controls overall noise level)")
    print("    - droplet_rate: 0.0-0.8 (higher = more sparse)")
    print("    - high_freq_emphasis: 1.0-3.0 (boost high frequencies)")
    print("    - splash_sharpness: 0.0-0.9 (sharper = shorter transients)")
    print("=" * 80)
