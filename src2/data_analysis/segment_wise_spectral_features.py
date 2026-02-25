"""
Spectral Energy Analysis Script

This script analyzes spectral energy across frequency bands for each vehicle class.
It computes statistics (mean, std, min, max, quartiles) and visualizes the results.
"""

import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import get_config
from dataset_utils.MultiModalDataLoader import create_single_dataloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def compute_fft_and_energy(time_data, dim=-1):
    """
    Apply FFT to time-domain data and compute spectral energy.
    
    Args:
        time_data: Time-domain tensor [batch, channels, segments, samples]
        dim: Dimension along which to apply FFT (default: -1, the sample dimension)
    
    Returns:
        energy: Spectral energy tensor [batch, segments, freq_bins]
    """
    # Apply FFT along the sample dimension
    freq_data = torch.fft.fft(time_data, dim=dim)
    
    # Convert to real representation (real and imaginary parts)
    freq_data = torch.view_as_real(freq_data)  # [..., samples, 2]
    
    # Compute energy: real^2 + imag^2
    energy = freq_data[..., 0]**2 + freq_data[..., 1]**2  # [..., samples]
    
    # Reshape to [batch, segments, freq_bins]
    if time_data.dim() == 4:
        # Input was [batch, channels, segments, samples]
        batch_size, channels, segments, freq_bins = time_data.shape
        # Average across channels if multiple
        energy = energy.mean(dim=1)  # [batch, segments, freq_bins]
    
    return energy


def compute_frequency_band_energy(energy, num_bands=4, sampling_rate=8000):
    """
    Split spectral energy into frequency bands and compute band-wise energy.
    
    Args:
        energy: Spectral energy [batch, segments, freq_bins]
        num_bands: Number of frequency bands to split into
        sampling_rate: Sampling rate in Hz
    
    Returns:
        band_energy: Energy per band [batch, segments, num_bands]
        band_ranges: List of (start_freq, end_freq) tuples for each band
    """
    batch_size, segments, freq_bins = energy.shape
    bins_per_band = freq_bins // num_bands
    
    # Calculate actual frequency ranges
    freq_resolution = sampling_rate / freq_bins
    band_ranges = []
    for i in range(num_bands):
        start_freq = i * bins_per_band * freq_resolution
        end_freq = (i + 1) * bins_per_band * freq_resolution
        band_ranges.append((start_freq, end_freq))
    
    # Reshape to group frequency bins into bands
    # [batch, segments, freq_bins] -> [batch, segments, num_bands, bins_per_band]
    energy_trimmed = energy[:, :, :num_bands * bins_per_band]
    energy_bands = energy_trimmed.reshape(batch_size, segments, num_bands, bins_per_band)
    
    # Average within each band
    band_energy = energy_bands.mean(dim=3)  # [batch, segments, num_bands]
    
    return band_energy, band_ranges


def collect_class_wise_spectral_features(dataloader, config, num_bands=4):
    """
    Collect spectral energy features for all samples, organized by class.
    Uses lazy loading to avoid memory overflow.
    
    Args:
        dataloader: PyTorch DataLoader
        config: Configuration dictionary
        num_bands: Number of frequency bands
    
    Returns:
        class_features: Dict mapping class_idx -> list of band energies [segments, num_bands]
    """
    class_features = defaultdict(list)
    
    # Get configuration
    location = config.get("location_names", ["shake"])[0]
    modality = config.get("modality_names", ["seismic"])[0]  # Use first modality
    sampling_rate = config.get("sampling_rate", 8000)
    
    logging.info(f"Collecting features from location: {location}, modality: {modality}")
    logging.info(f"Using {num_bands} frequency bands")
    
    # Process batches
    for batch_idx, (data, labels, _) in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Extract time-domain data for the specific location and modality
        time_data = data[location][modality]  # [batch, channels, segments, samples]
        
        # Ensure 4D shape
        if time_data.dim() < 4:
            time_data = time_data.reshape(
                time_data.shape[0], 1, 
                config.get("num_segments", 10), -1
            )
        
        # Compute FFT and spectral energy
        energy = compute_fft_and_energy(time_data)  # [batch, segments, freq_bins]
        
        # Split into frequency bands
        band_energy, band_ranges = compute_frequency_band_energy(
            energy, num_bands=num_bands, sampling_rate=sampling_rate
        )  # [batch, segments, num_bands]
        
        # Organize by class
        for i in range(len(labels)):
            class_idx = labels[i].item()
            sample_band_energy = band_energy[i].cpu().numpy()  # [segments, num_bands]
            class_features[class_idx].append(sample_band_energy)
    
    # Store band ranges for later use
    _, band_ranges = compute_frequency_band_energy(
        torch.zeros(1, 10, 1600), num_bands=num_bands, sampling_rate=sampling_rate
    )
    
    logging.info(f"Collected features for {len(class_features)} classes")
    for class_idx, features in class_features.items():
        logging.info(f"  Class {class_idx}: {len(features)} samples")
    
    return class_features, band_ranges


def compute_statistics_per_class(class_features, num_bands):
    """
    Compute statistics (mean, std, min, max, quartiles) for each class and band.
    
    Args:
        class_features: Dict mapping class_idx -> list of [segments, num_bands] arrays
        num_bands: Number of frequency bands
    
    Returns:
        stats: Dict with statistics for each class and band
    """
    stats = {}
    
    for class_idx, features_list in class_features.items():
        # Stack all samples: [num_samples, segments, num_bands]
        features_array = np.stack(features_list, axis=0)
        
        # Average across segments to get per-sample band energy: [num_samples, num_bands]
        features_per_sample = features_array.mean(axis=1)
        
        # Compute statistics for each band
        class_stats = {}
        for band_idx in range(num_bands):
            band_data = features_per_sample[:, band_idx]
            
            class_stats[band_idx] = {
                'mean': np.mean(band_data),
                'std': np.std(band_data),
                'min': np.min(band_data),
                'max': np.max(band_data),
                'q25': np.percentile(band_data, 25),
                'q75': np.percentile(band_data, 75),
                'median': np.median(band_data),
                'raw_data': band_data  # Keep for plotting
            }
        
        stats[class_idx] = class_stats
    
    return stats


def plot_band_comparison_across_classes(stats, band_ranges, class_names, output_dir):
    """
    Create plots comparing all classes for each frequency band.
    
    Args:
        stats: Statistics dictionary from compute_statistics_per_class
        band_ranges: List of (start_freq, end_freq) tuples
        class_names: List of class names
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_bands = len(band_ranges)
    num_classes = len(stats)
    
    # Create a plot for each frequency band
    for band_idx in range(num_bands):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        start_freq, end_freq = band_ranges[band_idx]
        band_name = f"{int(start_freq)}-{int(end_freq)}Hz"
        
        # Collect data for all classes
        class_indices = sorted(stats.keys())
        means = []
        stds = []
        mins = []
        maxs = []
        q25s = []
        q75s = []
        labels = []
        
        for class_idx in class_indices:
            band_stats = stats[class_idx][band_idx]
            means.append(band_stats['mean'])
            stds.append(band_stats['std'])
            mins.append(band_stats['min'])
            maxs.append(band_stats['max'])
            q25s.append(band_stats['q25'])
            q75s.append(band_stats['q75'])
            
            # Get class name
            if class_idx < len(class_names):
                labels.append(class_names[class_idx])
            else:
                labels.append(f"Class {class_idx}")
        
        means = np.array(means)
        stds = np.array(stds)
        mins = np.array(mins)
        maxs = np.array(maxs)
        q25s = np.array(q25s)
        q75s = np.array(q75s)
        
        x = np.arange(len(labels))
        width = 0.6
        
        # Plot mean with error bars (std)
        ax.bar(x, means, width, label='Mean', alpha=0.7, color='steelblue')
        ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='black', 
                   capsize=5, capthick=2, label='Std Dev')
        
        # Plot quartiles as markers
        ax.plot(x, q25s, 'v', color='green', markersize=8, label='25th percentile', alpha=0.7)
        ax.plot(x, q75s, '^', color='orange', markersize=8, label='75th percentile', alpha=0.7)
        
        # Plot min/max as small markers
        ax.plot(x, mins, '_', color='red', markersize=10, markeredgewidth=2, 
               label='Min', alpha=0.5)
        ax.plot(x, maxs, '_', color='darkred', markersize=10, markeredgewidth=2, 
               label='Max', alpha=0.5)
        
        # Formatting
        ax.set_xlabel('Vehicle Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Spectral Energy', fontsize=12, fontweight='bold')
        ax.set_title(f'Spectral Energy Comparison - Frequency Band: {band_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / f"band_{band_idx}_{band_name.replace('-', '_')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot: {output_file}")
        plt.close()
    
    logging.info(f"All plots saved to: {output_dir}")


def plot_all_bands_per_class(stats, band_ranges, class_names, output_dir):
    """
    Create plots showing all frequency bands for each class.
    
    Args:
        stats: Statistics dictionary
        band_ranges: List of (start_freq, end_freq) tuples
        class_names: List of class names
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_bands = len(band_ranges)
    
    # Create a plot for each class
    for class_idx in sorted(stats.keys()):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get class name
        if class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f"Class {class_idx}"
        
        # Collect data for all bands
        band_labels = []
        means = []
        stds = []
        mins = []
        maxs = []
        q25s = []
        q75s = []
        
        for band_idx in range(num_bands):
            start_freq, end_freq = band_ranges[band_idx]
            band_labels.append(f"{int(start_freq)}-{int(end_freq)}Hz")
            
            band_stats = stats[class_idx][band_idx]
            means.append(band_stats['mean'])
            stds.append(band_stats['std'])
            mins.append(band_stats['min'])
            maxs.append(band_stats['max'])
            q25s.append(band_stats['q25'])
            q75s.append(band_stats['q75'])
        
        means = np.array(means)
        stds = np.array(stds)
        mins = np.array(mins)
        maxs = np.array(maxs)
        q25s = np.array(q25s)
        q75s = np.array(q75s)
        
        x = np.arange(len(band_labels))
        width = 0.6
        
        # Plot mean with error bars
        ax.bar(x, means, width, label='Mean', alpha=0.7, color='steelblue')
        ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='black', 
                   capsize=5, capthick=2, label='Std Dev')
        
        # Plot quartiles
        ax.plot(x, q25s, 'v', color='green', markersize=8, label='25th percentile', alpha=0.7)
        ax.plot(x, q75s, '^', color='orange', markersize=8, label='75th percentile', alpha=0.7)
        
        # Plot min/max
        ax.plot(x, mins, '_', color='red', markersize=10, markeredgewidth=2, 
               label='Min', alpha=0.5)
        ax.plot(x, maxs, '_', color='darkred', markersize=10, markeredgewidth=2, 
               label='Max', alpha=0.5)
        
        # Formatting
        ax.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
        ax.set_ylabel('Spectral Energy', fontsize=12, fontweight='bold')
        ax.set_title(f'Spectral Energy Across Frequency Bands - {class_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / f"class_{class_idx}_{class_name.replace(' ', '_')}_all_bands.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot: {output_file}")
        plt.close()


def main():
    """Main analysis function."""
    
    logging.info("=" * 80)
    logging.info("SPECTRAL ENERGY ANALYSIS")
    logging.info("=" * 80)
    
    # Load configuration
    config = get_config()
    logging.info("Configuration loaded successfully")
    
    # Parameters
    num_bands = 8  # Split into 8 frequency bands for finer resolution (1000Hz each)
    split = "train"  # Analyze training data
    
    # Create dataloader
    logging.info(f"\nCreating dataloader for {split} split...")
    dataloader = create_single_dataloader(
        config, 
        split=split, 
        batch_size=config.get("batch_size", 128),
        num_workers=config.get("num_workers", 4),
        use_balanced_sampling=False  # Don't need balancing for analysis
    )
    
    # Collect class-wise spectral features
    logging.info("\nCollecting spectral features...")
    class_features, band_ranges = collect_class_wise_spectral_features(
        dataloader, config, num_bands=num_bands
    )
    
    # Compute statistics
    logging.info("\nComputing statistics...")
    stats = compute_statistics_per_class(class_features, num_bands)
    
    # Print statistics
    logging.info("\n" + "=" * 80)
    logging.info("STATISTICS SUMMARY")
    logging.info("=" * 80)
    
    class_names = config.get("vehicle_classification", {}).get("class_names", [])
    
    for class_idx in sorted(stats.keys()):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
        logging.info(f"\n{class_name} (Class {class_idx}):")
        
        for band_idx in range(num_bands):
            start_freq, end_freq = band_ranges[band_idx]
            band_name = f"{int(start_freq)}-{int(end_freq)}Hz"
            band_stats = stats[class_idx][band_idx]
            
            logging.info(f"  {band_name}:")
            logging.info(f"    Mean:   {band_stats['mean']:.6f}")
            logging.info(f"    Std:    {band_stats['std']:.6f}")
            logging.info(f"    Min:    {band_stats['min']:.6f}")
            logging.info(f"    Max:    {band_stats['max']:.6f}")
            logging.info(f"    Q25:    {band_stats['q25']:.6f}")
            logging.info(f"    Median: {band_stats['median']:.6f}")
            logging.info(f"    Q75:    {band_stats['q75']:.6f}")
    
    # Create output directory
    output_dir = Path(config.get("base_experiment_dir", "experiments")) / "spectral_analysis"
    
    # Generate plots
    logging.info("\n" + "=" * 80)
    logging.info("GENERATING PLOTS")
    logging.info("=" * 80)
    
    # Plot 1: Compare all classes for each frequency band
    logging.info("\nCreating band comparison plots (all classes per band)...")
    plot_band_comparison_across_classes(stats, band_ranges, class_names, 
                                       output_dir / "band_comparison")
    
    # Plot 2: Show all bands for each class
    logging.info("\nCreating per-class plots (all bands per class)...")
    plot_all_bands_per_class(stats, band_ranges, class_names, 
                            output_dir / "per_class")
    
    logging.info("\n" + "=" * 80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Results saved to: {output_dir}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
