"""
Script to investigate the shapes of seismic and audio samples.

This script loads a batch of data and prints the shapes of the data tensors
to understand the data structure, both before and after FFT augmentation.

USAGE:
    cd /home/misra8/sensing-nn/src2/data_analysis
    python investigating_shape.py --yaml_path ../data/Parkland.yaml --model ResNet
    
    # Or from any directory:
    python /home/misra8/sensing-nn/src2/data_analysis/investigating_shape.py \
        --yaml_path /home/misra8/sensing-nn/src2/data/Parkland.yaml \
        --model ResNet

OUTPUT:
    - Shows raw data shapes (time domain)
    - Shows augmented data shapes (frequency domain after FFT)
    - Detailed dimension breakdown and interpretation
    - Summary of the complete transformation pipeline
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import get_config
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_augmenter import create_augmenter, apply_augmentation


def plot_spectrogram_like_visualization(augmented_data, labels, num_samples=3, save_dir=None):
    """
    Plot the FFT output as a 2D spectrogram-like visualization.
    
    Args:
        augmented_data: Dictionary of augmented data after FFT
        labels: Labels for the samples
        num_samples: Number of samples to visualize
        save_dir: Directory to save the plots
    """
    if save_dir is None:
        save_dir = Path(__file__).parent.parent / "tmp"
    else:
        save_dir = Path(save_dir)
    
    # Create tmp directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating spectrogram-like visualizations...")
    print(f"Saving to: {save_dir}")
    
    for loc in augmented_data:
        for mod in augmented_data[loc]:
            data = augmented_data[loc][mod]  # Shape: [batch, 2, segments, freq_bins]
            
            # Compute magnitude from real and imaginary parts
            # data[:, 0, :, :] = real part
            # data[:, 1, :, :] = imaginary part
            real = data[:, 0, :, :]  # [batch, segments, freq_bins]
            imag = data[:, 1, :, :]  # [batch, segments, freq_bins]
            magnitude = torch.sqrt(real**2 + imag**2)  # [batch, segments, freq_bins]
            
            # Convert to numpy for plotting
            magnitude_np = magnitude.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Plot first few samples
            num_to_plot = min(num_samples, magnitude_np.shape[0])
            
            for sample_idx in range(num_to_plot):
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                sample_data = magnitude_np[sample_idx]  # [segments, freq_bins]
                sample_label = labels_np[sample_idx]
                
                # Plot 1: Full spectrogram-like view
                ax = axes[0]
                im = ax.imshow(sample_data, aspect='auto', origin='lower', cmap='viridis')
                ax.set_xlabel('Frequency Bin', fontsize=12)
                ax.set_ylabel('Segment Number', fontsize=12)
                ax.set_title(f'{mod.capitalize()} - Sample {sample_idx} (Label: {sample_label})\n' + 
                           f'Y-axis: Segments (10), X-axis: Frequency bins', fontsize=14)
                plt.colorbar(im, ax=ax, label='Magnitude')
                
                # Plot 2: Log scale for better visualization
                ax = axes[1]
                log_data = np.log10(sample_data + 1e-10)  # Add small epsilon to avoid log(0)
                im = ax.imshow(log_data, aspect='auto', origin='lower', cmap='viridis')
                ax.set_xlabel('Frequency Bin', fontsize=12)
                ax.set_ylabel('Segment Number', fontsize=12)
                ax.set_title(f'{mod.capitalize()} - Sample {sample_idx} (Label: {sample_label})\n' + 
                           f'Log Scale - Better contrast', fontsize=14)
                plt.colorbar(im, ax=ax, label='Log10(Magnitude)')
                
                plt.tight_layout()
                
                # Save figure
                filename = f"{loc}_{mod}_sample{sample_idx}_label{sample_label}.png"
                save_path = save_dir / filename
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"  Saved: {filename}")
                plt.close(fig)
                
                # Also create a single frequency spectrum plot for middle segment
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                middle_segment = sample_data.shape[0] // 2
                ax.plot(sample_data[middle_segment, :], linewidth=1.5)
                ax.set_xlabel('Frequency Bin', fontsize=12)
                ax.set_ylabel('Magnitude', fontsize=12)
                ax.set_title(f'{mod.capitalize()} - Sample {sample_idx} (Label: {sample_label})\n' + 
                           f'Frequency Spectrum of Segment {middle_segment}', fontsize=14)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                filename = f"{loc}_{mod}_sample{sample_idx}_segment{middle_segment}_spectrum.png"
                save_path = save_dir / filename
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
    
    print(f"\n✓ Visualizations saved to: {save_dir}")


def investigate_data_shapes(config):
    """
    Load data and print shapes of seismic and audio samples.
    
    Args:
        config: Configuration dictionary
    """
    print("=" * 80)
    print("INVESTIGATING DATA SHAPES")
    print("=" * 80)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create augmenter (same as in training)
    print("\nCreating augmenter...")
    augmenter = create_augmenter(config, augmentation_mode="fixed")
    print("  Augmenter created successfully")
    
    # Get one batch from train loader
    print("\n" + "=" * 80)
    print("TRAIN DATA - First Batch (BEFORE AUGMENTATION)")
    print("=" * 80)
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Unpack batch (following the same pattern as in train_test_utils.py)
        if len(batch_data) == 3:
            data, labels, idx = batch_data
        else:
            data, labels = batch_data[0], batch_data[1]
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels dtype: {labels.dtype}")
        print(f"  Number of samples in batch: {labels.shape[0]}")
        
        # Check if data is a dictionary (multi-modal)
        if isinstance(data, dict):
            print(f"\n  Data structure: Multi-modal dictionary")
            print(f"  Locations: {list(data.keys())}")
            
            # Iterate through locations and modalities
            for loc in data:
                print(f"\n  Location: '{loc}'")
                print(f"    Modalities: {list(data[loc].keys())}")
                
                for mod in data[loc]:
                    shape = data[loc][mod].shape
                    dtype = data[loc][mod].dtype
                    print(f"    {mod}:")
                    print(f"      Shape: {shape}")
                    print(f"      Dtype: {dtype}")
                    print(f"      Dimensions: [batch_size, channels, height, width] or [batch_size, channels, time_steps]")
                    
                    # Print some statistics
                    print(f"      Min value: {data[loc][mod].min().item():.4f}")
                    print(f"      Max value: {data[loc][mod].max().item():.4f}")
                    print(f"      Mean value: {data[loc][mod].mean().item():.4f}")
        else:
            print(f"\n  Data structure: Single tensor")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
        
        # Save the original data for augmentation
        data_for_augmentation = data
        labels_for_augmentation = labels
        
        # Only process first batch
        break
    
    # Now apply augmentation and show the results
    print("\n" + "=" * 80)
    print("TRAIN DATA - First Batch (AFTER AUGMENTATION - FFT Applied)")
    print("=" * 80)
    print("\nApplying augmentation (this is what goes into the model)...")
    print("Note: FFT is applied along the last dimension (time → frequency)")
    
    # Apply augmentation (same as in training)
    augmented_data, augmented_labels = apply_augmentation(augmenter, data_for_augmentation, labels_for_augmentation)
    
    print(f"\nBatch 1 (After Augmentation):")
    print(f"  Labels shape: {augmented_labels.shape}")
    print(f"  Labels dtype: {augmented_labels.dtype}")
    print(f"  Number of samples in batch: {augmented_labels.shape[0]}")
    
    # Check if data is a dictionary (multi-modal)
    if isinstance(augmented_data, dict):
        print(f"\n  Data structure: Multi-modal dictionary")
        print(f"  Locations: {list(augmented_data.keys())}")
        
        # Iterate through locations and modalities
        for loc in augmented_data:
            print(f"\n  Location: '{loc}'")
            print(f"    Modalities: {list(augmented_data[loc].keys())}")
            
            for mod in augmented_data[loc]:
                shape = augmented_data[loc][mod].shape
                dtype = augmented_data[loc][mod].dtype
                print(f"    {mod}:")
                print(f"      Shape: {shape}")
                print(f"      Dtype: {dtype}")
                
                # Detailed dimension explanation
                b, c, seg, freq = shape
                print(f"\n      DIMENSION BREAKDOWN:")
                print(f"        [0] Batch size: {b}")
                print(f"        [1] Channels: {c} (Real + Imaginary parts of FFT)")
                print(f"        [2] Segments: {seg} (signal divided into {seg} chunks)")
                print(f"        [3] Frequency bins: {freq} (FFT output, DC to Nyquist)")
                print(f"\n      INTERPRETATION:")
                print(f"        - Signal is divided into {seg} segments")
                print(f"        - Each segment has {freq} frequency bins after FFT")
                print(f"        - FFT produces complex output → 2 channels (real & imaginary)")
                print(f"        - NOT a spectrogram (no overlapping windows)")
                
                # Print some statistics
                print(f"\n      STATISTICS:")
                print(f"        Min value: {augmented_data[loc][mod].min().item():.4f}")
                print(f"        Max value: {augmented_data[loc][mod].max().item():.4f}")
                print(f"        Mean value: {augmented_data[loc][mod].mean().item():.4f}")
                
                # Compare with original shape
                if loc in data_for_augmentation and mod in data_for_augmentation[loc]:
                    original_shape = data_for_augmentation[loc][mod].shape
                    if original_shape != shape:
                        print(f"\n      TRANSFORMATION:")
                        print(f"        Before: {original_shape}")
                        print(f"                [batch, channels, segments, time_steps]")
                        print(f"        After:  {shape}")
                        print(f"                [batch, channels(real+imag), segments, freq_bins]")
                        print(f"        ⚠️  Time domain ({original_shape[-1]} samples) → Frequency domain ({freq} bins)")
                    else:
                        print(f"      ✓ Shape unchanged from original")
    else:
        print(f"\n  Data structure: Single tensor")
        print(f"  Shape: {augmented_data.shape}")
        print(f"  Dtype: {augmented_data.dtype}")
    
    # Create spectrogram-like visualizations
    print("\n" + "=" * 80)
    print("CREATING SPECTROGRAM-LIKE VISUALIZATIONS")
    print("=" * 80)
    print("\nHypothesis: Can we interpret this as a 2D spectrogram?")
    print("  Y-axis: Segment number (10 segments)")
    print("  X-axis: Frequency bins (1600 for audio, 20 for seismic)")
    print("  Value: Magnitude = sqrt(real^2 + imag^2)")
    
    tmp_dir = Path(__file__).parent.parent / "tmp"
    plot_spectrogram_like_visualization(augmented_data, augmented_labels, num_samples=3, save_dir=tmp_dir)
    
    # Get one batch from validation loader
    print("\n" + "=" * 80)
    print("VALIDATION DATA - First Batch")
    print("=" * 80)
    
    for batch_idx, batch_data in enumerate(val_loader):
        # Unpack batch
        if len(batch_data) == 3:
            data, labels, idx = batch_data
        else:
            data, labels = batch_data[0], batch_data[1]
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Labels shape: {labels.shape}")
        
        if isinstance(data, dict):
            for loc in data:
                print(f"\n  Location: '{loc}'")
                for mod in data[loc]:
                    shape = data[loc][mod].shape
                    print(f"    {mod} shape: {shape}")
        else:
            print(f"  Data shape: {data.shape}")
        
        # Only process first batch
        break
    
    # Get one batch from test loader
    print("\n" + "=" * 80)
    print("TEST DATA - First Batch")
    print("=" * 80)
    
    for batch_idx, batch_data in enumerate(test_loader):
        # Unpack batch
        if len(batch_data) == 3:
            data, labels, idx = batch_data
        else:
            data, labels = batch_data[0], batch_data[1]
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Labels shape: {labels.shape}")
        
        if isinstance(data, dict):
            for loc in data:
                print(f"\n  Location: '{loc}'")
                for mod in data[loc]:
                    shape = data[loc][mod].shape
                    print(f"    {mod} shape: {shape}")
        else:
            print(f"  Data shape: {data.shape}")
        
        # Only process first batch
        break
    
    print("\n" + "=" * 80)
    print("SUMMARY: DATA TRANSFORMATION PIPELINE")
    print("=" * 80)
    
    # Generate summary based on what we saw
    print("\n1. RAW DATA (Time Domain):")
    print("   - Audio: [batch, 1, 10 segments, 1600 time_steps]")
    print("   - Seismic: [batch, 1, 10 segments, 20 time_steps]")
    
    print("\n2. AFTER FFT (Frequency Domain) → What goes into the model:")
    print("   - Audio: [batch, 2, 10 segments, 1600 freq_bins]")
    print("   - Seismic: [batch, 2, 10 segments, 20 freq_bins]")
    
    print("\n3. KEY TRANSFORMATIONS:")
    print("   ✓ Channels: 1 → 2 (complex FFT output split into real + imaginary)")
    print("   ✓ Last dim: time_steps → frequency_bins (1D FFT along time axis)")
    print("   ✓ Segments: preserved (each segment transformed independently)")
    
    print("\n4. WHAT EACH DIMENSION MEANS:")
    print("   - Dimension 0: Batch size (number of samples)")
    print("   - Dimension 1: Channels (2 = real and imaginary parts)")
    print("   - Dimension 2: Segments (signal divided into chunks)")
    print("   - Dimension 3: Frequency bins (FFT output from DC to Nyquist)")
    
    print("\n5. IMPORTANT NOTES:")
    print("   ⚠️  This is NOT a spectrogram")
    print("   ⚠️  Each segment is transformed independently (no overlap)")
    print("   ⚠️  Frequency bins = original time steps (FFT property)")
    print("   ⚠️  Model must expect 2 input channels, not 1")
    
    print("\n" + "=" * 80)
    print("SHAPE INVESTIGATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Load configuration
    config = get_config()
    
    print(f"Configuration loaded:")
    print(f"  Dataset: {config.get('yaml_path', 'Unknown')}")
    print(f"  Batch size: {config.get('batch_size', 'Unknown')}")
    print(f"  Locations: {config.get('location_names', 'Unknown')}")
    print(f"  Modalities: {config.get('modality_names', 'Unknown')}")
    
    # Investigate shapes
    investigate_data_shapes(config)

