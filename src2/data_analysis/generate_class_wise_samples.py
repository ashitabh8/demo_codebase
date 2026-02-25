import sys
import yaml
import torch
from pathlib import Path

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import get_config
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_augmenter import create_augmenter, apply_augmentation
from models.create_models import create_model
from train_test.loss import get_loss_function
from train_test.train_test_utils import (
    setup_experiment_dir, train, setup_optimizer, setup_scheduler
)
from train_test.normalize import setup_normalization


def get_parklad_samples_per_class(dataloader, config, device, num_samples_per_class):
    """
    Collect random samples per class from the dataloader.
    Stops early once num_samples_per_class is collected for all classes.
    
    Args:
        dataloader: PyTorch DataLoader
        config: Configuration dictionary
        device: Device to use
        augmenter: Data augmenter object
        apply_augmentation_fn: Function to apply augmentation
        num_samples_per_class: Number of samples to collect per class
    
    Returns:
        Dictionary with class names as keys and lists of samples as values
        Each sample is a dictionary with 'data', 'label', and optionally 'idx'
    """
    # Get class names from config
    class_names = config['vehicle_classification']['class_names']
    num_classes = config['vehicle_classification']['num_classes']
    
    # Initialize dictionary to store samples per class
    # class_name -> list of samples
    class_samples = {class_name: [] for class_name in class_names}
    
    # Track how many samples we have for each class index
    samples_count = {i: 0 for i in range(num_classes)}
    
    print(f"\nCollecting {num_samples_per_class} samples per class from dataloader...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Unpack batch
            if len(batch_data) == 3:
                data, labels, idx = batch_data
                has_idx = True
            else:
                data, labels = batch_data[0], batch_data[1]
                has_idx = False
            
            # Apply augmentation if provided (for frequency transformation)
            # breakpoint()
            
            # Handle one-hot labels
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                label_indices = torch.argmax(labels, dim=1)
            else:
                label_indices = labels
            
            # Store samples by class
            batch_size = label_indices.size(0)
            for i in range(batch_size):
                class_idx = label_indices[i].item()
                class_name = class_names[class_idx]
                
                # Skip if we already have enough samples for this class
                if samples_count[class_idx] >= num_samples_per_class:
                    continue
                
                # Extract single sample from batch
                if isinstance(data, dict):
                    # Multi-modal data
                    sample_data = {}
                    for loc in data:
                        sample_data[loc] = {}
                        for mod in data[loc]:
                            sample_data[loc][mod] = data[loc][mod][i:i+1]  # Keep batch dimension
                else:
                    # Single tensor data
                    sample_data = data[i:i+1]  # Keep batch dimension
                
                # Create sample dictionary
                sample = {
                    'data': sample_data,
                    'label': class_idx,
                    'class_name': class_name
                }
                
                # check if this idx is necessary
                # breakpoint()
                if has_idx:
                    sample['idx'] = idx[i].item()
                
                class_samples[class_name].append(sample)
                samples_count[class_idx] += 1
            
            # Check if we have enough samples for all classes
            if all(count >= num_samples_per_class for count in samples_count.values()):
                print(f"  Collected {num_samples_per_class} samples for all classes after {batch_idx + 1} batches")
                break
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches - Status: {dict(zip(class_names, [samples_count[i] for i in range(num_classes)]))}")
    
    # Report final counts
    print("\nFinal samples collected per class:")
    for class_idx in range(num_classes):
        class_name = class_names[class_idx]
        count = len(class_samples[class_name])
        print(f"  {class_name}: {count} samples")
        if count < num_samples_per_class:
            print(f"    WARNING: Only {count} samples available (requested {num_samples_per_class})")
    
    return class_samples
     




def get_class_wise_time_domain_sampled(dataloader, config, device):
    """
    Get class-wise time-domain sampled data from the dataloader.
    """
    num_samples_per_class = 10

    if 'Parkland' in config['yaml_path']:
        samples_time_domain = get_parklad_samples_per_class(dataloader, config, device, num_samples_per_class)
    else:
        raise ValueError(f"Dataset {config['yaml_path']} not supported")
    return samples_time_domain


def main():
    """Main hard noise analysis function."""
    
    # ========================================================================
    # 1. Load Configuration
    # ========================================================================
    print("=" * 80)
    print("HARD NOISE ANALYSIS")
    print("=" * 80)
    
    config = get_config()
    print("Configuration loaded successfully")
    print(f"  Dataset: {config.get('yaml_path', 'Unknown')}")
    print(f"  Device: {config.get('device', 'cpu')}")
    
    # ========================================================================
    # 2. Create Dataloaders
    # ========================================================================
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    print("  Train batches: {}".format(len(train_loader)))
    print("  Val batches: {}".format(len(val_loader)))
    print("  Test batches: {}".format(len(test_loader)))
    
    # ========================================================================
    # 3. Create Augmenter
    # ========================================================================
    print("\nCreating augmenter...")
    
    # For training, we typically want augmentation enabled

    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')

    samples_time_domain = get_class_wise_time_domain_sampled(test_loader, config, device)
    
    # ========================================================================
    # 4. Save Samples
    # ========================================================================
    breakpoint()
    print("\n" + "=" * 80)
    print("Saving samples...")
    output_path = Path(__file__).parent / "class_samples.pt"
    torch.save(samples_time_domain, output_path)
    print(f"Samples saved to: {output_path}")
    print("=" * 80)
    
    return samples_time_domain




if __name__ == "__main__":
    main()