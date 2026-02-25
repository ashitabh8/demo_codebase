# Training and Testing Infrastructure

This directory contains a comprehensive training and testing infrastructure for multi-modal neural networks.

## File Structure

```
train_test/
├── main.py                  # Quick testing/debugging script
├── train.py                 # Training orchestration script
├── test.py                  # Testing script with checkpoint loading
├── loss.py                  # Loss functions module
└── train_test_utils.py      # Core training/testing utilities
```

## Features

- **Experiment Tracking**: Automatic experiment ID generation with timestamps
- **Comprehensive Logging**: 
  - Text files for training logs
  - TensorBoard for real-time visualization
  - Confusion matrices logged as images
- **Checkpointing**: 
  - Best model (based on validation accuracy)
  - Last epoch model
- **Flexible Architecture**:
  - Custom validation functions
  - Custom test functions
  - Extensible loss functions
- **Metrics**:
  - Loss (train/val/test)
  - Accuracy (overall and per-class)
  - Confusion matrices

## Quick Start

### Training

```bash
cd /home/misra8/sensing-nn/src2/train_test

# Basic training
python train.py --model ResNet --model_variant resnet18 \
    --yaml_path ../data/Parkland.yaml --gpu 0

# Training will create an experiment directory like:
# experiments/20231118_143052_ResNet_resnet18/
```

### Testing

```bash
# Test with specific checkpoint
python test.py --checkpoint_path ../experiments/20231118_143052_ResNet_resnet18/models/best_model.pth

# Auto-find and test latest experiment (using best model)
python test.py --auto_latest --use_best --gpu 0

# Auto-find and test latest experiment (using last epoch)
python test.py --auto_latest --gpu 0

# Test without loading checkpoint (fresh model)
python test.py --model ResNet --model_variant resnet18 \
    --yaml_path ../data/Parkland.yaml --gpu 0
```

### Debugging

```bash
# Quick model forward pass test
python main.py --model ResNet --model_variant resnet18 \
    --yaml_path ../data/Parkland.yaml --gpu 0
```

## Experiment Directory Structure

After training, experiments are saved in:

```
experiments/
└── YYYYMMDD_HHMMSS_modelname_variant/
    ├── config.yaml              # Copy of configuration
    ├── models/
    │   ├── best_model.pth      # Best validation accuracy
    │   └── last_epoch.pth      # Last training epoch
    ├── logs/
    │   ├── train.log           # Training logs
    │   ├── test_results.txt    # Test results
    │   └── confusion_matrix.png # Test confusion matrix
    └── tensorboard/
        └── events.out.tfevents.* # TensorBoard logs
```

## Viewing Training Progress

### TensorBoard

```bash
# View real-time training progress
tensorboard --logdir=../experiments/YYYYMMDD_HHMMSS_modelname_variant/tensorboard

# View all experiments
tensorboard --logdir=../experiments/
```

Open browser to `http://localhost:6006` to view:
- Training/validation loss curves
- Training/validation accuracy
- Learning rate schedule
- Confusion matrices (updated every 5 epochs)

### Text Logs

```bash
# View training logs
tail -f ../experiments/YYYYMMDD_HHMMSS_modelname_variant/logs/train.log

# View test results
cat ../experiments/YYYYMMDD_HHMMSS_modelname_variant/logs/test_results.txt
```

## Loss Functions

The `loss.py` module provides a flexible interface for different loss functions:

```python
from train_test.loss import get_loss_function

# Standard cross-entropy
loss_fn = get_loss_function("cross_entropy")

# Cross-entropy with label smoothing
loss_fn = get_loss_function("label_smoothing_ce", label_smoothing=0.1)

# Easy to extend for custom losses
```

## Custom Validation/Test Functions

### Default Validation Function

The infrastructure includes a built-in `validate()` function that handles standard validation:

```python
from train_test.train_test_utils import validate

# Use the default validation function directly
val_results = validate(model, val_loader, loss_fn, device)
print(f"Validation accuracy: {val_results['accuracy']:.4f}")
```

### Custom Validation

You can provide custom validation functions that follow the same signature:

```python
def my_custom_validation(model, val_loader, loss_fn, device):
    """
    Custom validation function.
    
    Must return dict with keys:
        - 'loss': float
        - 'accuracy': float
        - 'predictions': list
        - 'labels': list
    """
    # Your custom validation logic here
    model.eval()
    # ... compute metrics ...
    return {
        'loss': val_loss,
        'accuracy': val_acc,
        'predictions': all_preds,
        'labels': all_labels
    }

# Use in training
from train_test.train_test_utils import train

train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    experiment_dir=experiment_dir,
    val_fn=my_custom_validation  # Pass custom function
)
```

## Checkpoint Format

Saved checkpoints contain:

```python
{
    'epoch': int,                    # Epoch number
    'model_state_dict': OrderedDict, # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'val_acc': float,                # Validation accuracy
    'val_loss': float,               # Validation loss
    'config': dict                   # Full configuration
}
```

## Loading Checkpoints

```python
import torch
from models.create_models import create_model

# Load checkpoint
checkpoint = torch.load('path/to/checkpoint.pth')

# Create model from saved config
model = create_model(checkpoint['config'])

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])

# Check performance
print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
```

## Configuration

Training parameters are specified in the YAML config file under the model section:

```yaml
ResNet:
  # Training
  optimizer:
    name: "AdamW"
    start_lr: 0.0001
    weight_decay: 0.05
    clip_grad: 5.0
  
  lr_scheduler:
    name: "cosine"
    train_epochs: 50
    warmup_epochs: 0
    min_lr: 0.000001
  
  # Model architecture
  dropout_ratio: 0.2
  fc_dim: 256
  
  # Augmentation
  fixed_augmenters:
    time_augmenters: ["no"]
    freq_augmenters: ["no"]
```

## Advanced Usage

### Custom Loss Function

Add to `loss.py`:

```python
def get_loss_function(loss_name="cross_entropy", **kwargs):
    # ... existing code ...
    
    elif loss_name == "focal":
        alpha = kwargs.get("alpha", 0.25)
        gamma = kwargs.get("gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
```

### Custom Metrics

Add to `train_test_utils.py`:

```python
def calculate_f1_score(predictions, labels, num_classes):
    """Calculate per-class F1 scores."""
    from sklearn.metrics import f1_score
    return f1_score(labels, predictions, average=None, labels=range(num_classes))
```

### Resume Training

```python
# Load checkpoint
checkpoint = torch.load('path/to/last_epoch.pth')

# Create model and optimizer
model = create_model(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = setup_optimizer(model, config)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Continue training
start_epoch = checkpoint['epoch'] + 1
```

## Tips and Best Practices

1. **Monitor Training**: Always use TensorBoard to monitor training in real-time
2. **Save Checkpoints**: Best model is automatically saved when validation accuracy improves
3. **Experiment Naming**: Experiment IDs include timestamps, so no name conflicts
4. **GPU Memory**: Adjust batch size in config if you encounter OOM errors
5. **Data Augmentation**: Configure in model-specific section of YAML config
6. **Learning Rate**: Use warmup for large batch sizes
7. **Early Stopping**: Monitor validation metrics and stop if performance plateaus

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size in config
batch_size: 64  # Try 32 or 16
```

### Slow Training

```bash
# Increase number of workers
num_workers: 8  # Match your CPU cores
```

### NaN Loss

- Check learning rate (try lower value)
- Check data normalization
- Enable gradient clipping (already configured)
- Check for invalid values in data

### Import Errors

```bash
# Ensure you're running from the correct directory
cd /home/misra8/sensing-nn/src2/train_test

# Or use absolute paths
export PYTHONPATH=/home/misra8/sensing-nn/src2:$PYTHONPATH
```

## Dependencies

All dependencies are listed in `/home/misra8/sensing-nn/requirements.txt`:

- PyTorch (torch, torchvision)
- TensorBoard
- scikit-learn (confusion matrix)
- matplotlib, seaborn (plotting)
- PyYAML (config loading)
- numpy, pandas

Install with:
```bash
pip install -r /home/misra8/sensing-nn/requirements.txt
```

## Contact

For issues or questions about the training infrastructure, refer to this README or check the inline documentation in the source files.

