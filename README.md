# Model Distillation Training Pipeline - Use this for Demo Purposes

A flexible framework for training compressed neural networks using distillation and early exit techniques for time-series sensor data.

## Table of Contents
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Configuration Guide](#configuration-guide)
  - [Dataset Configuration](#1-dataset-configuration)
  - [Model Zoo](#2-model-zoo-defining-models)
  - [Experiments](#3-distillation-experiments)
- [Running Training](#running-training)
- [Monitoring with TensorBoard](#monitoring-with-tensorboard)

---

## Quick Start


### 1. Create Conda Environment

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate cenv
```


**Run the demo experiment** (to verify setup):

```bash
# Activate the environment
conda activate cenv

# Run a quick 2-epoch test with early exit model
cd <your_path>/baseline/src2/train_test
python train.py -experiment_name only_audio_resnet_early_exit -yaml_path ../data/ACIDS.yaml -gpu 0
```

This will train a small ResNet with early exits for 2 epochs. Check the output in `src2/experiments/`.

---

## Setup

### 1. Create Conda Environment

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate cenv
```

**Note:** The `environment.yml` file contains all necessary dependencies including PyTorch, TensorBoard, and other required packages.

---

## Configuration Guide

All configuration is done in YAML files (e.g., `src2/data/ACIDS.yaml`). The config has three main sections:

### 1. Dataset Configuration

Update the dataset paths to point to your data:

```yaml
# Top of ACIDS.yaml
vehicle_classification:
    num_classes: 10 
    class_names: ["background", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    train_index_file: /path/to/your/data/train_index.txt 
    val_index_file: /path/to/your/data/val_index.txt  
    test_index_file: /path/to/your/data/test_index.txt 
    train_class_count_file: /path/to/your/data/train_class_count.txt

# Base directory for experiment outputs
base_experiment_dir: /home/misra8/baseline/src2/experiments

# Batch size and data loading
batch_size: 32
num_workers: 4
```

**Key parameters to update:**
- `train_index_file`, `val_index_file`, `test_index_file`: Paths to your dataset splits
- `base_experiment_dir`: Where checkpoints and logs will be saved
- `batch_size`: Adjust based on your GPU memory

---

### 2. Model Zoo: Defining Models

Define reusable models once in the `models` section. Each model can be used as a teacher or student.

#### Basic Model Structure

```yaml
models:
    my_model_name:
        model_source: "create_single_modal_model"  # Factory function
        model_type: "resnet"  # or "convonly"
        active_modality: "audio"  # or "seismic"
        layers: [2, 2, 2, 2]  # Architecture depth
        filter_sizes: [64, 128, 256, 512]  # Channels per stage
        stem_kernel: 7
        stem_stride: 2
        use_maxpool: true
        dropout_ratio: 0.2
        fc_dim: 512
        early_exits: []  # No early exits (see below for early exits)
```

#### Model Types

**ResNet Model:**
```yaml
teacher_audio_resnet18:
    model_source: "create_single_modal_model"
    model_type: "resnet"
    active_modality: "audio"
    layers: [2, 2, 2, 2]  # Standard ResNet18
    filter_sizes: [64, 128, 256, 512]
    stem_kernel: 7
    stem_stride: 2
    use_maxpool: true
    dropout_ratio: 0.2
    fc_dim: 512
    early_exits: []
```

**Compressed ResNet Model (smaller):**
```yaml
student_audio_resnet:
    model_source: "create_single_modal_model"
    model_type: "resnet"
    active_modality: "audio"
    layers: [1, 1, 1, 1]  # Fewer layers
    filter_sizes: [16, 32, 48, 96]  # Smaller filters
    stem_kernel: 3
    stem_stride: 1
    use_maxpool: false
    dropout_ratio: 0.1
    fc_dim: 64
    early_exits: [1, 2]  # Add early exits at stages 1 and 2
```

**ConvOnly Model (pure convolutional):**
```yaml
student_audio_convonly:
    model_source: "create_single_modal_model"
    model_type: "convonly"
    active_modality: "audio"
    num_blocks: [2, 2, 2, 2]  # Blocks per stage
    filter_sizes: [16, 32, 48, 96]
    kernel_sizes: [3, 3, 3, 3]  # Kernel size per stage
    strides: [1, 2, 2, 2]  # Stride per stage
    stem_kernel: 3
    stem_stride: 1
    dropout_ratio: 0.1
    fc_dim: 64
    early_exits: [1, 2]
    early_exit_type: "gap_linear"  # Global average pooling + linear
```

#### Adding Early Exits

Early exits allow the model to make predictions at intermediate layers, enabling adaptive inference.

**Key parameters:**
- `early_exits: [1, 2]`: Add exits after stage 1 and stage 2
- `early_exits: []`: No early exits (standard model)

**How it works:**
- For ResNet: Each exit is `GlobalAvgPool → Linear(num_classes)`
- For ConvOnly: Each exit is `1x1 Conv → GlobalAvgPool` or `GlobalAvgPool → Linear`

**Example - Model with 3 exits:**
```yaml
my_early_exit_model:
    model_type: "resnet"
    layers: [1, 1, 1, 1]  # 4 stages
    early_exits: [0, 2, 3]  # Exits after stage 0, 2, and 3 + final exit
```

This creates 4 total outputs: 3 early exits + 1 final exit.

---

### 3. Distillation Experiments

Define training experiments that use models from the zoo.

#### Basic Experiment Structure

```yaml
distillation:
    enabled: true
    
    my_experiment_name:
        models: ["model1", "model2", "model3"]  # From model zoo
        stages:
            - train_type: "vanilla_supervised"
              teacher_idx: 0  # Not used for vanilla training
              epochs: 50
              loss_name: "cross_entropy"
              exit_weights: [0.3, 0.3, 0.4]  # Optional
        
        optimizer:
            name: "AdamW"
            start_lr: 0.0001
            warmup_lr: 0.000001
            min_lr: 0.000001
            clip_grad: 5.0
            weight_decay: 0.05
        
        lr_scheduler:
            name: "cosine"
            warmup_prefix: True
            warmup_epochs: 0
            start_epoch: 0
            decay_epochs: 50
            decay_rate: 0.2
```

#### Training Types

**1. Vanilla Supervised Training** (no distillation):

```yaml
only_audio_resnet18:
    models: ["student_audio_resnet"]  # Single model
    stages:
        - train_type: "vanilla_supervised"
          teacher_idx: 0
          epochs: 50
          loss_name: "cross_entropy"
    optimizer: {...}
    lr_scheduler: {...}
```

**2. Vanilla Supervised with Early Exits:**

```yaml
only_audio_resnet_early_exit:
    models: ["student_audio_resnet"]  # Model with early_exits: [1, 2]
    stages:
        - train_type: "vanilla_supervised"
          teacher_idx: 0
          epochs: 50
          loss_name: "cross_entropy"
          exit_weights: [0.3, 0.3, 0.4]  # Weight for exit1, exit2, final
    optimizer: {...}
    lr_scheduler: {...}
```

**How `exit_weights` works:**
- If your model has `early_exits: [1, 2]`, you have 3 total exits: exit1, exit2, final
- `exit_weights: [0.3, 0.3, 0.4]` means:
  - 30% weight on exit1 loss
  - 30% weight on exit2 loss
  - 40% weight on final exit loss
- If you **omit** `exit_weights`, the system uses **equal weights** for all exits (e.g., `[0.333, 0.333, 0.333]`)

**Example without exit_weights (equal weighting):**
```yaml
stages:
    - train_type: "vanilla_supervised"
      epochs: 50
      loss_name: "cross_entropy"
      # No exit_weights specified → equal weights automatically
```

**3. Multi-Stage Distillation** (future feature):

```yaml
audio_cascade_2stage:
    models: ["teacher_audio_resnet18", "student_audio_resnet", "student_audio_convonly"]
    stages:
        # Stage 0: teacher_resnet18 → student_audio_resnet
        - name: "resnet18_to_audio_resnet"
          teacher_idx: 0
          student_idx: 1
          epochs: 50
          loss_name: "kd_loss"  # Knowledge distillation (not yet implemented)
          temperature: 4.0
          alpha: 0.5
        
        # Stage 1: student_audio_resnet → student_audio_convonly
        - name: "audio_resnet_to_convonly"
          teacher_idx: 1
          student_idx: 2
          epochs: 100
          loss_name: "kd_loss"
          temperature: 3.0
          alpha: 0.7
    optimizer: {...}
    lr_scheduler: {...}
```

---

## Running Training

### Command Format

```bash
cd src2/train_test
python train.py -experiment_name <NAME> -yaml_path <PATH> -gpu <GPU_ID>
```

### Examples

**1. Run the demo experiment (quick test):**

```bash
python train.py -experiment_name only_audio_resnet_early_exit -yaml_path ../data/ACIDS.yaml -gpu 0
```
- Trains `student_audio_resnet` with early exits
- 2 epochs (configured for quick testing)
- Uses `exit_weights: [0.3, 0.3, 0.4]`

**2. Train a model without early exits:**

```bash
python train.py -experiment_name only_audio_resnet18 -yaml_path ../data/ACIDS.yaml -gpu 0
```
- Trains `student_audio_resnet` without early exits
- 50 epochs (configured in YAML)

**3. Train ConvOnly model with early exits:**

```bash
python train.py -experiment_name only_audio_convonly_early_exit -yaml_path ../data/ACIDS.yaml -gpu 0
```

**4. Use CPU instead of GPU:**

```bash
python train.py -experiment_name only_audio_resnet_early_exit -yaml_path ../data/ACIDS.yaml -gpu -1
```

### Output Structure

Training outputs are saved to `base_experiment_dir` (configured in YAML):

```
experiments/
└── YYYYMMDD_HHMMSS_experiment_name/
    ├── checkpoints/
    │   ├── best_model.pth
    │   └── last_model.pth
    ├── logs/
    │   ├── train.log
    │   └── train_early_exits.log  # For early exit models
    └── tensorboard/
        └── events.out.tfevents...
```

---

## Testing Trained Models

Test your trained models to evaluate performance on the test set.

### Command Format

```bash
cd src2/train_test
python test.py --experiment_dir <PATH_TO_EXPERIMENT> --gpu <GPU_ID>
```

### Examples

**Test with best checkpoint (default):**

```bash
python test.py --experiment_dir ../experiments/20260214_173826_only_audio_resnet_early_exit --gpu 0
```

This will:
- Load config from the experiment directory
- Use `models/best_model.pth` by default
- Test on the test set
- Save results to `experiment_dir/test_{timestamp}/`

**Test with last epoch checkpoint instead of best:**

```bash
python test.py --experiment_dir ../experiments/20260214_173826_only_audio_resnet_early_exit \
               --checkpoint_path ../experiments/20260214_173826_only_audio_resnet_early_exit/models/last_model.pth \
               --gpu 0
```

**Test on CPU:**

```bash
python test.py --experiment_dir ../experiments/20260214_173826_only_audio_resnet_early_exit --gpu -1
```

### Test Output Structure

Testing creates a timestamped directory inside your experiment:

```
experiments/YYYYMMDD_HHMMSS_experiment_name/
    └── test_YYYYMMDD_HHMMSS/
        ├── logs/
        │   └── test.log
        ├── tensorboard/  # If model has early exits
        │   └── events.out.tfevents...
        └── test_results.txt
```

### Viewing Test Results

**Console output shows:**
- Per-exit performance (if early exit model)
- Final test accuracy and loss

**test_results.txt contains:**
- Experiment name and model details
- Checkpoint path used
- Test accuracy and loss for each exit (if applicable)

**TensorBoard (for early exit models):**
```bash
tensorboard --logdir=experiments/YYYYMMDD_HHMMSS_experiment_name/test_YYYYMMDD_HHMMSS/tensorboard
```

---

## Monitoring with TensorBoard

### Start TensorBoard

```bash
tensorboard --logdir=src2/experiments --port=6006
```

Then open your browser to: `http://localhost:6006`

### What You'll See

**For standard models:**
- Training loss and accuracy (per epoch)
- Validation loss and accuracy (per epoch)
- Confusion matrix (validation set)
- Learning rate schedule

**For early exit models (additional):**
- Loss and accuracy **per exit** (exit1, exit2, final)
- Confusion matrix **per exit**
- Exit-specific performance trends

### Remote Server Access

If running on a remote server, forward the port:

```bash
# On your local machine
ssh -L 6006:localhost:6006 user@remote_server

# Then access http://localhost:6006 in your browser
```

---

## Creating Your Own Models and Experiments

### Step 1: Define a New Model

Add to the `models` section in your YAML:

```yaml
models:
    my_tiny_model:
        model_source: "create_single_modal_model"
        model_type: "resnet"
        active_modality: "audio"
        layers: [1, 1, 1, 1]
        filter_sizes: [8, 16, 32, 64]  # Very small
        stem_kernel: 3
        stem_stride: 1
        use_maxpool: false
        dropout_ratio: 0.1
        fc_dim: 32
        early_exits: [2]  # One early exit at stage 2
```

### Step 2: Create an Experiment

Add to the `distillation` section:

```yaml
distillation:
    my_tiny_experiment:
        models: ["my_tiny_model"]
        stages:
            - train_type: "vanilla_supervised"
              teacher_idx: 0
              epochs: 50
              loss_name: "cross_entropy"
              exit_weights: [0.5, 0.5]  # Equal weight for early + final
        
        optimizer:
            name: "AdamW"
            start_lr: 0.0001
            weight_decay: 0.05
        
        lr_scheduler:
            name: "cosine"
            decay_epochs: 50
```

### Step 3: Run Training

```bash
python train.py -experiment_name my_tiny_experiment -yaml_path ../data/ACIDS.yaml -gpu 0
```

---

## Troubleshooting

### Issue: "Permission denied" when saving checkpoints

**Solution:** Update `base_experiment_dir` in your YAML to a writable location:
```yaml
base_experiment_dir: /home/YOUR_USERNAME/baseline/src2/experiments
```

### Issue: "KeyError: 'model_name'"

**Solution:** Ensure the model name in your experiment's `models` list matches a model defined in the `models` section.

### Issue: "Expected N weights, got M"

**Solution:** Your `exit_weights` list length must match the number of total exits. For `early_exits: [1, 2]`, you need 3 weights (2 early + 1 final).

### Issue: Out of memory (CUDA OOM)

**Solution:** Reduce `batch_size` in your YAML or use a smaller model.

---

## Additional Resources

- **Model implementations:** `src2/models/ResNet.py`, `src2/models/ConvOnlyModels.py`
- **Training utilities:** `src2/train_test/train_test_utils.py`
- **Loss functions:** `src2/train_test/loss.py`
- **Config examples:** `src2/data/ACIDS.yaml`

---

## Citation

If you use this code, please cite:
```
[Your paper citation here]
```
