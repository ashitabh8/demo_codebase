"""
Training and Testing Utilities

This module provides core training/testing functionality with:
- Experiment tracking and directory management
- Training loop with checkpointing and logging
- Testing function with flexible evaluation
- Metrics calculation (accuracy, confusion matrix)
- TensorBoard and text file logging
"""

import os
import copy
import logging
import yaml
import shutil
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Logging helpers (input shapes + peak RAM)
# ----------------------------------------------------------------------------

def _format_tensor_shape(t: torch.Tensor) -> str:
    return "x".join(str(d) for d in t.shape)


def _log_single_modality_input_shape(model, data, logger, *, epoch: int, batch_idx: int) -> None:
    """
    Log the input tensor shape that is actually used by a single-modality model.

    Our single-modal model wrappers index the dict input as:
      freq_x[model.location_name][model.modality_name] = tensor [B, C, H, W]
    The dataloader may still include other modalities; we log:
      - the tensor used by the model (preferred)
      - any other modalities present for the same location (if present)
    """
    if logger is None:
        return

    if not isinstance(data, dict):
        if isinstance(data, torch.Tensor):
            logger.info(
                f"Input batch shape (epoch={epoch}, batch={batch_idx}): "
                f"{tuple(data.shape)} dtype={data.dtype} device={data.device}"
            )
        else:
            logger.info(
                f"Input batch type (epoch={epoch}, batch={batch_idx}): {type(data)}"
            )
        return

    loc = getattr(model, "location_name", None)
    mod = getattr(model, "modality_name", None)

    # Preferred: log the shape used by the wrapper's forward().
    if loc is not None and mod is not None and loc in data and mod in data[loc]:
        t = data[loc][mod]
        if isinstance(t, torch.Tensor):
            # Reuse the existing input-memory estimator for consistent accounting.
            try:
                from models.create_models import get_input_memory

                input_mem_info = get_input_memory(data, unit="MB")
                used_mem_mb = None
                for info in input_mem_info.get("shape_info", []):
                    if info.get("location") == loc and info.get("modality") == mod:
                        used_mem_mb = info.get("memory")
                        break

                total_input_mb = input_mem_info.get("total_memory", None)
            except Exception:
                used_mem_mb = None
                total_input_mb = None

            mem_str = ""
            if used_mem_mb is not None:
                mem_str = f" estimated_input_mem_per_sample={used_mem_mb:.4f} MB"
            if total_input_mb is not None:
                mem_str += f" total_input_mem_per_sample={total_input_mb:.4f} MB"

            logger.info(
                f"Input tensor shape used by model (epoch={epoch}, batch={batch_idx}): "
                f"{loc}/{mod} -> ({_format_tensor_shape(t)}) dtype={t.dtype} device={t.device}{mem_str}"
            )
            other_modalities = [k for k in data[loc].keys() if k != mod]
            if other_modalities:
                logger.info(
                    f"  Note: other modalities present for {loc} (model uses {mod}): {other_modalities}"
                )
            return

    # Fallback: dump shapes for all tensors in the dict.
    parts = []
    for loc_k in data:
        for mod_k in data[loc_k]:
            t = data[loc_k][mod_k]
            if isinstance(t, torch.Tensor):
                parts.append(f"{loc_k}/{mod_k}=({_format_tensor_shape(t)})")
            else:
                parts.append(f"{loc_k}/{mod_k}=<non-tensor {type(t)}>")
    logger.info(
        f"Input dict tensor shapes (epoch={epoch}, batch={batch_idx}): " + ", ".join(parts)
    )


def _get_process_peak_rss_kb() -> int:
    """
    Return peak resident-set-size (RSS) in KB for the current process.

    On Linux, `ru_maxrss` is in KB.
    """
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

#check this line
# ============================================================================
# Optimizer and Scheduler Setup
# ============================================================================

def setup_optimizer(model, config, training_config=None):
    """
    Create optimizer based on configuration.

    Args:
        model: PyTorch model
        config: Configuration dictionary (full config)
        training_config: Training config dict with optimizer settings

    Returns:
        optimizer: Configured optimizer
    """
    optimizer_config = training_config['optimizer']
    
    optimizer_name = optimizer_config['name']
    start_lr = optimizer_config['start_lr']
    weight_decay = optimizer_config['weight_decay']
    
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=start_lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=start_lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        momentum = optimizer_config.get("momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=start_lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    logging.info(f"Optimizer created: {optimizer_name}")
    logging.info(f"  Learning rate: {start_lr}")
    logging.info(f"  Weight decay: {weight_decay}")
    
    return optimizer


def setup_scheduler(optimizer, config, training_config):
    """
    Create learning rate scheduler based on configuration.

    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary (full config)
        training_config: Training config dict with lr_scheduler and epochs

    Returns:
        scheduler: Learning rate scheduler (or None)
    """
    scheduler_config = training_config['lr_scheduler']
    scheduler_name = scheduler_config['name']
    train_epochs = training_config['epochs']
    warmup_epochs = scheduler_config['warmup_epochs']
    
    if scheduler_name == "cosine":
        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_epochs - warmup_epochs,
            eta_min=scheduler_config.get("min_lr", 1e-6)
        )
        logging.info(f"Scheduler created: CosineAnnealingLR")
        logging.info(f"  Train epochs: {train_epochs}, Warmup epochs: {warmup_epochs}, Min LR: {scheduler_config.get('min_lr', 1e-6)}")
    
    elif scheduler_name == "step":
        # Step decay
        decay_epochs = scheduler_config.get("decay_epochs", 30)
        decay_rate = scheduler_config.get("decay_rate", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=decay_epochs,
            gamma=decay_rate
        )
        logging.info(f"Scheduler created: StepLR")
        logging.info(f"  Step size: {decay_epochs}, Gamma: {decay_rate}")
    
    elif scheduler_name == "multistep":
        # Multi-step decay
        milestones = scheduler_config.get("milestones", [30, 60, 90])
        decay_rate = scheduler_config.get("decay_rate", 0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=decay_rate
        )
        logging.info(f"Scheduler created: MultiStepLR")
        logging.info(f"  Milestones: {milestones}, Gamma: {decay_rate}")
    
    elif scheduler_name == "none" or scheduler_name is None:
        scheduler = None
        logging.info("No learning rate scheduler")
    
    else:
        logging.warning(f"Unknown scheduler: {scheduler_name}. Using no scheduler.")
        scheduler = None
    
    return scheduler


# ============================================================================
# Config Utilities
# ============================================================================

def validate_and_resolve_training_config(config):
    """
    Validate top-level experiment/training config and log key selections.

    Args:
        config: Full config dict

    Returns:
        tuple:
            experiment_name,
            experiment_config,
            model_name,
            training_config_name,
            training_config,
            train_type,
            stage_epochs,
            loss_name
    """
    experiment_name = config.get("experiment_name")
    if experiment_name is None:
        raise ValueError(
            "experiment_name not found in config. Please provide --experiment_name argument."
        )

    logging.info(f"  Experiment: {experiment_name}")
    logging.info(f"  Dataset: {config.get('yaml_path', 'Unknown')}")
    logging.info(f"  Device: {config.get('device', 'cpu')}")

    if "experiments" not in config or not config["experiments"].get("enabled", False):
        raise ValueError(
            "Experiments not enabled in config. Set experiments.enabled: true"
        )

    available = [k for k in config["experiments"] if k != "enabled"]
    if experiment_name not in config["experiments"]:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. "
            f"Available experiments: {available}"
        )

    experiment_config = config["experiments"][experiment_name]
    model_name = experiment_config["model"]
    training_config_name = experiment_config["training"]

    if training_config_name not in config["training_configs"]:
        raise ValueError(
            f"Training config '{training_config_name}' not found in training_configs"
        )
    training_config = config["training_configs"][training_config_name]

    train_type = training_config["type"]
    stage_epochs = training_config["epochs"]
    loss_name = training_config["loss_name"]

    logging.info(f"  Student model: {model_name}")
    logging.info(f"  Training config: {training_config_name}")
    logging.info(f"  Training type: {train_type}")
    logging.info(f"  Loss: {loss_name}")

    return (
        experiment_name,
        experiment_config,
        model_name,
        training_config_name,
        training_config,
        train_type,
        stage_epochs,
        loss_name,
    )


def apply_class_subset(config):
    """
    If include_classes is set in config, restrict vehicle_classification to
    that subset and update config in-place with the remapped class info.

    Args:
        config: Full config dict (mutated in place)
    """
    include = config.get('include_classes')
    if not include:
        return

    include = sorted(set(include))
    task_cfg = config['vehicle_classification']

    num_classes_orig = task_cfg['num_classes']
    invalid = [c for c in include if c < 0 or c >= num_classes_orig]
    if invalid:
        raise ValueError(
            f"include_classes contains invalid indices {invalid} "
            f"for num_classes={num_classes_orig}"
        )

    old_to_new = {old: new for new, old in enumerate(include)}
    new_class_names = [task_cfg['class_names'][i] for i in include]

    task_cfg['num_classes'] = len(include)
    task_cfg['class_names'] = new_class_names
    config['include_classes'] = include
    config['include_classes_mapping'] = old_to_new

    logging.info(f"Using class subset: {include}")
    logging.info(f"Updated num_classes: {task_cfg['num_classes']}")
    logging.info(f"Updated class_names: {task_cfg['class_names']}")


# ============================================================================
# Experiment Management
# ============================================================================

def create_experiment_id(model_name, model_variant=None):
    """
    Generate a unique experiment ID with timestamp and model information.
    
    Format: YYYYMMDD_HHMMSS_modelname_variant
    
    Args:
        model_name: Name of the model (e.g., "resnet", "deepsense")
        model_variant: Optional model variant (e.g., "resnet18", "resnet50")
    
    Returns:
        experiment_id: String identifier for this experiment
    
    Example:
        >>> create_experiment_id("resnet", "resnet18")
        "20231118_143052_resnet_resnet18"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_variant:
        experiment_id = f"{timestamp}_{model_name}_{model_variant}"
    else:
        experiment_id = f"{timestamp}_{model_name}"
    
    return experiment_id


def setup_experiment_dir(config, experiment_name=None):
    """
    Create experiment directory structure and save configuration.
    
    Structure:
        experiments/
        └── <experiment_id>/
            ├── config.yaml
            ├── models/
            ├── logs/
            └── tensorboard/
    
    Args:
        config: Configuration dictionary
        experiment_name: Name of the experiment (for distillation), optional
    
    Returns:
        experiment_dir: Path to the created experiment directory
        tensorboard_dir: Path to tensorboard logs
    """
    # Create experiment ID
    base_experiments_dir = config.get("base_experiment_dir", "/home/misra8/sensing-nn/src2/experiments")
    
    if experiment_name is not None:
        # For distillation experiments, use experiment_name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{timestamp}_{experiment_name}"
    else:
        raise ValueError("experiment_name not found in config. Please provide --experiment_name argument.")
    
    # Create directory structure
    experiment_dir = Path(base_experiments_dir) / experiment_id
    models_dir = experiment_dir / "models"
    logs_dir = experiment_dir / "logs"
    tensorboard_dir = experiment_dir / "tensorboard"
    
    # Create directories
    experiment_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config_path = experiment_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logging.info(f"Experiment directory created: {experiment_dir}")
    logging.info(f"  Experiment ID: {experiment_id}")
    
    return str(experiment_dir), str(tensorboard_dir)


def setup_train_file_logging(experiment_dir, argv=None):
    """
    Attach a file logger for training and log the invoked command line.

    Args:
        experiment_dir: Path to experiment directory
        argv: Command-line args list (e.g., sys.argv)

    Returns:
        tuple:
            log_file: Path object to train log file
            file_handler: Attached logging handler
    """
    logs_dir = Path(experiment_dir) / "logs"
    log_file = logs_dir / "train.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logging.info(f"Logging to file: {log_file}")

    if argv is None:
        argv = []
    command_line = " ".join(argv)
    logging.info("")
    logging.info("Command line used to run this script:")
    logging.info(f"  {command_line}")

    return log_file, file_handler


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded checkpoint from: {checkpoint_path}")
    logging.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'val_acc' in checkpoint:
        logging.info(f"  Val Acc: {checkpoint['val_acc']:.4f}")
    return model


# ============================================================================
# Metrics Functions
# ============================================================================

def calculate_accuracy(outputs, labels):
    """
    Calculate classification accuracy.
    
    Args:
        outputs: Model outputs (logits) of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
    
    Returns:
        accuracy: Accuracy as a float between 0 and 1
    """
    predictions = torch.argmax(outputs, dim=1)
    
    # Handle one-hot encoded labels
    if len(labels.shape) == 2 and labels.shape[1] > 1:
        labels = torch.argmax(labels, dim=1)
    
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    
    return accuracy


def calculate_confusion_matrix(all_predictions, all_labels, num_classes):
    """
    Calculate confusion matrix.
    
    Args:
        all_predictions: Numpy array or list of predicted class indices
        all_labels: Numpy array or list of true class indices
        num_classes: Number of classes
    
    Returns:
        cm: Confusion matrix as numpy array of shape (num_classes, num_classes)
    """
    cm = sklearn_confusion_matrix(all_labels, all_predictions, labels=range(num_classes))
    return cm


def calculate_macro_f1_from_confusion_matrix(cm: np.ndarray) -> float:
    """
    Calculate macro-F1 from a confusion matrix.
    
    Macro-F1 = mean(F1_i) over classes i, where:
      F1_i = 2*TP_i / (2*TP_i + FP_i + FN_i)
    
    If a class has no support (TP=FP=FN=0), its F1 is defined as 0.
    
    Args:
        cm: Confusion matrix of shape (C, C)
    
    Returns:
        float: macro-F1 in [0, 1]
    """
    if cm is None:
        return 0.0
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got shape={cm.shape}")
    num_classes = cm.shape[0]
    if num_classes == 0:
        return 0.0

    f1s = []
    for i in range(num_classes):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        denom = 2.0 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)

    return float(np.mean(f1s)) if f1s else 0.0


def log_confusion_matrix_table(cm, class_names, logger, title="Validation Confusion Matrix"):
    """
    Log a confusion matrix as an ASCII table with per-class recall and precision.

    Example output:
        ============================================================
        Best Validation Confusion Matrix  (epoch 72  |  val_acc=0.7240)
        Predicted:   Polaris  Warhog   Truck  ...
        ----------------------------------------------------------------
          Polaris |      45       2       1  ...  | Recall: 0.900
           Warhog |       3      38       5  ...  | Recall: 0.808
        ...
        ----------------------------------------------------------------
        Precision   0.900   0.844   ...
        Macro-F1: 0.895
        ============================================================

    Args:
        cm:          Confusion matrix (num_classes × num_classes) numpy array, rows=actual, cols=predicted
        class_names: List of class name strings
        logger:      Logger instance
        title:       Title string logged above the table
    """
    import numpy as np
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    col_w = max(max(len(name) for name in class_names), 6) + 1
    label_w = max(max(len(name) for name in class_names), 8) + 2
    sep = "-" * (label_w + 3 + n * col_w + 2 + 12)

    logger.info("=" * len(sep))
    logger.info(title)
    logger.info(sep)

    # Header row
    header = " " * (label_w + 3)
    for name in class_names:
        header += name.rjust(col_w)
    header += "   Recall"
    logger.info(header)
    logger.info(sep)

    # Data rows (one per actual class)
    row_totals = cm.sum(axis=1)
    for i, name in enumerate(class_names):
        row_str = name.rjust(label_w) + " |"
        for j in range(n):
            row_str += str(int(cm[i, j])).rjust(col_w)
        recall = cm[i, i] / row_totals[i] if row_totals[i] > 0 else 0.0
        row_str += f"  | {recall:.3f}"
        logger.info(row_str)

    logger.info(sep)

    # Precision row
    col_totals = cm.sum(axis=0)
    prec_str = "Precision".rjust(label_w) + "  "
    for j in range(n):
        p = cm[j, j] / col_totals[j] if col_totals[j] > 0 else 0.0
        prec_str += f"{p:.3f}".rjust(col_w)
    logger.info(prec_str)

    # Macro-F1
    f1 = calculate_macro_f1_from_confusion_matrix(cm)
    logger.info(f"Macro-F1: {f1:.4f}")
    logger.info("=" * len(sep))


def plot_confusion_matrix(cm, class_names=None, normalize=False):
    """
    Create a matplotlib figure of the confusion matrix.
    
    Args:
        cm: Confusion matrix as numpy array
        class_names: List of class names (optional)
        normalize: Whether to normalize the confusion matrix
    
    Returns:
        fig: Matplotlib figure object
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap='Blues', ax=ax, cbar=True)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    
    if class_names:
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names, rotation=0)
    
    plt.tight_layout()
    return fig


# ============================================================================
# Validation Function
# ============================================================================

def validate(model, val_loader, loss_fn, device, augmenter=None, apply_augmentation_fn=None, num_classes=None):
    """
    Default validation function.
    
    Args:
        model: PyTorch model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run validation on
        augmenter: Data augmenter object (optional)
        apply_augmentation_fn: Function to apply augmentation (optional)
    
    Returns:
        val_results: Dictionary with validation metrics
            - 'loss': float
            - 'accuracy': float
            - 'predictions': list
            - 'labels': list
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_val_preds = []
    all_val_labels = []
    num_classes_from_outputs = None
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation", leave=False):
            # Unpack batch
            if len(batch_data) == 3:
                data, labels, idx = batch_data
            else:
                data, labels = batch_data[0], batch_data[1]
            
            # Apply augmentation if provided (for frequency transformation)
            if augmenter is not None and apply_augmentation_fn is not None:
                data, labels = apply_augmentation_fn(augmenter, data, labels)
            
            # Move to device
            labels = labels.to(device)
            if isinstance(data, dict):
                for loc in data:
                    for mod in data[loc]:
                        data[loc][mod] = data[loc][mod].to(device)
            else:
                data = data.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Extract logits for metrics (handle dict outputs)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            try:
                num_classes_from_outputs = int(logits.shape[1])
            except Exception:
                pass
            
            # Handle one-hot labels
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels
            
            loss = loss_fn(outputs, loss_labels)
            
            val_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(logits, dim=1)
            val_correct += (predictions == loss_labels).sum().item()
            val_total += labels.size(0)
            
            all_val_preds.extend(predictions.cpu().numpy())
            all_val_labels.extend(loss_labels.cpu().numpy())
    
    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_correct / val_total

    # Confusion-matrix-derived metrics
    inferred_num_classes = num_classes
    if inferred_num_classes is None:
        if num_classes_from_outputs is not None:
            inferred_num_classes = num_classes_from_outputs
        elif len(all_val_preds) > 0 and len(all_val_labels) > 0:
            inferred_num_classes = int(max(np.max(all_val_preds), np.max(all_val_labels)) + 1)

    cm = None
    f1_macro = None
    if inferred_num_classes is not None and inferred_num_classes > 0 and len(all_val_preds) > 0:
        cm = calculate_confusion_matrix(all_val_preds, all_val_labels, inferred_num_classes)
        f1_macro = calculate_macro_f1_from_confusion_matrix(cm)
    
    return {
        'loss': epoch_val_loss,
        'accuracy': epoch_val_acc,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'predictions': all_val_preds,
        'labels': all_val_labels
    }


def log_supcon_embedding_scatter(
    writer,
    *,
    features,
    labels,
    class_names,
    global_step,
    tag="SupCon/embeddings_val_pca2d",
    max_points=2048,
    random_state=0,
    logger=None,
):
    """
    PCA-reduce validation embeddings to 2D (or 1D if feature dim is 1) and log a scatter figure to TensorBoard.
    Embeddings are L2-normalized before projection (matching SupCon geometry).
    """
    log = logger or logging.getLogger("train_supcon")
    if features is None or labels is None:
        return False
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().float().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().long().numpy()
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    n = features.shape[0]
    if n == 0:
        log.warning("SupCon embedding scatter skipped: no samples.")
        return False
    d = features.shape[1]
    if d < 1:
        log.warning("SupCon embedding scatter skipped: empty feature dimension.")
        return False

    # Match contrastive geometry used in loss computation.
    features = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), a_min=1e-12, a_max=None)

    if n > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        features = features[idx]
        labels = labels[idx]
        n = max_points

    n_comp = min(2, d)
    coords = PCA(n_components=n_comp, random_state=random_state).fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for c in unique_labels:
        mask = labels == c
        name = class_names[int(c)] if class_names is not None and int(c) < len(class_names) else str(int(c))
        if n_comp == 2:
            ax.scatter(coords[mask, 0], coords[mask, 1], label=name, alpha=0.6, s=12)
        else:
            ax.scatter(coords[mask, 0], np.zeros(mask.sum()), label=name, alpha=0.6, s=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2" if n_comp == 2 else "(1D)")
    ax.set_title("Validation embeddings (clean val, normalized, PCA)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    writer.add_figure(tag, fig, global_step)
    plt.close(fig)
    return True


def log_supcon_embedding_tsne_scatter(
    writer,
    *,
    features,
    labels,
    class_names,
    global_step,
    tag="SupCon/embeddings_val_tsne2d",
    max_points=1000,
    random_state=0,
    logger=None,
):
    """
    t-SNE-reduce validation embeddings to 2D and log a scatter figure to TensorBoard.
    Embeddings are L2-normalized before projection.
    """
    log = logger or logging.getLogger("train_supcon")
    if features is None or labels is None:
        return False
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().float().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().long().numpy()
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    n = features.shape[0]
    if n < 3:
        log.warning("SupCon t-SNE scatter skipped: too few samples.")
        return False
    d = features.shape[1]
    if d < 1:
        log.warning("SupCon t-SNE scatter skipped: empty feature dimension.")
        return False

    features = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), a_min=1e-12, a_max=None)

    if n > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        features = features[idx]
        labels = labels[idx]
        n = max_points

    perplexity = min(30, max(5, n // 10))
    if perplexity >= n:
        perplexity = max(2, n - 1)
    coords = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for c in unique_labels:
        mask = labels == c
        name = class_names[int(c)] if class_names is not None and int(c) < len(class_names) else str(int(c))
        ax.scatter(coords[mask, 0], coords[mask, 1], label=name, alpha=0.6, s=12)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.set_title("Validation embeddings (clean val, normalized, t-SNE)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    writer.add_figure(tag, fig, global_step)
    plt.close(fig)
    return True


def validate_vanilla_supervised_contrastive(
    model,
    val_loader,
    loss_fn,
    device,
    augmenter=None,
    apply_augmentation_fn=None,
    num_classes=None,
    collect_embeddings=False,
    max_embedding_samples=2048,
):
    """
    Validation for `vanilla_supervised_contrastive`.

    Uses clean validation inputs only (augmenter mode "no" when available, so FFT
    and layout match training without stochastic augments). Loss is CE-only via
    single-output `loss_fn(outputs_clean, labels)`. Accuracy is from clean logits.

    When collect_embeddings is True, reuses the same forward for features
    (at most max_embedding_samples points, subsampled if needed).
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_val_preds = []
    all_val_labels = []
    num_classes_from_outputs = None
    emb_chunks = []
    emb_label_chunks = []
    embedding_collection_failed = False

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation", leave=False):
            # Unpack batch
            if len(batch_data) == 3:
                data, labels, idx = batch_data
            else:
                data, labels = batch_data[0], batch_data[1]

            if augmenter is not None and apply_augmentation_fn is not None:
                clean_data_src = copy.deepcopy(data)
                clean_labels_src = labels.clone() if isinstance(labels, torch.Tensor) else labels
                prev_mode = getattr(augmenter, "augmentation_mode", "no")
                augmenter.augmentation_mode = "no"
                try:
                    data_clean, labels_clean = apply_augmentation_fn(
                        augmenter, clean_data_src, clean_labels_src
                    )
                finally:
                    augmenter.augmentation_mode = prev_mode
            else:
                data_clean, labels_clean = data, labels

            labels_clean = labels_clean.to(device)
            if len(labels_clean.shape) == 2 and labels_clean.shape[1] > 1:
                loss_labels = torch.argmax(labels_clean, dim=1)
            else:
                loss_labels = labels_clean

            if isinstance(data_clean, dict):
                for loc in data_clean:
                    for mod in data_clean[loc]:
                        data_clean[loc][mod] = data_clean[loc][mod].to(device)
            else:
                data_clean = data_clean.to(device)

            outputs_clean = model(data_clean)
            logits = (
                outputs_clean["logits"]
                if isinstance(outputs_clean, dict)
                else outputs_clean
            )
            try:
                num_classes_from_outputs = int(logits.shape[1])
            except Exception:
                pass

            # Single-output path => CE only (see CrossEntropyPlusSupConLoss.forward)
            loss = loss_fn(outputs_clean, loss_labels)

            val_loss += loss.item() * loss_labels.size(0)
            predictions = torch.argmax(logits, dim=1)
            val_correct += (predictions == loss_labels).sum().item()
            val_total += loss_labels.size(0)

            all_val_preds.extend(predictions.cpu().numpy())
            all_val_labels.extend(loss_labels.cpu().numpy())

            if collect_embeddings and not embedding_collection_failed:
                if isinstance(outputs_clean, dict) and "features" in outputs_clean:
                    f_clean = outputs_clean["features"]
                    emb_chunks.append(f_clean.detach().cpu().float().numpy())
                    emb_label_chunks.append(loss_labels.detach().cpu().long().numpy())
                else:
                    embedding_collection_failed = True

    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_correct / val_total

    # Confusion-matrix-derived metrics
    inferred_num_classes = num_classes
    if inferred_num_classes is None:
        if num_classes_from_outputs is not None:
            inferred_num_classes = num_classes_from_outputs
        elif len(all_val_preds) > 0 and len(all_val_labels) > 0:
            inferred_num_classes = int(max(np.max(all_val_preds), np.max(all_val_labels)) + 1)

    cm = None
    f1_macro = None
    if inferred_num_classes is not None and inferred_num_classes > 0 and len(all_val_preds) > 0:
        cm = calculate_confusion_matrix(all_val_preds, all_val_labels, inferred_num_classes)
        f1_macro = calculate_macro_f1_from_confusion_matrix(cm)

    embedding_features = None
    embedding_labels = None
    if collect_embeddings and not embedding_collection_failed and emb_chunks:
        embedding_features = np.concatenate(emb_chunks, axis=0)
        embedding_labels = np.concatenate(emb_label_chunks, axis=0)
        total_emb = embedding_features.shape[0]
        if total_emb > max_embedding_samples:
            rng = np.random.RandomState(0)
            idx = rng.choice(total_emb, size=max_embedding_samples, replace=False)
            embedding_features = embedding_features[idx]
            embedding_labels = embedding_labels[idx]

    return {
        "loss": epoch_val_loss,
        "accuracy": epoch_val_acc,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "predictions": all_val_preds,
        "labels": all_val_labels,
        "embedding_features": embedding_features,
        "embedding_labels": embedding_labels,
    }


# ============================================================================
# Training Function
# ============================================================================

def train(model, train_loader, val_loader, config, experiment_dir, 
          loss_fn, optimizer, scheduler, num_epochs,
          val_fn=None, augmenter=None, apply_augmentation_fn=None):
    """
    Train the model with comprehensive logging and checkpointing.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        experiment_dir: Path to experiment directory
        loss_fn: Loss function (REQUIRED)
        optimizer: Pre-configured optimizer (REQUIRED)
        scheduler: Pre-configured scheduler (REQUIRED)
        num_epochs: Number of training epochs (REQUIRED)
        val_fn: Custom validation function (optional)
        augmenter: Data augmenter object (optional)
        apply_augmentation_fn: Function to apply augmentation (optional)
    
    Returns:
        model: Trained model
        train_history: Dictionary with training history
        best_checkpoint_path: Path to best model checkpoint
    """
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup loss function
    if loss_fn is None:
        raise ValueError("loss_fn is required - must be passed explicitly")

    # ---------------------------------------------------------------------
    # Peak RAM tracking (CPU RSS + optional CUDA peak)
    # ---------------------------------------------------------------------
    rss_kb_start = _get_process_peak_rss_kb()
    peak_rss_kb = rss_kb_start
    if device.type == "cuda":
        # Track CUDA peaks during the run (separate from system RAM).
        torch.cuda.reset_peak_memory_stats(device=device)
    
    # Setup directories
    experiment_path = Path(experiment_dir)
    logs_dir = experiment_path / "logs"
    models_dir = experiment_path / "models"
    tensorboard_dir = experiment_path / "tensorboard"
    
    # Setup logging
    log_file = logs_dir / "train.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger('train')
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    # Setup TensorBoard
    writer = SummaryWriter(str(tensorboard_dir))
    
    # Training parameters (all passed explicitly, no fallbacks)
    num_classes = config['vehicle_classification']['num_classes']
    class_names = config['vehicle_classification']['class_names']
    
    # Training history
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Best model tracking
    best_val_acc = 0.0
    best_epoch = 0
    best_val_cm = None

    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info(f"Device: {device}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info("=" * 80)
    
    for epoch in range(num_epochs):
        # ====================================================================
        # Training Phase
        # ====================================================================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)):
            # Unpack batch
            if len(batch_data) == 3:
                data, labels, idx = batch_data
            else:
                data, labels = batch_data[0], batch_data[1]
            
            # Apply augmentation if provided
            if augmenter is not None and apply_augmentation_fn is not None:
                data, labels = apply_augmentation_fn(augmenter, data, labels)
            
            # Move to device
            labels = labels.to(device)
            if isinstance(data, dict):
                # Multi-modal data
                for loc in data:
                    for mod in data[loc]:
                        data[loc][mod] = data[loc][mod].to(device)
            else:
                data = data.to(device)
            
            # Log input shape once (single-modality models still receive dict input).
            if epoch == 0 and batch_idx == 0:
                _log_single_modality_input_shape(
                    model, data, logger, epoch=epoch, batch_idx=batch_idx
                )
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            
            # Handle one-hot labels if needed
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels
            
            loss = loss_fn(outputs, loss_labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            clip_grad = config.get(config.get('model', 'ResNet'), {}).get('optimizer', {}).get('clip_grad', None)
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            
            # Update peak RSS (ru_maxrss is already a peak, but we keep an
            # explicit max for clear end-of-run logging).
            rss_kb_now = _get_process_peak_rss_kb()
            if rss_kb_now > peak_rss_kb:
                peak_rss_kb = rss_kb_now
            
            # Metrics
            train_loss += loss.item() * labels.size(0)
            
            # Extract logits for metrics (handle dict outputs)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            predictions = torch.argmax(logits, dim=1)
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                labels_idx = torch.argmax(labels, dim=1)
            else:
                labels_idx = labels
            
            train_correct += (predictions == labels_idx).sum().item()
            train_total += labels.size(0)
            
            all_train_preds.extend(predictions.cpu().numpy())
            all_train_labels.extend(labels_idx.cpu().numpy())
        
        # Calculate epoch training metrics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        train_history['train_loss'].append(epoch_train_loss)
        train_history['train_acc'].append(epoch_train_acc)
        
        # ====================================================================
        # Validation Phase
        # ====================================================================
        if val_fn is not None:
            # Use custom validation function
            val_results = val_fn(model, val_loader, loss_fn, device, config)
        else:
            # Use default validation function
            val_results = validate(model, val_loader, loss_fn, device, augmenter, apply_augmentation_fn)
        
        epoch_val_loss = val_results['loss']
        epoch_val_acc = val_results['accuracy']
        all_val_preds = val_results['predictions']
        all_val_labels = val_results['labels']
        
        train_history['val_loss'].append(epoch_val_loss)
        train_history['val_acc'].append(epoch_val_acc)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        train_history['learning_rates'].append(current_lr)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # ====================================================================
        # Logging
        # ====================================================================
        logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logger.info(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        logger.info(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Confusion matrix logging (every 5 epochs or last epoch)
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            # Training confusion matrix
            train_cm = calculate_confusion_matrix(all_train_preds, all_train_labels, num_classes)
            train_cm_fig = plot_confusion_matrix(train_cm, class_names=class_names, normalize=True)
            writer.add_figure('Confusion_Matrix/train', train_cm_fig, epoch)
            plt.close(train_cm_fig)
            
            # Validation confusion matrix
            val_cm = calculate_confusion_matrix(all_val_preds, all_val_labels, num_classes)
            val_cm_fig = plot_confusion_matrix(val_cm, class_names=class_names, normalize=True)
            writer.add_figure('Confusion_Matrix/val', val_cm_fig, epoch)
            plt.close(val_cm_fig)
            
            logger.info(f"  Confusion matrices logged to TensorBoard")
        
        # ====================================================================
        # Save Checkpoints
        # ====================================================================
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            best_val_cm = calculate_confusion_matrix(all_val_preds, all_val_labels, num_classes)
            best_model_path = models_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
                'config': config
            }, best_model_path)
            logger.info(f"  Best model saved! (Val Acc: {best_val_acc:.4f})")
        
        # Save last epoch
        last_model_path = models_dir / "last_epoch.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': epoch_val_acc,
            'val_loss': epoch_val_loss,
            'config': config
        }, last_model_path)
    
    # Final summary
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"TensorBoard logs: {tensorboard_dir}")
    logger.info("=" * 80)

    if best_val_cm is not None:
        log_confusion_matrix_table(
            best_val_cm,
            class_names,
            logger,
            title=f"Best Validation Confusion Matrix  (epoch {best_epoch + 1}  |  val_acc={best_val_acc:.4f})",
        )

    writer.close()

    # ---------------------------------------------------------------------
    # Final peak RAM logging
    # ---------------------------------------------------------------------
    rss_kb_end = _get_process_peak_rss_kb()
    peak_rss_mb = peak_rss_kb / 1024.0
    rss_delta_mb = (rss_kb_end - rss_kb_start) / 1024.0
    logger.info(
        f"Peak CPU RSS (ru_maxrss): {peak_rss_mb:.2f} MB (delta since start: {rss_delta_mb:.2f} MB)"
    )
    if device.type == "cuda":
        cuda_peak_alloc_mb = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)
        cuda_peak_reserved_mb = torch.cuda.max_memory_reserved(device=device) / (1024 * 1024)
        logger.info(
            f"Peak CUDA memory: allocated={cuda_peak_alloc_mb:.2f} MB, reserved={cuda_peak_reserved_mb:.2f} MB"
        )
    
    # Return model, history, and best checkpoint path
    best_checkpoint_path = str(models_dir / "best_model.pth")
    return model, train_history, best_checkpoint_path


def train_vanilla_supervised_contrastive(
    model,
    train_loader,
    val_loader,
    config,
    experiment_dir,
    loss_fn,
    optimizer,
    scheduler,
    num_epochs,
    val_fn=None,
    augmenter=None,
    apply_augmentation_fn=None,
):
    """
    Two-view supervised contrastive training.

    For each batch:
      - create two independent augmented views (view1, view2)
      - forward pass both views
      - compute CE + SupCon loss via `loss_fn((out1, out2), labels)`
      - use averaged logits from the two views for accuracy/confusion matrix
    """
    device = torch.device(config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if loss_fn is None:
        raise ValueError("loss_fn is required - must be passed explicitly")

    # ---------------------------------------------------------------------
    # Peak RAM tracking (CPU RSS + optional CUDA peak)
    # ---------------------------------------------------------------------
    rss_kb_start = _get_process_peak_rss_kb()
    peak_rss_kb = rss_kb_start
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    experiment_path = Path(experiment_dir)
    logs_dir = experiment_path / "logs"
    models_dir = experiment_path / "models"
    tensorboard_dir = experiment_path / "tensorboard"

    # Setup logging
    log_file = logs_dir / "train.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger = logging.getLogger("train_supcon")
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Setup TensorBoard
    writer = SummaryWriter(str(tensorboard_dir))

    num_classes = config["vehicle_classification"]["num_classes"]
    class_names = config["vehicle_classification"]["class_names"]

    train_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_val_cm = None

    logger.info("=" * 80)
    logger.info("Starting Training (vanilla_supervised_contrastive)")
    logger.info(f"Device: {device}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info("=" * 80)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        for batch_idx, batch_data in enumerate(
            tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs} [Train SupCon]",
                leave=False,
            )
        ):
            # Unpack batch
            if len(batch_data) == 3:
                data, labels, idx = batch_data
            else:
                data, labels = batch_data[0], batch_data[1]

            # Two augmented views
            if augmenter is not None and apply_augmentation_fn is not None:
                data_view1, labels_view1 = apply_augmentation_fn(augmenter, data, labels)
                data_view2, labels_view2 = apply_augmentation_fn(augmenter, data, labels)
            else:
                data_view1, data_view2 = data, data
                labels_view1, labels_view2 = labels, labels

            labels_view1 = labels_view1.to(device)
            if len(labels_view1.shape) == 2 and labels_view1.shape[1] > 1:
                loss_labels = torch.argmax(labels_view1, dim=1)
            else:
                loss_labels = labels_view1

            # Move both views to device
            if isinstance(data_view1, dict):
                for loc in data_view1:
                    for mod in data_view1[loc]:
                        data_view1[loc][mod] = data_view1[loc][mod].to(device)
            else:
                data_view1 = data_view1.to(device)

            if isinstance(data_view2, dict):
                for loc in data_view2:
                    for mod in data_view2[loc]:
                        data_view2[loc][mod] = data_view2[loc][mod].to(device)
            else:
                data_view2 = data_view2.to(device)

            # Log input shape once (view1 is sufficient since the two views share shape).
            if epoch == 0 and batch_idx == 0:
                _log_single_modality_input_shape(
                    model, data_view1, logger, epoch=epoch, batch_idx=batch_idx
                )

            optimizer.zero_grad()

            outputs_view1 = model(data_view1)
            outputs_view2 = model(data_view2)

            loss = loss_fn((outputs_view1, outputs_view2), loss_labels)
            loss.backward()

            clip_grad = config.get(config.get("model", "ResNet"), {}).get("optimizer", {}).get("clip_grad", None)
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            rss_kb_now = _get_process_peak_rss_kb()
            if rss_kb_now > peak_rss_kb:
                peak_rss_kb = rss_kb_now

            train_loss += loss.item() * loss_labels.size(0)

            logits1 = outputs_view1["logits"] if isinstance(outputs_view1, dict) else outputs_view1
            logits2 = outputs_view2["logits"] if isinstance(outputs_view2, dict) else outputs_view2
            logits_avg = 0.5 * (logits1 + logits2)
            predictions = torch.argmax(logits_avg, dim=1)

            train_correct += (predictions == loss_labels).sum().item()
            train_total += loss_labels.size(0)

            all_train_preds.extend(predictions.cpu().numpy())
            all_train_labels.extend(loss_labels.cpu().numpy())

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        train_history["train_loss"].append(epoch_train_loss)
        train_history["train_acc"].append(epoch_train_acc)

        # Validation Phase
        log_embedding_epoch = (epoch + 1) % 5 == 0 or epoch == num_epochs - 1
        if val_fn is not None:
            val_results = val_fn(model, val_loader, loss_fn, device, config)
        else:
            val_results = validate_vanilla_supervised_contrastive(
                model,
                val_loader,
                loss_fn,
                device,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation_fn,
                num_classes=num_classes,
                collect_embeddings=log_embedding_epoch,
                max_embedding_samples=2048,
            )

        epoch_val_loss = val_results["loss"]
        epoch_val_acc = val_results["accuracy"]
        all_val_preds = val_results["predictions"]
        all_val_labels = val_results["labels"]

        train_history["val_loss"].append(epoch_val_loss)
        train_history["val_acc"].append(epoch_val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        train_history["learning_rates"].append(current_lr)

        if scheduler is not None:
            scheduler.step()

        logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logger.info(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        logger.info(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")

        writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        writer.add_scalar("Loss/val", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_train_acc, epoch)
        writer.add_scalar("Accuracy/val", epoch_val_acc, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            train_cm = calculate_confusion_matrix(all_train_preds, all_train_labels, num_classes)
            train_cm_fig = plot_confusion_matrix(train_cm, class_names=class_names, normalize=True)
            writer.add_figure("Confusion_Matrix/train", train_cm_fig, epoch)
            plt.close(train_cm_fig)

            val_cm = calculate_confusion_matrix(all_val_preds, all_val_labels, num_classes)
            val_cm_fig = plot_confusion_matrix(val_cm, class_names=class_names, normalize=True)
            writer.add_figure("Confusion_Matrix/val", val_cm_fig, epoch)
            plt.close(val_cm_fig)

            logger.info("  Confusion matrices logged to TensorBoard")

            emb_f = val_results.get("embedding_features")
            emb_y = val_results.get("embedding_labels")
            if emb_f is not None and emb_y is not None:
                if log_supcon_embedding_scatter(
                    writer,
                    features=emb_f,
                    labels=emb_y,
                    class_names=class_names,
                    global_step=epoch,
                    logger=logger,
                ):
                    logger.info("  SupCon validation embedding PCA scatter logged to TensorBoard")
                if log_supcon_embedding_tsne_scatter(
                    writer,
                    features=emb_f,
                    labels=emb_y,
                    class_names=class_names,
                    global_step=epoch,
                    logger=logger,
                ):
                    logger.info("  SupCon validation embedding t-SNE scatter logged to TensorBoard")

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            best_val_cm = calculate_confusion_matrix(all_val_preds, all_val_labels, num_classes)
            best_model_path = models_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": epoch_val_acc,
                    "val_loss": epoch_val_loss,
                    "config": config,
                },
                best_model_path,
            )
            logger.info(f"  Best model saved! (Val Acc: {best_val_acc:.4f})")

        last_model_path = models_dir / "last_epoch.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": epoch_val_acc,
                "val_loss": epoch_val_loss,
                "config": config,
            },
            last_model_path,
        )

    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"TensorBoard logs: {tensorboard_dir}")
    logger.info("=" * 80)

    if best_val_cm is not None:
        log_confusion_matrix_table(
            best_val_cm,
            class_names,
            logger,
            title=f"Best Validation Confusion Matrix  (epoch {best_epoch + 1}  |  val_acc={best_val_acc:.4f})",
        )

    writer.close()
    best_checkpoint_path = str(models_dir / "best_model.pth")
    return model, train_history, best_checkpoint_path


# ============================================================================
# Testing Function
# ============================================================================

def test(model, test_loader, config, experiment_dir, checkpoint_path=None,
         loss_fn=None, test_fn=None, augmenter=None, apply_augmentation_fn=None):
    """
    Test the model and save results.
    
    Args:
        model: PyTorch model to test
        test_loader: Test data loader
        config: Configuration dictionary
        experiment_dir: Path to experiment directory
        checkpoint_path: Path to checkpoint file (optional, if None uses current model)
        loss_fn: Loss function (if None, uses CrossEntropyLoss)
        test_fn: Custom test function (optional)
        augmenter: Data augmenter for transformations (optional)
        apply_augmentation_fn: Function to apply augmentation (optional)
    
    Returns:
        test_results: Dictionary with test metrics
    """
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded checkpoint from: {checkpoint_path}")
        if 'epoch' in checkpoint:
            logging.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint:
            logging.info(f"  Checkpoint val accuracy: {checkpoint['val_acc']:.4f}")
    
    model = model.to(device)
    
    # Setup loss function
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    # Setup logging
    experiment_path = Path(experiment_dir)
    logs_dir = experiment_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / "test_results.txt"
    
    num_classes = config.get('vehicle_classification', {}).get('num_classes', 7)
    class_names = config.get('vehicle_classification', {}).get('class_names', None)
    
    # Use custom test function if provided
    if test_fn is not None:
        test_results = test_fn(model, test_loader, loss_fn, device, config)
        test_loss = test_results['loss']
        test_acc = test_results['accuracy']
        all_preds = test_results['predictions']
        all_labels = test_results['labels']
    else:
        # Default testing
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                # Unpack batch
                if len(batch_data) == 3:
                    data, labels, idx = batch_data
                else:
                    data, labels = batch_data[0], batch_data[1]
                
                # Apply augmentation if provided (for frequency transformation)
                if augmenter is not None and apply_augmentation_fn is not None:
                    data, labels = apply_augmentation_fn(augmenter, data, labels)
                
                # Move to device
                labels = labels.to(device)
                if isinstance(data, dict):
                    for loc in data:
                        for mod in data[loc]:
                            data[loc][mod] = data[loc][mod].to(device)
                else:
                    data = data.to(device)
                
                # Forward pass
                outputs = model(data)
                
                # Handle one-hot labels
                if len(labels.shape) == 2 and labels.shape[1] > 1:
                    loss_labels = torch.argmax(labels, dim=1)
                else:
                    loss_labels = labels
                
                loss = loss_fn(outputs, loss_labels)
                
                test_loss += loss.item() * labels.size(0)
                predictions = torch.argmax(outputs, dim=1)
                test_correct += (predictions == loss_labels).sum().item()
                test_total += labels.size(0)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(loss_labels.cpu().numpy())
        
        test_loss = test_loss / test_total
        test_acc = test_correct / test_total
    
    # Calculate confusion matrix
    cm = calculate_confusion_matrix(all_preds, all_labels, num_classes)
    f1_macro = calculate_macro_f1_from_confusion_matrix(cm)
    
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Save results to file
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(f"Test Macro-F1: {f1_macro:.4f}\n\n")
        
        f.write("Per-Class Accuracy:\n")
        for i, acc in enumerate(per_class_acc):
            class_name = class_names[i] if class_names else f"Class {i}"
            f.write(f"  {class_name}: {acc:.4f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        
        if checkpoint_path:
            f.write(f"Checkpoint: {checkpoint_path}\n")
        
        f.write("=" * 80 + "\n")
    
    # Save confusion matrix plot
    cm_fig = plot_confusion_matrix(cm, class_names=class_names, normalize=True)
    cm_fig.savefig(logs_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close(cm_fig)
    
    # Print results
    logging.info("=" * 80)
    logging.info("TEST RESULTS")
    logging.info("=" * 80)
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info(f"Test Macro-F1: {f1_macro:.4f}")
    logging.info(f"Results saved to: {log_file}")
    logging.info(f"Confusion matrix saved to: {logs_dir / 'confusion_matrix.png'}")
    logging.info("=" * 80)
    
    # Return results
    test_results = {
        'loss': test_loss,
        'accuracy': test_acc,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_acc,
        'predictions': all_preds,
        'labels': all_labels
    }
    
    return test_results

