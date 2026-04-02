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
import json
import logging
import yaml
import shutil
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import average_precision_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.W8A8Quant import log_w8a8_scales, has_w8a8_layers

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


def _logits_from_output(raw):
    """Normalize model output to logits tensor (dict with 'logits' or raw tensor)."""
    if isinstance(raw, dict):
        return raw["logits"]
    return raw


def simple_training_forward(model, data, config, model_name):
    """
    Single-location batch dict -> tensor forward matching FX trace targets.

    Uses config for location/modality keys (backbone-only models have no wrapper attrs).
    If model.backbone exists (e.g. SingleModalResNet), runs backbone(x); else model(x).
    """
    if model_name is None:
        raise ValueError("simple_model_training requires model_name for active_modality lookup")
    if not isinstance(data, dict):
        raise TypeError(
            "simple_model_training expects batch data as dict[location][modality]"
        )
    location_name = config["location_names"][0]
    modality_name = config["models"][model_name]["active_modality"]
    x = data[location_name][modality_name]
    if hasattr(model, "backbone"):
        return model.backbone(x)
    return model(x)


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

def _get_backbone_and_head_params(model):
    """Split model parameters into backbone vs head groups.

    Head modules: sample_embd_layer, output_dims_mlp, class_layer, projection_head.
    Everything else under model.backbone is backbone.

    Returns (backbone_params, head_params) — lists of parameter dicts.
    """
    head_module_names = {
        "sample_embd_layer", "output_dims_mlp", "class_layer", "projection_head",
    }
    bb = getattr(model, "backbone", model)

    backbone_params = []
    head_params = []

    for child_name, child_module in bb.named_children():
        dest = head_params if child_name in head_module_names else backbone_params
        for p in child_module.parameters():
            if p.requires_grad:
                dest.append(p)

    for p in model.parameters():
        if p.requires_grad and not any(p is q for q in backbone_params + head_params):
            head_params.append(p)

    return backbone_params, head_params


def setup_optimizer(model, config, training_config=None):
    """
    Create optimizer based on configuration.

    When training_config contains ``backbone_lr_scale`` (float, 0-1), parameters
    are split into backbone and head groups with the backbone receiving
    ``start_lr * backbone_lr_scale``.  This allows gentle finetuning of the
    pretrained backbone while training the head at full learning rate.

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

    backbone_lr_scale = training_config.get("backbone_lr_scale")
    if backbone_lr_scale is not None:
        backbone_params, head_params = _get_backbone_and_head_params(model)
        backbone_lr = start_lr * backbone_lr_scale
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": start_lr},
        ]
        logging.info(
            f"Differential LR: backbone_lr={backbone_lr:.2e} "
            f"(scale={backbone_lr_scale}), head_lr={start_lr:.2e}"
        )
        logging.info(
            f"  backbone params={sum(p.numel() for p in backbone_params):,}, "
            f"head params={sum(p.numel() for p in head_params):,}"
        )
    else:
        param_groups = model.parameters()

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=start_lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=start_lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        momentum = optimizer_config.get("momentum", 0.9)
        optimizer = torch.optim.SGD(
            param_groups,
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
    if "task_name" not in experiment_config:
        raise ValueError(
            f"Experiment '{experiment_name}' must define 'task_name'"
        )
    config["task_name"] = experiment_config["task_name"]
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
    If include_classes is set in config, restrict the active task to
    that subset and update config in-place with the remapped class info.

    Args:
        config: Full config dict (mutated in place)
    """
    include = config.get('include_classes')
    if not include:
        return

    include = sorted(set(include))
    task_cfg = config[config['task_name']]

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


# ----------------------------------------------------------------------------
# Machine-readable Claude log (JSON Lines)
# ----------------------------------------------------------------------------

def setup_claude_logging(experiment_dir):
    """Open a line-buffered JSON Lines file for machine-readable training logs.

    Returns:
        (Path, file handle) — caller is responsible for closing the handle.
    """
    logs_dir = Path(experiment_dir) / "logs"
    claude_log_file = logs_dir / "train_log_claude.jsonl"
    claude_log_fh = open(claude_log_file, "w", buffering=1)
    return claude_log_file, claude_log_fh


def log_header_to_claude(fh, model, config, num_epochs, optimizer, scheduler,
                          model_name, experiment_dir, train_type):
    """Write the header record (one per training run) to the Claude log."""
    model_cfg = config.get("models", {}).get(model_name, {}) if model_name else {}
    location = config.get("location_names", ["unknown"])[0]
    active_modality = model_cfg.get("active_modality", "unknown")

    record = {
        "record_type": "header",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_dir": str(experiment_dir),
        "train_type": train_type,
        "num_epochs": num_epochs,
        "num_classes": config[config["task_name"]]["num_classes"],
        "class_names": config[config["task_name"]]["class_names"],
        "model_summary": {
            "model_type": model_cfg.get("model_type", type(model).__name__),
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "filter_sizes": model_cfg.get("filter_sizes"),
            "stem_channels": model_cfg.get("stem_channels"),
            "dropout": model_cfg.get("dropout"),
            "num_early_exits": len(model_cfg.get("exit_after_blocks", [])),
        },
        "optimizer": {
            "name": type(optimizer).__name__,
            "start_lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay"),
        },
        "scheduler": {
            "name": type(scheduler).__name__ if scheduler is not None else "none",
        },
        "dataset": {
            "batch_size": config.get("batch_size"),
            "location": location,
            "active_modality": active_modality,
            "in_channels": config.get("loc_mod_in_freq_channels", {}).get(location, {}).get(active_modality),
            "num_segments": config.get("num_segments"),
        },
    }
    fh.write(json.dumps(record) + "\n")


def log_epoch_to_claude(fh, epoch, num_epochs, epoch_train_loss, epoch_train_acc,
                         val_results, current_lr, train_history, class_names, is_best):
    """Write one epoch record to the Claude log."""
    epoch_val_loss = val_results["loss"]
    epoch_val_acc = val_results["accuracy"]
    f1_macro = val_results.get("f1_macro")
    cm = val_results.get("confusion_matrix")

    acc_gap = round(float(epoch_val_acc - epoch_train_acc), 6)
    loss_gap = round(float(epoch_val_loss - epoch_train_loss), 6)

    val_loss_delta = None
    train_loss_delta = None
    if len(train_history["val_loss"]) >= 2:
        val_loss_delta = round(float(epoch_val_loss - train_history["val_loss"][-2]), 6)
        train_loss_delta = round(float(epoch_train_loss - train_history["train_loss"][-2]), 6)

    per_class_recall = {}
    per_class_precision = {}
    if cm is not None:
        cm_arr = np.asarray(cm)
        row_totals = cm_arr.sum(axis=1)
        col_totals = cm_arr.sum(axis=0)
        for i, name in enumerate(class_names):
            per_class_recall[name] = round(
                float(cm_arr[i, i] / row_totals[i]) if row_totals[i] > 0 else 0.0, 4
            )
            per_class_precision[name] = round(
                float(cm_arr[i, i] / col_totals[i]) if col_totals[i] > 0 else 0.0, 4
            )

    record = {
        "record_type": "epoch",
        "epoch": epoch + 1,
        "total_epochs": num_epochs,
        "train_loss": round(float(epoch_train_loss), 6),
        "train_acc": round(float(epoch_train_acc), 6),
        "val_loss": round(float(epoch_val_loss), 6),
        "val_acc": round(float(epoch_val_acc), 6),
        "f1_macro": round(float(f1_macro), 6) if f1_macro is not None else None,
        "learning_rate": current_lr,
        "overfit_gap": {"acc_gap": acc_gap, "loss_gap": loss_gap},
        "loss_convergence": {"val_loss_delta": val_loss_delta, "train_loss_delta": train_loss_delta},
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "is_best": is_best,
    }
    fh.write(json.dumps(record) + "\n")


def log_summary_to_claude(fh, best_epoch, best_val_acc, best_val_f1, best_val_cm,
                            train_history, peak_rss_kb, device, best_checkpoint_path,
                            class_names, status="completed"):
    """Write the final summary record to the Claude log and close the handle."""
    final_train_loss = train_history["train_loss"][-1] if train_history["train_loss"] else None
    final_val_loss = train_history["val_loss"][-1] if train_history["val_loss"] else None
    final_train_acc = train_history["train_acc"][-1] if train_history["train_acc"] else None
    final_val_acc = train_history["val_acc"][-1] if train_history["val_acc"] else None

    best_per_class_recall = {}
    if best_val_cm is not None:
        cm_arr = np.asarray(best_val_cm)
        row_totals = cm_arr.sum(axis=1)
        for i, name in enumerate(class_names):
            best_per_class_recall[name] = round(
                float(cm_arr[i, i] / row_totals[i]) if row_totals[i] > 0 else 0.0, 4
            )

    peak_cpu_rss_mb = round(peak_rss_kb / 1024.0, 2)
    peak_cuda_alloc_mb = None
    peak_cuda_reserved_mb = None
    if device.type == "cuda":
        peak_cuda_alloc_mb = round(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024), 2)
        peak_cuda_reserved_mb = round(torch.cuda.max_memory_reserved(device=device) / (1024 * 1024), 2)

    final_overfit_acc = round(float(final_val_acc - final_train_acc), 6) if (final_val_acc is not None and final_train_acc is not None) else None
    final_overfit_loss = round(float(final_val_loss - final_train_loss), 6) if (final_val_loss is not None and final_train_loss is not None) else None

    record = {
        "record_type": "summary",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_epochs_run": len(train_history["train_loss"]),
        "best_epoch": best_epoch + 1,
        "best_val_acc": round(float(best_val_acc), 6),
        "best_val_f1_macro": round(float(best_val_f1), 6) if best_val_f1 is not None else None,
        "best_val_per_class_recall": best_per_class_recall,
        "final_train_loss": round(float(final_train_loss), 6) if final_train_loss is not None else None,
        "final_val_loss": round(float(final_val_loss), 6) if final_val_loss is not None else None,
        "final_train_acc": round(float(final_train_acc), 6) if final_train_acc is not None else None,
        "final_val_acc": round(float(final_val_acc), 6) if final_val_acc is not None else None,
        "final_overfit_gap": {"acc_gap": final_overfit_acc, "loss_gap": final_overfit_loss},
        "peak_cpu_rss_mb": peak_cpu_rss_mb,
        "peak_cuda_alloc_mb": peak_cuda_alloc_mb,
        "peak_cuda_reserved_mb": peak_cuda_reserved_mb,
        "best_checkpoint_path": best_checkpoint_path,
        "status": status,
    }
    fh.write(json.dumps(record) + "\n")
    fh.close()


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

def validate(model, val_loader, loss_fn, device, augmenter=None, apply_augmentation_fn=None, num_classes=None,
             simple_model_training=False, model_name=None, config=None):
    """
    Default validation function.
    
    Args:
        model: PyTorch model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run validation on
        augmenter: Data augmenter object (optional)
        apply_augmentation_fn: Function to apply augmentation (optional)
        simple_model_training: If True, same tensor forward path as train() (requires config, model_name).
    
    Returns:
        val_results: Dictionary with validation metrics
            - 'loss': float
            - 'accuracy': float
            - 'predictions': list
            - 'labels': list
    """
    if simple_model_training:
        if config is None or model_name is None:
            raise ValueError(
                "validate(simple_model_training=True) requires config and model_name"
            )

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
            if simple_model_training:
                raw = simple_training_forward(model, data, config, model_name)
                logits = _logits_from_output(raw)
            else:
                outputs = model(data)
                logits = _logits_from_output(outputs)

            try:
                num_classes_from_outputs = int(logits.shape[1])
            except Exception:
                pass
            
            # Handle one-hot labels
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels
            
            loss = loss_fn(logits, loss_labels)
            
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


def validate_multilabel(
    model,
    val_loader,
    loss_fn,
    device,
    training_config,
    augmenter=None,
    apply_augmentation_fn=None,
    simple_model_training=False,
    model_name=None,
    config=None,
):
    """
    Threshold-free validation for multi-label BCE targets [B, C] float in {0, 1}.

    Metrics:
        - val loss (BCE)
        - mAP (mean Average Precision across classes)

    No thresholded metrics are computed here; optimal per-class thresholds
    are determined post-training via find_optimal_per_class_thresholds().

    Returns dict with:
        loss, mAP, accuracy (0.0 placeholder), confusion_matrix (None),
        raw_probs [N,C], raw_labels [N,C]
    """
    if simple_model_training:
        if config is None or model_name is None:
            raise ValueError(
                "validate_multilabel(simple_model_training=True) requires config and model_name"
            )

    model.eval()
    val_loss = 0.0
    val_total = 0
    all_prob_rows = []
    all_label_rows = []

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation", leave=False):
            if len(batch_data) == 3:
                data, labels, idx = batch_data
            else:
                data, labels = batch_data[0], batch_data[1]

            if augmenter is not None and apply_augmentation_fn is not None:
                data, labels = apply_augmentation_fn(augmenter, data, labels)

            labels = labels.to(device)
            if isinstance(data, dict):
                for loc in data:
                    for mod in data[loc]:
                        data[loc][mod] = data[loc][mod].to(device)
            else:
                data = data.to(device)

            if simple_model_training:
                raw = simple_training_forward(model, data, config, model_name)
                logits = _logits_from_output(raw)
            else:
                outputs = model(data)
                logits = _logits_from_output(outputs)

            loss = loss_fn(logits, labels)
            val_loss += loss.item() * labels.size(0)
            val_total += labels.size(0)

            probs = torch.sigmoid(logits)
            y_true = labels.float()

            all_prob_rows.append(probs.cpu().numpy())
            all_label_rows.append(y_true.cpu().numpy())

    epoch_val_loss = val_loss / val_total

    if len(all_prob_rows) == 0:
        return {
            "loss": epoch_val_loss,
            "mAP": 0.0,
            "accuracy": 0.0,
            "confusion_matrix": None,
            "raw_probs": np.empty((0, 0)),
            "raw_labels": np.empty((0, 0)),
        }

    y_prob = np.concatenate(all_prob_rows, axis=0)
    y_true = np.concatenate(all_label_rows, axis=0)

    per_class_ap = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            per_class_ap.append(0.0)
        else:
            per_class_ap.append(
                float(average_precision_score(y_true[:, c], y_prob[:, c]))
            )
    mAP = float(np.mean(per_class_ap))

    return {
        "loss": epoch_val_loss,
        "mAP": mAP,
        "accuracy": 0.0,
        "confusion_matrix": None,
        "raw_probs": y_prob,
        "raw_labels": y_true,
    }


def find_optimal_per_class_thresholds(
    y_true,
    y_prob,
    class_names,
    thresholds=None,
    metric="f1",
):
    """
    Sweep thresholds per class on raw sigmoid probabilities to find each
    class's F1-maximizing (or precision/recall-maximizing) cutoff.

    Args:
        y_true:  np.ndarray [N, C] binary ground-truth
        y_prob:  np.ndarray [N, C] sigmoid probabilities
        class_names: list[str] length C
        thresholds: 1-D array of candidate thresholds to evaluate.
            Defaults to np.arange(0.05, 0.96, 0.05).
        metric: "f1" (default) — maximize per-class F1.

    Returns:
        dict with:
            "per_class": list of dicts, one per class:
                {"class": str, "best_threshold": float, "best_f1": float,
                 "curve": list of {"threshold": float, "precision": float,
                                   "recall": float, "f1": float}}
            "global_macro_f1_at_best": float  (macro-F1 when each class uses its own best threshold)
            "global_subset_acc_at_best": float
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    n_samples, n_classes = y_prob.shape
    per_class_results = []

    best_thresholds = np.zeros(n_classes, dtype=np.float64)

    for c in range(n_classes):
        col_true = y_true[:, c]
        col_prob = y_prob[:, c]
        curve = []
        best_f1 = -1.0
        best_t = 0.5

        for t in thresholds:
            col_pred = (col_prob >= t).astype(np.float64)
            tp = float(np.sum((col_pred == 1) & (col_true == 1)))
            fp = float(np.sum((col_pred == 1) & (col_true == 0)))
            fn = float(np.sum((col_pred == 0) & (col_true == 1)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            curve.append({
                "threshold": round(float(t), 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            })
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        best_thresholds[c] = best_t
        per_class_results.append({
            "class": class_names[c],
            "best_threshold": round(best_t, 4),
            "best_f1": round(best_f1, 4),
            "curve": curve,
        })

    y_pred_optimal = (y_prob >= best_thresholds[np.newaxis, :]).astype(np.float64)
    global_macro_f1 = float(
        f1_score(y_true, y_pred_optimal, average="macro", zero_division=0)
    )
    global_subset_acc = float(
        (y_pred_optimal == y_true).all(axis=1).mean()
    )

    return {
        "per_class": per_class_results,
        "best_thresholds": {
            class_names[c]: round(best_thresholds[c], 4) for c in range(n_classes)
        },
        "global_macro_f1_at_best": round(global_macro_f1, 4),
        "global_subset_acc_at_best": round(global_subset_acc, 4),
    }


def log_threshold_analysis(logger, writer, threshold_results, experiment_dir):
    """Log per-class threshold sweep results and save to disk."""
    logger.info("=" * 80)
    logger.info("OPTIMAL PER-CLASS THRESHOLD ANALYSIS (validation set)")
    logger.info("=" * 80)

    for cls_result in threshold_results["per_class"]:
        logger.info(
            f"  {cls_result['class']:20s}  best_threshold={cls_result['best_threshold']:.4f}  "
            f"best_f1={cls_result['best_f1']:.4f}"
        )

    logger.info(
        f"  Global macro-F1 at per-class optimal: "
        f"{threshold_results['global_macro_f1_at_best']:.4f}"
    )
    logger.info(
        f"  Global subset accuracy at per-class optimal: "
        f"{threshold_results['global_subset_acc_at_best']:.4f}"
    )
    logger.info(f"  Per-class thresholds: {threshold_results['best_thresholds']}")
    logger.info("=" * 80)

    for cls_result in threshold_results["per_class"]:
        cls_name = cls_result["class"]
        for pt in cls_result["curve"]:
            t = pt["threshold"]
            writer.add_scalar(f"ThresholdCurve/{cls_name}/precision", pt["precision"], int(t * 1000))
            writer.add_scalar(f"ThresholdCurve/{cls_name}/recall", pt["recall"], int(t * 1000))
            writer.add_scalar(f"ThresholdCurve/{cls_name}/f1", pt["f1"], int(t * 1000))

    out_path = Path(experiment_dir) / "logs" / "optimal_thresholds.json"
    with open(out_path, "w") as f:
        json.dump(threshold_results, f, indent=2)
    logger.info(f"Threshold analysis saved to: {out_path}")


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
          val_fn=None, augmenter=None, apply_augmentation_fn=None,
          model_name=None, training_config=None, simple_model_training=False):
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
        training_config: Experiment training_configs entry; required fields for
            bce_multilabel (loss_name, multilabel_best_metric)
        simple_model_training: If True, unpack dict batch via config + model_name,
            forward tensor through model.backbone or model, apply loss on logits only.

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

    multilabel = (
        training_config is not None
        and training_config["loss_name"] == "bce_multilabel"
    )
    if multilabel and val_fn is not None:
        raise ValueError(
            "bce_multilabel training does not support a custom val_fn; use val_fn=None"
        )

    if simple_model_training and model_name is None:
        raise ValueError("simple_model_training requires model_name")

    # ---------------------------------------------------------------------
    # Peak RAM tracking (CPU RSS + optional CUDA peak)
    # ---------------------------------------------------------------------
    rss_kb_start = _get_process_peak_rss_kb()
    peak_rss_kb = rss_kb_start
    # if device.type == "cuda":
    #     # Track CUDA peaks during the run (separate from system RAM).
    #     torch.cuda.reset_peak_memory_stats(device=device)

    # Setup directories
    experiment_path = Path(experiment_dir)
    logs_dir = experiment_path / "logs"
    models_dir = experiment_path / "models"
    tensorboard_dir = experiment_path / "tensorboard"
    
    # Logger — NO extra file handler to avoid double-write bug.
    # The root logger already has a file handler from setup_train_file_logging().
    # This named logger propagates to root, giving a single copy per log line.
    logger = logging.getLogger('train')

    # Setup TensorBoard
    writer = SummaryWriter(str(tensorboard_dir))

    # Setup Claude machine-readable log
    _claude_log_file, _claude_fh = setup_claude_logging(experiment_dir)
    log_header_to_claude(
        _claude_fh, model, config, num_epochs,
        optimizer, scheduler,
        model_name=model_name,
        experiment_dir=experiment_dir,
        train_type="vanilla_supervised",
    )

    # Training parameters: resolve task block via task_name (same as create_models).
    _task_name = config["task_name"]
    task_cfg = config[_task_name]
    num_classes = task_cfg["num_classes"]
    class_names = task_cfg["class_names"]

    ml_best_metric = None
    if multilabel:
        ml_best_metric = training_config["multilabel_best_metric"]
        allowed_ml_metrics = ("val_loss", "mAP")
        if ml_best_metric not in allowed_ml_metrics:
            raise ValueError(
                f"multilabel_best_metric must be one of {allowed_ml_metrics}, "
                f"got '{ml_best_metric}'"
            )

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
    best_val_f1 = None
    best_val_mAP = None
    best_epoch = 0
    best_val_cm = None
    best_selection_score = None

    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info(f"Device: {device}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Task (metrics): {_task_name}")
    if multilabel:
        logger.info(f"Multilabel BCE: best_metric={ml_best_metric}")
    logger.info(f"Experiment directory: {experiment_dir}")
    if simple_model_training:
        logger.info(
            "simple_model_training: unpack dict in loop; forward tensor via "
            "model.backbone(x) or model(x); loss on logits tensor only"
        )
    logger.info("=" * 80)
    
    for epoch in range(num_epochs):
        # ====================================================================
        # Training Phase
        # ====================================================================
        model.train()
        re_eval_frozen_modules(model)
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
            if simple_model_training:
                raw = simple_training_forward(model, data, config, model_name)
                logits = _logits_from_output(raw)
            else:
                outputs = model(data)
                logits = _logits_from_output(outputs)

            if multilabel:
                loss_labels = labels.float()
            elif len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels

            loss = loss_fn(logits, loss_labels)

            # Backward pass
            loss.backward()
            
            # Gradient clipping
            clip_grad = config.get(config.get('model', 'ResNet'), {}).get('optimizer', {}).get('clip_grad', None)
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            
            # Update peak RSS (ru_maxrss is already a peak, but we keep an
            # explicit max for clear end-of-run logging).
            # rss_kb_now = _get_process_peak_rss_kb()
            # if rss_kb_now > peak_rss_kb:
            #     peak_rss_kb = rss_kb_now
            
            # Metrics
            train_loss += loss.item() * labels.size(0)
            if not multilabel:
                predictions = torch.argmax(logits, dim=1)
                if len(labels.shape) == 2 and labels.shape[1] > 1:
                    labels_idx = torch.argmax(labels, dim=1)
                else:
                    labels_idx = labels
                train_correct += (predictions == labels_idx).sum().item()
                all_train_preds.extend(predictions.cpu().numpy())
                all_train_labels.extend(labels_idx.cpu().numpy())
            train_total += labels.size(0)
        
        # Calculate epoch training metrics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        train_history['train_loss'].append(epoch_train_loss)
        train_history['train_acc'].append(epoch_train_acc)
        
        # ====================================================================
        # Validation Phase
        # ====================================================================
        if val_fn is not None:
            val_results = val_fn(model, val_loader, loss_fn, device, config)
        elif multilabel:
            val_results = validate_multilabel(
                model,
                val_loader,
                loss_fn,
                device,
                training_config,
                augmenter,
                apply_augmentation_fn,
                simple_model_training=simple_model_training,
                model_name=model_name,
                config=config,
            )
        else:
            val_results = validate(
                model,
                val_loader,
                loss_fn,
                device,
                augmenter,
                apply_augmentation_fn,
                simple_model_training=simple_model_training,
                model_name=model_name,
                config=config,
            )
        
        epoch_val_loss = val_results['loss']
        epoch_val_acc = val_results['accuracy']
        if multilabel:
            all_val_preds = val_results['raw_probs']
            all_val_labels = val_results['raw_labels']
        else:
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

        # Claude epoch log (before checkpoint block so is_best reflects pre-update state)
        if multilabel:
            if ml_best_metric == "val_loss":
                curr_sel = val_results["loss"]
                _is_best_epoch = best_selection_score is None or curr_sel < best_selection_score
            elif ml_best_metric == "mAP":
                curr_sel = val_results["mAP"]
                _is_best_epoch = best_selection_score is None or curr_sel > best_selection_score
            else:
                raise ValueError(f"Unknown multilabel_best_metric: {ml_best_metric}")
        else:
            _is_best_epoch = epoch_val_acc > best_val_acc
        log_epoch_to_claude(
            _claude_fh,
            epoch=epoch,
            num_epochs=num_epochs,
            epoch_train_loss=epoch_train_loss,
            epoch_train_acc=epoch_train_acc,
            val_results=val_results,
            current_lr=current_lr,
            train_history=train_history,
            class_names=class_names,
            is_best=_is_best_epoch,
        )

        # ====================================================================
        # Logging
        # ====================================================================
        logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logger.info(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        logger.info(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        if multilabel:
            logger.info(f"  Val mAP: {val_results['mAP']:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        if multilabel:
            writer.add_scalar('Metrics/val_mAP', val_results['mAP'], epoch)

        if has_w8a8_layers(model):
            log_w8a8_scales(model, writer, epoch)


        # Confusion matrix logging (every 5 epochs or last epoch)
        if not multilabel and ((epoch + 1) % 5 == 0 or epoch == num_epochs - 1):
            train_cm = calculate_confusion_matrix(all_train_preds, all_train_labels, num_classes)
            train_cm_fig = plot_confusion_matrix(train_cm, class_names=class_names, normalize=True)
            writer.add_figure('Confusion_Matrix/train', train_cm_fig, epoch)
            plt.close(train_cm_fig)
            
            val_cm = calculate_confusion_matrix(all_val_preds, all_val_labels, num_classes)
            val_cm_fig = plot_confusion_matrix(val_cm, class_names=class_names, normalize=True)
            writer.add_figure('Confusion_Matrix/val', val_cm_fig, epoch)
            plt.close(val_cm_fig)
            
            logger.info(f"  Confusion matrices logged to TensorBoard")
        
        # ====================================================================
        # Save Checkpoints
        # ====================================================================
        if multilabel:
            if _is_best_epoch:
                best_selection_score = curr_sel
                best_val_mAP = val_results["mAP"]
                best_epoch = epoch
                best_val_cm = None
                best_model_path = models_dir / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_val_loss,
                    'val_mAP': val_results['mAP'],
                    'config': config
                }, best_model_path)
                logger.info(
                    f"  Best model saved! (metric={ml_best_metric}={curr_sel:.4f}, "
                    f"mAP={best_val_mAP:.4f}, loss={epoch_val_loss:.4f})"
                )
        elif epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_val_f1 = val_results.get('f1_macro')
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
    if multilabel:
        logger.info(
            f"Best checkpoint at epoch {best_epoch + 1} "
            f"(selected by {ml_best_metric}={best_selection_score:.4f}, "
            f"mAP={best_val_mAP:.4f})"
        )
    else:
        logger.info(
            f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}"
        )
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

    # -----------------------------------------------------------------
    # Per-class threshold optimization (multilabel only)
    # -----------------------------------------------------------------
    if multilabel:
        logger.info("\nRunning per-class threshold optimization on best model...")
        best_ckpt = torch.load(
            str(models_dir / "best_model.pth"),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_ckpt["model_state_dict"])
        final_val = validate_multilabel(
            model,
            val_loader,
            loss_fn,
            device,
            training_config,
            augmenter,
            apply_augmentation_fn,
            simple_model_training=simple_model_training,
            model_name=model_name,
            config=config,
        )
        threshold_results = find_optimal_per_class_thresholds(
            final_val["raw_labels"],
            final_val["raw_probs"],
            class_names,
        )
        log_threshold_analysis(logger, writer, threshold_results, experiment_dir)

    writer.close()

    # ---------------------------------------------------------------------
    # Final peak RAM logging
    # ---------------------------------------------------------------------
    # rss_kb_end = _get_process_peak_rss_kb()
    # peak_rss_mb = peak_rss_kb / 1024.0
    # rss_delta_mb = (rss_kb_end - rss_kb_start) / 1024.0
    # logger.info(
    #     f"Peak CPU RSS (ru_maxrss): {peak_rss_mb:.2f} MB (delta since start: {rss_delta_mb:.2f} MB)"
    # )
    # if device.type == "cuda":
    #     cuda_peak_alloc_mb = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)
    #     cuda_peak_reserved_mb = torch.cuda.max_memory_reserved(device=device) / (1024 * 1024)
    #     logger.info(
    #         f"Peak CUDA memory: allocated={cuda_peak_alloc_mb:.2f} MB, reserved={cuda_peak_reserved_mb:.2f} MB"
    #     )

    # Return model, history, and best checkpoint path
    best_checkpoint_path = str(models_dir / "best_model.pth")

    log_summary_to_claude(
        _claude_fh,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        best_val_f1=best_val_f1,
        best_val_cm=best_val_cm,
        train_history=train_history,
        peak_rss_kb=peak_rss_kb,
        device=device,
        best_checkpoint_path=best_checkpoint_path,
        class_names=class_names,
        status="completed",
    )

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
    model_name=None,
    training_config=None,
):
    """
    Two-view supervised contrastive training.

    training_config is accepted for API parity with train() / finetune.py; unused here.

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

    # Logger — NO extra file handler to avoid double-write bug.
    # The root logger already has a file handler from setup_train_file_logging().
    # This named logger propagates to root, giving a single copy per log line.
    logger = logging.getLogger("train_supcon")

    # Setup TensorBoard
    writer = SummaryWriter(str(tensorboard_dir))

    # Setup Claude machine-readable log
    _claude_log_file, _claude_fh = setup_claude_logging(experiment_dir)
    log_header_to_claude(
        _claude_fh, model, config, num_epochs,
        optimizer, scheduler,
        model_name=model_name,
        experiment_dir=experiment_dir,
        train_type="vanilla_supervised_contrastive",
    )

    num_classes = config[config["task_name"]]["num_classes"]
    class_names = config[config["task_name"]]["class_names"]

    train_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
    }

    best_val_acc = 0.0
    best_val_f1 = None
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
        re_eval_frozen_modules(model)
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

        # Claude epoch log (before checkpoint block so is_best reflects pre-update state)
        _is_best_epoch = epoch_val_acc > best_val_acc
        log_epoch_to_claude(
            _claude_fh,
            epoch=epoch,
            num_epochs=num_epochs,
            epoch_train_loss=epoch_train_loss,
            epoch_train_acc=epoch_train_acc,
            val_results=val_results,
            current_lr=current_lr,
            train_history=train_history,
            class_names=class_names,
            is_best=_is_best_epoch,
        )

        logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logger.info(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        logger.info(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")

        writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        writer.add_scalar("Loss/val", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_train_acc, epoch)
        writer.add_scalar("Accuracy/val", epoch_val_acc, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        if has_w8a8_layers(model):
            log_w8a8_scales(model, writer, epoch)

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
            best_val_f1 = val_results.get("f1_macro")
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

    log_summary_to_claude(
        _claude_fh,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        best_val_f1=best_val_f1,
        best_val_cm=best_val_cm,
        train_history=train_history,
        peak_rss_kb=peak_rss_kb,
        device=device,
        best_checkpoint_path=best_checkpoint_path,
        class_names=class_names,
        status="completed",
    )

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
    
    num_classes = config[config["task_name"]]["num_classes"]
    class_names = config[config["task_name"]]["class_names"]
    
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


# ============================================================================
# Pretrain Config Validation
# ============================================================================

def validate_pretrain_config(config):
    """
    Validate config before starting SSL pretraining.

    Checks:
      - experiment_name is set and exists in config["experiments"]
      - resolved training config has type == "ssl_pretrain"
      - pretrain_index_file is set and exists on disk
      - pretrain_subset_ratio is in (0, 1]
      - pretrain_subset_mode is "global" or "stratified"
        (stratified raises NotImplementedError immediately)

    Raises:
        ValueError: for missing or invalid config fields
        FileNotFoundError: if pretrain_index_file path does not exist
        NotImplementedError: if pretrain_subset_mode == "stratified"
    """
    experiment_name = config.get("experiment_name")
    if not experiment_name:
        raise ValueError("experiment_name not set in config")

    experiments = config.get("experiments", {})
    if experiment_name not in experiments:
        available = [k for k in experiments if k != "enabled"]
        raise ValueError(
            f"Experiment '{experiment_name}' not found in config['experiments']. "
            f"Available: {available}"
        )

    experiment_config = experiments[experiment_name]
    training_config_name = experiment_config.get("training")
    if not training_config_name:
        raise ValueError(f"Experiment '{experiment_name}' has no 'training' key")

    training_configs = config.get("training_configs", {})
    if training_config_name not in training_configs:
        raise ValueError(
            f"Training config '{training_config_name}' not found in config['training_configs']"
        )

    training_config = training_configs[training_config_name]
    train_type = training_config.get("type")
    if train_type not in ("ssl_pretrain", "supervised_pretrain"):
        raise ValueError(
            f"Training config '{training_config_name}' has type='{train_type}', "
            "expected 'ssl_pretrain' or 'supervised_pretrain' for pretraining."
        )

    pretrain_index_file = config.get("pretrain_index_file")
    if not pretrain_index_file:
        raise ValueError(
            "pretrain_index_file not set in config. "
            "Add 'pretrain_index_file: /path/to/index.txt' to your YAML."
        )
    if not os.path.exists(pretrain_index_file):
        raise FileNotFoundError(
            f"pretrain_index_file not found on disk: {pretrain_index_file}"
        )

    subset_ratio = config.get("pretrain_subset_ratio", 1.0)
    if not (0 < float(subset_ratio) <= 1.0):
        raise ValueError(
            f"pretrain_subset_ratio must be in (0, 1], got {subset_ratio}"
        )

    subset_mode = config.get("pretrain_subset_mode", "global")
    if subset_mode not in ("global", "stratified"):
        raise ValueError(
            f"pretrain_subset_mode must be 'global' or 'stratified', got '{subset_mode}'"
        )
    if subset_mode == "stratified":
        raise NotImplementedError(
            "Stratified pretrain subset is configured but not yet implemented. "
            "Use pretrain_subset_mode: 'global' instead."
        )

    logging.info("Pretrain config validation passed")
    logging.info(f"  experiment: {experiment_name}")
    logging.info(f"  training config: {training_config_name} (type={train_type})")
    logging.info(f"  pretrain_index_file: {pretrain_index_file}")
    logging.info(f"  subset_ratio={subset_ratio}, subset_mode={subset_mode}")


def validate_finetune_config(config):
    """
    Validate config before supervised fine-tuning from a pretrained checkpoint.

    Expects training_configs[..].type == "finetune", loss_name == "cross_entropy",
    and a non-empty checkpoint path (training_config or model entry).

    Returns:
        tuple:
            experiment_name,
            experiment_config,
            model_name,
            training_config_name,
            training_config,
            resolved_checkpoint_path (str),
            num_epochs (int),
    """
    experiment_name = config["experiment_name"]

    if "experiments" not in config or not config["experiments"]["enabled"]:
        raise ValueError(
            "Experiments not enabled in config. Set experiments.enabled: true"
        )

    experiments = config["experiments"]
    available = [k for k in experiments if k != "enabled"]
    if experiment_name not in experiments:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. Available: {available}"
        )

    experiment_config = experiments[experiment_name]
    if "task_name" not in experiment_config:
        raise ValueError(
            f"Experiment '{experiment_name}' must define 'task_name'"
        )
    config["task_name"] = experiment_config["task_name"]
    model_name = experiment_config["model"]
    training_config_name = experiment_config["training"]

    training_configs = config["training_configs"]
    if training_config_name not in training_configs:
        raise ValueError(
            f"Training config '{training_config_name}' not found in training_configs"
        )

    training_config = training_configs[training_config_name]
    train_type = training_config["type"]
    if train_type != "finetune":
        raise ValueError(
            f"Training config '{training_config_name}' has type='{train_type}', "
            "expected 'finetune'."
        )

    loss_name = training_config["loss_name"]
    allowed = ("cross_entropy", "ce_supcon", "bce_multilabel")
    if loss_name not in allowed:
        raise ValueError(
            f"Finetune expects loss_name one of {allowed}, got '{loss_name}'"
        )

    if loss_name == "bce_multilabel":
        if "multilabel_best_metric" not in training_config:
            raise ValueError(
                "bce_multilabel finetune requires multilabel_best_metric in training_config "
                "('val_loss' or 'mAP')"
            )
        mbm = training_config["multilabel_best_metric"]
        if mbm not in ("val_loss", "mAP"):
            raise ValueError(
                f"multilabel_best_metric must be 'val_loss' or 'mAP', got '{mbm}'"
            )

    if "freeze_backbone" not in training_config:
        raise ValueError(
            "finetune training_config must set 'freeze_backbone' (true or false)"
        )

    # breakpoint()
    tc_path = experiment_config["checkpoint_path"]
    if isinstance(tc_path, str) and tc_path.strip():
        checkpoint_path = tc_path.strip()
    else:
        model_cfg = config["models"][model_name]
        if "checkpoint_path" not in model_cfg:
            raise ValueError(
                "finetune requires checkpoint_path in training_config or "
                f"config['models']['{model_name}']['checkpoint_path']"
            )
        mp = experiment_config["checkpoint_path"]
        if not isinstance(mp, str) or not mp.strip():
            raise ValueError(
                "finetune checkpoint_path is empty in training_config and in "
                f"models['{model_name}']['checkpoint_path']"
            )
        checkpoint_path = mp.strip()

    checkpoint_path = str(Path(checkpoint_path).expanduser())
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Finetune checkpoint not found: {checkpoint_path}"
        )

    num_epochs = training_config["epochs"]

    logging.info("Finetune config validation passed")
    logging.info(f"  experiment: {experiment_name}")
    logging.info(f"  model: {model_name}")
    logging.info(f"  training config: {training_config_name}")
    logging.info(f"  checkpoint_path: {checkpoint_path}")
    logging.info(f"  freeze_backbone: {training_config['freeze_backbone']}")
    logging.info(f"  epochs: {num_epochs}")

    return (
        experiment_name,
        experiment_config,
        model_name,
        training_config_name,
        training_config,
        checkpoint_path,
        num_epochs,
    )


def _filter_checkpoint_keys(state_dict, model_sd):
    """Filter checkpoint state_dict for safe loading into model.

    Removes:
      - keys containing 'projection_head' (SSL head, not needed for finetune)
      - keys whose shape differs from the model (e.g. class_layer when num_classes changed)
    """
    out = {}
    shape_skipped = []
    for k, v in state_dict.items():
        if "projection_head" in k:
            continue
        if k in model_sd and model_sd[k].shape != v.shape:
            shape_skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        out[k] = v
    return out, shape_skipped


def load_pretrained_backbone(model, checkpoint_path, logger=None):
    """
    Load weights from a pretrain (or full) checkpoint into a supervised model.

    Strips projection_head keys and any keys with shape mismatches (e.g.
    class_layer when num_classes differs).  Uses strict=False so missing
    keys (new head layers) are silently initialized from scratch.

    Args:
        model: nn.Module (e.g. SingleModalResNet with pretrain_mode=False)
        checkpoint_path: Path to .pth file with 'model_state_dict' or raw state_dict
        logger: optional logging.Logger

    Returns:
        tuple: (missing_keys, unexpected_keys) from load_state_dict
    """
    log = logger if logger is not None else logging.getLogger("finetune")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        raw_sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and ckpt and all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        raw_sd = ckpt
    else:
        raise ValueError(
            f"Unrecognized checkpoint format at {checkpoint_path}: "
            "expected dict with 'model_state_dict' or a flat state_dict of tensors"
        )

    model_sd = model.state_dict()
    filtered, shape_skipped = _filter_checkpoint_keys(raw_sd, model_sd)
    incompatible = model.load_state_dict(filtered, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    log.info(
        "Loaded pretrained weights: "
        f"{len(filtered)} keys loaded, missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if shape_skipped:
        log.info(f"  shape_skipped (ckpt->model): {shape_skipped}")
    if missing:
        log.info(f"  missing_keys (first 20): {missing[:20]}")
    if unexpected:
        log.info(f"  unexpected_keys (first 20): {unexpected[:20]}")
    return missing, unexpected


def apply_finetune_backbone_freeze(model):
    """
    Set requires_grad=False on early backbone blocks; train head (embed + classifier).

    Frozen modules are also switched to eval mode so that BatchNorm layers
    preserve their pretrained running_mean / running_var instead of being
    overwritten by fine-tune batch statistics.  A list of frozen modules is
    stored on the model so the training loop can re-apply eval after the
    global model.train() call each epoch (see re_eval_frozen_modules).

    Supports ConfigurableResNet (conv1, bn1, maxpool, stages), DeepSenseDepthwiseBackbone
    (freq_stack, spectrum_proj, temporal_stack), and DeepSenseBackbone
    (conv_stack, spectrum_proj, recurrent_layer).
    """
    if not hasattr(model, "backbone"):
        raise ValueError(
            "apply_finetune_backbone_freeze expects model.backbone (single-modal wrapper)"
        )
    bb = model.backbone

    frozen_modules = []

    def _freeze_module(m):
        for p in m.parameters():
            p.requires_grad = False
        m.eval()
        frozen_modules.append(m)

    if hasattr(bb, "conv1") and hasattr(bb, "stages"):
        _freeze_module(bb.conv1)
        _freeze_module(bb.bn1)
        _freeze_module(bb.maxpool)
        for stage in bb.stages:
            _freeze_module(stage)
    elif hasattr(bb, "freq_stack"):
        for layer in bb.freq_stack:
            _freeze_module(layer)
        _freeze_module(bb.spectrum_proj)
        for layer in bb.temporal_stack:
            _freeze_module(layer)
        if getattr(bb, "recurrent_layer", None) is not None:
            _freeze_module(bb.recurrent_layer)
    elif hasattr(bb, "conv_stack"):
        for layer in bb.conv_stack:
            _freeze_module(layer)
        _freeze_module(bb.spectrum_proj)
        _freeze_module(bb.recurrent_layer)
    else:
        raise ValueError(
            f"Unknown backbone type for freeze_backbone: {type(bb).__name__}"
        )

    model._frozen_modules = frozen_modules

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logging.info(
        f"freeze_backbone=True: trainable params={trainable:,}, frozen={frozen:,}"
    )


def freeze_backbone_bn(model):
    """Set all backbone BatchNorm layers to eval mode (preserve running stats).

    Unlike apply_finetune_backbone_freeze, this does NOT freeze weights — the
    backbone parameters remain trainable.  Use this when doing differential-LR
    finetuning so the backbone learns slowly while BatchNorm statistics stay
    anchored to the pretrained distribution.

    Stores the affected BN modules in ``model._frozen_bn_modules`` so that
    ``re_eval_frozen_modules`` can re-apply eval after ``model.train()``.
    """
    if not hasattr(model, "backbone"):
        return
    bb = model.backbone

    head_names = {
        "sample_embd_layer", "output_dims_mlp", "class_layer", "projection_head",
    }
    frozen_bn = []
    for name, module in bb.named_modules():
        top_level = name.split(".")[0] if name else ""
        if top_level in head_names:
            continue
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()
            frozen_bn.append(module)

    model._frozen_bn_modules = frozen_bn
    logging.info(f"freeze_backbone_bn: {len(frozen_bn)} BN layers set to eval (weights still trainable)")


def re_eval_frozen_modules(model):
    """
    After model.train(), re-set any frozen backbone modules back to eval mode.

    This prevents BatchNorm layers in frozen modules from updating their
    running statistics during fine-tuning.  Call this immediately after
    model.train() in the training loop when freeze_backbone=True or
    backbone_lr_scale is used.
    """
    for m in getattr(model, "_frozen_modules", []):
        m.eval()
    for m in getattr(model, "_frozen_bn_modules", []):
        m.eval()


# ============================================================================
# Pretrain JSON Logging (machine-readable JSONL, separate from Python logging)
# ============================================================================

def _open_pretrain_json_log(experiment_dir):
    """Open a line-buffered JSONL file for machine-readable pretrain logs."""
    logs_dir = Path(experiment_dir) / "logs"
    json_log_path = logs_dir / "pretrain_log.json"
    fh = open(json_log_path, "w", buffering=1)
    return json_log_path, fh


def _log_pretrain_header_json(fh, model, config, num_epochs, optimizer, scheduler,
                               model_name, experiment_dir):
    """Write the pretrain header record to the JSON log."""
    model_cfg = config.get("models", {}).get(model_name, {}) if model_name else {}
    record = {
        "record_type": "header",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_dir": str(experiment_dir),
        "train_type": "ssl_pretrain",
        "num_epochs": num_epochs,
        "model_summary": {
            "model_type": model_cfg.get("model_type", type(model).__name__),
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "pretrain_mode": model_cfg.get("pretrain_mode", True),
            "proj_hidden_dim": model_cfg.get("proj_hidden_dim", 256),
            "proj_out_dim": model_cfg.get("proj_out_dim", 128),
        },
        "optimizer": {
            "name": type(optimizer).__name__,
            "start_lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay"),
        },
        "scheduler": {
            "name": type(scheduler).__name__ if scheduler is not None else "none",
        },
        "pretrain_dataset": {
            "pretrain_index_file": config.get("pretrain_index_file"),
            "pretrain_subset_ratio": config.get("pretrain_subset_ratio", 1.0),
            "pretrain_subset_mode": config.get("pretrain_subset_mode", "global"),
            "batch_size": config.get("batch_size", 256),
        },
    }
    fh.write(json.dumps(record) + "\n")


def _log_pretrain_epoch_json(fh, epoch, num_epochs, epoch_loss, current_lr,
                              ssl_metrics, is_best, loss_history):
    """Write one SSL epoch record to the JSON log."""
    loss_delta = None
    if len(loss_history) >= 2:
        loss_delta = round(float(epoch_loss - loss_history[-2]), 6)

    record = {
        "record_type": "epoch",
        "epoch": epoch + 1,
        "total_epochs": num_epochs,
        "loss": round(float(epoch_loss), 6),
        "loss_delta": loss_delta,
        "learning_rate": current_lr,
        "ssl_metrics": {k: round(float(v), 6) for k, v in ssl_metrics.items()},
        "is_best": is_best,
    }
    fh.write(json.dumps(record) + "\n")


def _log_pretrain_summary_json(fh, best_epoch, best_loss, num_epochs_run,
                                loss_history, peak_rss_kb, device,
                                best_checkpoint_path, status="completed"):
    """Write final summary record to the JSON log and close the handle."""
    peak_cpu_rss_mb = round(peak_rss_kb / 1024.0, 2)
    peak_cuda_alloc_mb = None
    peak_cuda_reserved_mb = None
    if device.type == "cuda":
        peak_cuda_alloc_mb = round(
            torch.cuda.max_memory_allocated(device=device) / (1024 * 1024), 2
        )
        peak_cuda_reserved_mb = round(
            torch.cuda.max_memory_reserved(device=device) / (1024 * 1024), 2
        )

    record = {
        "record_type": "summary",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_epochs_run": num_epochs_run,
        "best_epoch": best_epoch + 1,
        "best_loss": round(float(best_loss), 6) if best_loss is not None else None,
        "final_loss": round(float(loss_history[-1]), 6) if loss_history else None,
        "peak_cpu_rss_mb": peak_cpu_rss_mb,
        "peak_cuda_alloc_mb": peak_cuda_alloc_mb,
        "peak_cuda_reserved_mb": peak_cuda_reserved_mb,
        "best_checkpoint_path": best_checkpoint_path,
        "status": status,
    }
    fh.write(json.dumps(record) + "\n")
    fh.close()


# ============================================================================
# Pretrain Visualization Helpers
# ============================================================================

def _log_pretrain_embedding_pca(writer, features_np, labels_np, global_step,
                                  viz_dir, logger, tag="pretrain/embeddings_pca"):
    """
    PCA-reduce pretrain backbone features to 2D, log to TensorBoard, and save PNG.
    Features are L2-normalized before projection (matches NT-Xent geometry).
    """
    if features_np is None or len(features_np) == 0:
        return False
    n, d = features_np.shape
    if d < 1:
        return False

    norms = np.linalg.norm(features_np, axis=1, keepdims=True)
    features_np = features_np / np.clip(norms, a_min=1e-12, a_max=None)

    n_comp = min(2, d)
    coords = PCA(n_components=n_comp, random_state=0).fit_transform(features_np)

    fig, ax = plt.subplots(figsize=(10, 8))
    for c in np.unique(labels_np):
        mask = labels_np == c
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1] if n_comp == 2 else np.zeros(mask.sum()),
            label=str(int(c)),
            alpha=0.6,
            s=12,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2" if n_comp == 2 else "(1D)")
    ax.set_title(f"Pretrain features PCA (epoch {global_step + 1}, n={n})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()

    writer.add_figure(tag, fig, global_step)
    pca_path = viz_dir / f"pca_epoch_{global_step + 1}.png"
    fig.savefig(str(pca_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  PCA scatter saved: {pca_path.name}")
    return True


def _log_pretrain_embedding_tsne(writer, features_np, labels_np, global_step,
                                   viz_dir, logger, tag="pretrain/embeddings_tsne"):
    """
    t-SNE-reduce pretrain backbone features to 2D, log to TensorBoard, and save PNG.
    Hard cap at 2000 samples (t-SNE is O(n^2)). Features are L2-normalized.
    """
    n = len(features_np)
    if n < 3:
        logger.warning("Pretrain t-SNE skipped: too few samples.")
        return False

    norms = np.linalg.norm(features_np, axis=1, keepdims=True)
    features_np = features_np / np.clip(norms, a_min=1e-12, a_max=None)

    perplexity = min(30, max(5, n // 10))
    if perplexity >= n:
        perplexity = max(2, n - 1)

    coords = TSNE(
        n_components=2,
        random_state=0,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(features_np)

    fig, ax = plt.subplots(figsize=(10, 8))
    for c in np.unique(labels_np):
        mask = labels_np == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=str(int(c)), alpha=0.6, s=12,
        )
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.set_title(f"Pretrain features t-SNE (epoch {global_step + 1}, n={n})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()

    writer.add_figure(tag, fig, global_step)
    tsne_path = viz_dir / f"tsne_epoch_{global_step + 1}.png"
    fig.savefig(str(tsne_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  t-SNE scatter saved: {tsne_path.name}")
    return True


# ============================================================================
# SSL Pretraining Loop
# ============================================================================

def pretrain(
    model,
    train_loader,
    config,
    experiment_dir,
    loss_fn,
    augmenter,
    apply_augmentation_fn,
    optimizer,
    scheduler,
    num_epochs,
    model_name=None,
):
    """
    SSL pretraining loop with NT-Xent contrastive loss (SimCLR-style).

    For each batch, two independent augmented views are generated and passed
    through the model's projection head. NT-Xent loss is computed on the
    normalized projections. Backbone features are not touched by the loss —
    they are only used for collapse detection and visualization.

    Checkpoints saved to experiment_dir/models/:
      - pretrain_epoch_N.pth       every 25 epochs
      - best_pretrain_model.pth    lowest NT-Xent loss seen
      - last_pretrain_model.pth    end of training

    Metrics logged per epoch to TensorBoard and pretrain_log.json:
      pretrain/loss              NT-Xent loss (mean over batches)
      pretrain/lr                current learning rate
      pretrain/feature_norm_mean mean L2 norm of backbone features (collapse detector)
      pretrain/proj_norm_mean    mean L2 norm of raw projection outputs
      pretrain/pos_similarity    mean cosine sim between the two views of same sample
      pretrain/neg_similarity    mean cosine sim between views of different samples

    Visualizations (PCA + t-SNE) are generated every viz_every_n_epochs epochs
    (default 10) from at most 2000 backbone feature vectors, colored by label.
    Saved to experiment_dir/viz/ and logged to TensorBoard.

    Args:
        model:                 PyTorch model with pretrain_mode=True, returning
                               {'features': Tensor, 'projection': Tensor}
        train_loader:          DataLoader returning (data, labels, idx) batches
        config:                Full config dict (includes training_configs section)
        experiment_dir:        Path to experiment directory
        loss_fn:               NTXentLoss instance
        augmenter:             Augmenter instance with internal random state
        apply_augmentation_fn: Callable(augmenter, data) -> (aug_data, None)
        optimizer:             Pre-configured optimizer
        scheduler:             Pre-configured LR scheduler (or None)
        num_epochs:            Number of pretraining epochs
        model_name:            Model name key for JSON log header (optional)

    Returns:
        model:                 Trained model (still in pretrain_mode)
        best_loss:             Best NT-Xent loss achieved during training
        best_checkpoint_path:  Path to best_pretrain_model.pth
    """
    device = torch.device(
        config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    if loss_fn is None:
        raise ValueError("loss_fn is required for SSL pretraining")

    # ------------------------------------------------------------------
    # Resolve training config for pretrain-specific hyperparameters
    # ------------------------------------------------------------------
    experiment_name = config.get("experiment_name")
    experiment_config = config["experiments"][experiment_name]
    training_config = config["training_configs"][experiment_config["training"]]
    viz_every_n_epochs = int(training_config.get("viz_every_n_epochs", 10))
    clip_grad = training_config.get("optimizer", {}).get("clip_grad", None)
    if clip_grad is not None:
        clip_grad = float(clip_grad)

    # ------------------------------------------------------------------
    # Peak RAM tracking
    # ------------------------------------------------------------------
    rss_kb_start = _get_process_peak_rss_kb()
    peak_rss_kb = rss_kb_start
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    # ------------------------------------------------------------------
    # Directory setup
    # ------------------------------------------------------------------
    experiment_path = Path(experiment_dir)
    models_dir = experiment_path / "models"
    tensorboard_dir = experiment_path / "tensorboard"
    viz_dir = experiment_path / "viz"
    viz_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Logger — NO extra file handler to avoid double-write bug.
    # The root logger already has a file handler from setup_train_file_logging().
    # This named logger propagates to root, giving a single copy per log line.
    # ------------------------------------------------------------------
    logger = logging.getLogger("pretrain")

    # ------------------------------------------------------------------
    # TensorBoard writer
    # ------------------------------------------------------------------
    writer = SummaryWriter(str(tensorboard_dir))

    # ------------------------------------------------------------------
    # Machine-readable JSON log (written directly, not via logging module)
    # ------------------------------------------------------------------
    _json_log_path, _json_fh = _open_pretrain_json_log(experiment_dir)
    _log_pretrain_header_json(
        _json_fh, model, config, num_epochs,
        optimizer, scheduler,
        model_name=model_name,
        experiment_dir=experiment_dir,
    )
    logger.info(f"Pretrain JSON log: {_json_log_path}")

    # ------------------------------------------------------------------
    # Helper: move data (tensor or nested dict) to device
    # ------------------------------------------------------------------
    def _to_device(x):
        if isinstance(x, dict):
            return {
                loc: {mod: t.to(device) for mod, t in mods.items()}
                for loc, mods in x.items()
            }
        return x.to(device)

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------
    loss_history = []
    best_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path = str(models_dir / "best_pretrain_model.pth")

    from train_test.loss import SupervisedPretrainLoss
    pretrain_mode_label = (
        "Supervised SupCon + CE" if isinstance(loss_fn, SupervisedPretrainLoss)
        else "NT-Xent / SimCLR (self-supervised)"
    )
    logger.info("=" * 80)
    logger.info(f"Starting Pretraining ({pretrain_mode_label})")
    logger.info(f"  Device: {device}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batches per epoch: {len(train_loader)}")
    logger.info(f"  Experiment dir: {experiment_dir}")
    logger.info(f"  Visualization every {viz_every_n_epochs} epochs -> {viz_dir}")
    if clip_grad:
        logger.info(f"  Gradient clipping: {clip_grad}")
    logger.info("=" * 80)

    for epoch in range(num_epochs):
        model.train()

        # Batch-level accumulators (averaged at epoch end)
        epoch_loss_sum = 0.0
        epoch_feature_norm_sum = 0.0
        epoch_proj_norm_sum = 0.0
        epoch_pos_sim_sum = 0.0
        epoch_neg_sim_sum = 0.0
        epoch_batches = 0

        # Embedding collection for visualization (only on viz epochs)
        is_viz_epoch = (
            (epoch + 1) % viz_every_n_epochs == 0 or epoch == num_epochs - 1
        )
        viz_max_samples = 2000
        viz_features = []
        viz_labels = []
        viz_collected = 0

        for batch_idx, batch_data in enumerate(
            tqdm(train_loader, desc=f"Pretrain {epoch+1}/{num_epochs}", leave=False)
        ):
            data, labels, _ = batch_data

            # ----------------------------------------------------------
            # Two independent augmented views.
            # The augmenter's internal random state advances on each call,
            # so view1 and view2 have different random transforms applied.
            # ----------------------------------------------------------
            view1, _ = apply_augmentation_fn(augmenter, data)
            view2, _ = apply_augmentation_fn(augmenter, data)

            view1 = _to_device(view1)
            view2 = _to_device(view2)
            labels = labels.to(device)

            # Log input shape once to confirm data format
            if epoch == 0 and batch_idx == 0:
                _log_single_modality_input_shape(
                    model, view1, logger, epoch=epoch, batch_idx=batch_idx
                )

            # ----------------------------------------------------------
            # Forward pass through the model in pretrain_mode=True.
            # Model returns {'features': [B, feat_dim], 'projection': [B, proj_dim],
            #                'logits': [B, C]} (logits used only by SupervisedPretrainLoss).
            # ----------------------------------------------------------
            optimizer.zero_grad()
            out1 = model(view1)
            out2 = model(view2)

            proj1 = out1["projection"]    # [B, proj_dim]
            proj2 = out2["projection"]    # [B, proj_dim]
            feat1 = out1["features"]      # [B, feat_dim] — for metrics/viz only

            # Loss dispatch:
            #   SupervisedPretrainLoss — supervised SupCon + weighted CE; needs labels.
            #   NTXentLoss (default)   — self-supervised; operates on projections only.
            from train_test.loss import SupervisedPretrainLoss
            if isinstance(loss_fn, SupervisedPretrainLoss):
                loss = loss_fn(out1, out2, labels)
            else:
                loss = loss_fn(proj1, proj2)
            loss.backward()

            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            # ----------------------------------------------------------
            # SSL metrics — computed without gradients on detached tensors
            # ----------------------------------------------------------
            with torch.no_grad():
                B = proj1.shape[0]

                # Backbone feature norms (collapse detector)
                feat_norms = feat1.detach().norm(dim=1)        # [B]

                # Projection norms (before NT-Xent normalization)
                proj_norms = proj1.detach().norm(dim=1)        # [B]

                # L2-normalize projections (same as inside NT-Xent)
                z1 = F.normalize(proj1.detach(), dim=1)        # [B, D]
                z2 = F.normalize(proj2.detach(), dim=1)        # [B, D]

                # Positive-pair cosine similarity: dot product of same-sample views
                pos_sim_batch = (z1 * z2).sum(dim=1)           # [B]

                # Negative-pair cosine similarity: cross-sample (z1 vs z2 off-diag)
                sim_cross = torch.matmul(z1, z2.T)             # [B, B]
                off_diag = ~torch.eye(B, dtype=torch.bool, device=z1.device)
                neg_sim_batch = sim_cross[off_diag]            # B*(B-1) values

                epoch_feature_norm_sum += feat_norms.mean().item()
                epoch_proj_norm_sum += proj_norms.mean().item()
                epoch_pos_sim_sum += pos_sim_batch.mean().item()
                epoch_neg_sim_sum += neg_sim_batch.mean().item()

                # Collect backbone features + labels for visualization
                if is_viz_epoch and viz_collected < viz_max_samples:
                    n_take = min(B, viz_max_samples - viz_collected)
                    viz_features.append(
                        feat1.detach().cpu().float().numpy()[:n_take]
                    )
                    viz_labels.extend(labels.cpu().numpy()[:n_take].tolist())
                    viz_collected += n_take

            epoch_loss_sum += loss.item()
            epoch_batches += 1

            rss_now = _get_process_peak_rss_kb()
            if rss_now > peak_rss_kb:
                peak_rss_kb = rss_now

        # --------------------------------------------------------------
        # End of epoch — aggregate metrics
        # --------------------------------------------------------------
        epoch_loss = epoch_loss_sum / epoch_batches
        feature_norm_mean = epoch_feature_norm_sum / epoch_batches
        proj_norm_mean = epoch_proj_norm_sum / epoch_batches
        pos_similarity = epoch_pos_sim_sum / epoch_batches
        neg_similarity = epoch_neg_sim_sum / epoch_batches

        loss_history.append(epoch_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step()

        ssl_metrics = {
            "feature_norm_mean": feature_norm_mean,
            "proj_norm_mean": proj_norm_mean,
            "pos_similarity": pos_similarity,
            "neg_similarity": neg_similarity,
        }

        # --------------------------------------------------------------
        # Collapse detection
        # If feature norms collapse to near zero the backbone has degenerated.
        # --------------------------------------------------------------
        if feature_norm_mean < 1e-3:
            logger.warning(
                f"COLLAPSE DETECTED epoch {epoch+1}: "
                f"feature_norm_mean={feature_norm_mean:.2e} < 1e-3. "
                "Backbone may be outputting near-zero embeddings."
            )

        # --------------------------------------------------------------
        # Text logging (single write — propagates to root file handler only)
        # --------------------------------------------------------------
        logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logger.info(f"  loss={epoch_loss:.4f}  lr={current_lr:.6f}")
        logger.info(
            f"  feature_norm={feature_norm_mean:.4f}  proj_norm={proj_norm_mean:.4f}"
        )
        logger.info(
            f"  pos_sim={pos_similarity:.4f}  neg_sim={neg_similarity:.4f}"
        )

        # --------------------------------------------------------------
        # TensorBoard scalars
        # --------------------------------------------------------------
        writer.add_scalar("pretrain/loss", epoch_loss, epoch)
        writer.add_scalar("pretrain/lr", current_lr, epoch)
        writer.add_scalar("pretrain/feature_norm_mean", feature_norm_mean, epoch)
        writer.add_scalar("pretrain/proj_norm_mean", proj_norm_mean, epoch)
        writer.add_scalar("pretrain/pos_similarity", pos_similarity, epoch)
        writer.add_scalar("pretrain/neg_similarity", neg_similarity, epoch)

        # --------------------------------------------------------------
        # JSON epoch record
        # --------------------------------------------------------------
        is_best = epoch_loss < best_loss
        _log_pretrain_epoch_json(
            _json_fh,
            epoch=epoch,
            num_epochs=num_epochs,
            epoch_loss=epoch_loss,
            current_lr=current_lr,
            ssl_metrics=ssl_metrics,
            is_best=is_best,
            loss_history=loss_history,
        )

        # --------------------------------------------------------------
        # Checkpointing
        # --------------------------------------------------------------
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "config": config,
        }

        # Periodic checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            periodic_path = models_dir / f"pretrain_epoch_{epoch+1}.pth"
            torch.save(checkpoint_state, periodic_path)
            logger.info(f"  Checkpoint saved: {periodic_path.name}")

        # Best checkpoint (lowest NT-Xent loss)
        if is_best:
            best_loss = epoch_loss
            best_epoch = epoch
            torch.save(checkpoint_state, best_checkpoint_path)
            logger.info(f"  Best model saved (loss={best_loss:.4f})")

        # --------------------------------------------------------------
        # Visualization: PCA + t-SNE on collected backbone features
        # --------------------------------------------------------------
        if is_viz_epoch and viz_collected > 0:
            feats_np = np.concatenate(viz_features, axis=0)
            labels_np = np.array(viz_labels, dtype=np.int64)
            logger.info(
                f"  Generating visualizations ({viz_collected} samples)..."
            )
            _log_pretrain_embedding_pca(
                writer, feats_np, labels_np,
                global_step=epoch, viz_dir=viz_dir, logger=logger,
            )
            _log_pretrain_embedding_tsne(
                writer, feats_np, labels_np,
                global_step=epoch, viz_dir=viz_dir, logger=logger,
            )

    # ------------------------------------------------------------------
    # Save last-epoch checkpoint
    # ------------------------------------------------------------------
    last_checkpoint_path = str(models_dir / "last_pretrain_model.pth")
    torch.save(
        {
            "epoch": num_epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_history[-1] if loss_history else None,
            "config": config,
        },
        last_checkpoint_path,
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    rss_kb_end = _get_process_peak_rss_kb()
    peak_rss_mb = peak_rss_kb / 1024.0
    rss_delta_mb = (rss_kb_end - rss_kb_start) / 1024.0

    logger.info("=" * 80)
    logger.info("SSL Pretraining Complete!")
    logger.info(f"  Best loss: {best_loss:.4f}  (epoch {best_epoch + 1})")
    logger.info(f"  Best checkpoint: {best_checkpoint_path}")
    logger.info(f"  Last checkpoint: {last_checkpoint_path}")
    logger.info(
        f"  Peak CPU RSS: {peak_rss_mb:.2f} MB (delta: {rss_delta_mb:.2f} MB)"
    )
    if device.type == "cuda":
        cuda_alloc = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)
        cuda_rsv = torch.cuda.max_memory_reserved(device=device) / (1024 * 1024)
        logger.info(
            f"  Peak CUDA: allocated={cuda_alloc:.2f} MB, "
            f"reserved={cuda_rsv:.2f} MB"
        )
    logger.info("=" * 80)

    writer.close()

    _log_pretrain_summary_json(
        _json_fh,
        best_epoch=best_epoch,
        best_loss=best_loss,
        num_epochs_run=len(loss_history),
        loss_history=loss_history,
        peak_rss_kb=peak_rss_kb,
        device=device,
        best_checkpoint_path=best_checkpoint_path,
        status="completed",
    )

    return model, best_loss, best_checkpoint_path
