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
import logging
import yaml
import shutil
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ============================================================================
# Optimizer and Scheduler Setup
# ============================================================================

def setup_optimizer(model, config, experiment_config=None):
    """
    Create optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary (full config)
        experiment_config: Experiment-specific config (optional, for distillation)
    
    Returns:
        optimizer: Configured optimizer
    """
    # If experiment_config provided, use its optimizer settings
    optimizer_config = experiment_config['optimizer']
    
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


def setup_scheduler(optimizer, config, experiment_config, stage_config):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary (full config)
        experiment_config: Experiment-specific config (optional, for distillation)
        num_epochs: Number of training epochs (optional, for distillation stages)
    
    Returns:
        scheduler: Learning rate scheduler (or None)
    """
    # If experiment_config provided, use its scheduler settings
    scheduler_config = experiment_config['lr_scheduler']
    scheduler_name = scheduler_config['name']
    train_epochs = stage_config['epochs']
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


def validate_with_early_exits(model, val_loader, loss_fn, device, num_classes=None, augmenter=None, apply_augmentation_fn=None):
    """
    Validation function for models with early exits.
    
    Evaluates each exit independently to track how well each intermediate
    classifier is performing.
    
    Args:
        model: PyTorch model with early exits
        val_loader: Validation data loader
        loss_fn: Loss function (handles early exits)
        device: Device to run validation on
        num_classes: Number of classes (optional, inferred from outputs if None)
        augmenter: Data augmenter object (optional)
        apply_augmentation_fn: Function to apply augmentation (optional)
    
    Returns:
        dict: {
            'total_loss': float,  # Combined loss from all exits
            'exits': [
                {'loss': float, 'accuracy': float, 'predictions': list, 'labels': list},
                {'loss': float, 'accuracy': float, 'predictions': list, 'labels': list},
                ...  # One entry per early exit
            ],
            'final': {'loss': float, 'accuracy': float, 'predictions': list, 'labels': list}
        }
    """
    model.eval()
    
    # Tracking for combined loss
    total_loss = 0.0
    total_samples = 0
    
    # Initialize tracking for each exit
    # We'll determine the number of exits from the first batch
    exit_stats = None
    final_stats = None
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation (Early Exits)", leave=False):
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
            
            # Initialize stats on first batch
            if exit_stats is None:
                num_early_exits = len(outputs['exits'])
                exit_stats = [
                    {'loss': 0.0, 'correct': 0, 'total': 0, 'predictions': [], 'labels': []}
                    for _ in range(num_early_exits)
                ]
                final_stats = {'loss': 0.0, 'correct': 0, 'total': 0, 'predictions': [], 'labels': []}
            
            # Handle one-hot labels
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels
            
            # Calculate total loss (for all exits combined)
            batch_loss = loss_fn(outputs, loss_labels)
            total_loss += batch_loss.item() * labels.size(0)
            total_samples += labels.size(0)
            
            # Calculate individual cross-entropy loss for each exit
            ce_loss_fn = nn.CrossEntropyLoss()
            
            # Process each early exit
            for idx, exit_logits in enumerate(outputs['exits']):
                exit_loss = ce_loss_fn(exit_logits, loss_labels)
                predictions = torch.argmax(exit_logits, dim=1)
                
                exit_stats[idx]['loss'] += exit_loss.item() * labels.size(0)
                exit_stats[idx]['correct'] += (predictions == loss_labels).sum().item()
                exit_stats[idx]['total'] += labels.size(0)
                exit_stats[idx]['predictions'].extend(predictions.cpu().numpy())
                exit_stats[idx]['labels'].extend(loss_labels.cpu().numpy())
            
            # Process final exit
            final_logits = outputs['logits']
            final_loss = ce_loss_fn(final_logits, loss_labels)
            predictions = torch.argmax(final_logits, dim=1)
            
            final_stats['loss'] += final_loss.item() * labels.size(0)
            final_stats['correct'] += (predictions == loss_labels).sum().item()
            final_stats['total'] += labels.size(0)
            final_stats['predictions'].extend(predictions.cpu().numpy())
            final_stats['labels'].extend(loss_labels.cpu().numpy())
    
    # Calculate average losses and accuracies
    results = {
        'total_loss': total_loss / total_samples,
        'exits': [],
        'final': {}
    }
    
    # Format exit stats
    for stats in exit_stats:
        results['exits'].append({
            'loss': stats['loss'] / stats['total'],
            'accuracy': stats['correct'] / stats['total'],
            'predictions': stats['predictions'],
            'labels': stats['labels']
        })
    
    # Format final stats
    results['final'] = {
        'loss': final_stats['loss'] / final_stats['total'],
        'accuracy': final_stats['correct'] / final_stats['total'],
        'predictions': final_stats['predictions'],
        'labels': final_stats['labels']
    }
    
    return results


def log_early_exits_to_tensorboard(writer, epoch, train_results, val_results, 
                                    num_classes, class_names=None, memory_info=None, 
                                    input_memory_info=None, test=False):
    """
    Log early exit metrics to TensorBoard.
    
    For training (test=False):
    - Creates time-series plots for loss and accuracy per exit
    - Logs confusion matrices every 5 epochs or last epoch
    - Logs memory requirements as text (once at epoch 0)
    
    For testing (test=True):
    - Logs all metrics as text (single evaluation, no time-series)
    - Logs confusion matrices as figures
    - Logs memory requirements as text
    
    Args:
        writer: TensorBoard SummaryWriter
        epoch: Current epoch number
        train_results: Training results dict with 'exits' and 'final' keys (can be None if test=True)
        val_results: Validation/test results dict from validate_with_early_exits()
        num_classes: Number of classes
        class_names: List of class names (optional)
        memory_info: Dict from get_early_exit_memory() with memory requirements (optional)
        input_memory_info: Dict from get_input_memory() with input memory info (optional)
        test: If True, log all metrics as text (single evaluation). 
              If False, log time-series metrics as scalars (training).
    """
    num_exits = len(val_results['exits'])
    
    # --- Memory logging (always as text, not plots) ---
    if input_memory_info and epoch == 0:
        memory_text = "## Input Memory\n\n"
        for info in input_memory_info['shape_info']:
            input_name = f"{info['location']}_{info['modality']}"
            memory_text += f"- {input_name}: {info['memory']:.2f} KB\n"
        memory_text += f"\n**Total**: {input_memory_info['total_memory']:.2f} KB\n"
        writer.add_text('Memory/Input', memory_text, 0)
    
    if memory_info and epoch == 0:
        mem_text = "## Model Memory Requirements\n\n"
        if memory_info['has_early_exits']:
            for exit_idx in range(num_exits):
                exit_mem = memory_info['exits'][exit_idx]
                mem_text += f"### Exit {exit_idx + 1}\n"
                mem_text += f"- Parameters: {exit_mem['parameter_memory']:.2f} KB\n"
                mem_text += f"- Activations: {exit_mem['activation_memory']:.2f} KB\n"
                mem_text += f"- Total: {exit_mem['total_memory']:.2f} KB\n\n"
        
        final_mem = memory_info['final']
        mem_text += f"### Final Exit\n"
        mem_text += f"- Parameters: {final_mem['parameter_memory']:.2f} KB\n"
        mem_text += f"- Activations: {final_mem['activation_memory']:.2f} KB\n"
        mem_text += f"- Total: {final_mem['total_memory']:.2f} KB\n"
        writer.add_text('Memory/Model', mem_text, 0)
    
    # --- Test mode: log everything as text ---
    if test:
        results_text = "## Test Results\n\n"
        
        for exit_idx in range(num_exits):
            exit_num = exit_idx + 1
            val_exit = val_results['exits'][exit_idx]
            results_text += f"### Exit {exit_num}\n"
            results_text += f"- Loss: {val_exit['loss']:.4f}\n"
            results_text += f"- Accuracy: {val_exit['accuracy']:.4f}\n\n"
        
        val_final = val_results['final']
        results_text += f"### Final Exit\n"
        results_text += f"- Loss: {val_final['loss']:.4f}\n"
        results_text += f"- Accuracy: {val_final['accuracy']:.4f}\n\n"
        results_text += f"### Total\n"
        results_text += f"- Loss: {val_results['total_loss']:.4f}\n"
        
        writer.add_text('Test_Results', results_text, 0)
        
        # Log confusion matrices for test
        for exit_idx in range(num_exits):
            exit_num = exit_idx + 1
            val_exit = val_results['exits'][exit_idx]
            val_cm = calculate_confusion_matrix(
                val_exit['predictions'], val_exit['labels'], num_classes
            )
            val_cm_fig = plot_confusion_matrix(val_cm, class_names=class_names, normalize=True)
            writer.add_figure(f'Confusion_Matrix_Exit{exit_num}/test', val_cm_fig, 0)
            plt.close(val_cm_fig)
        
        val_final = val_results['final']
        val_cm = calculate_confusion_matrix(
            val_final['predictions'], val_final['labels'], num_classes
        )
        val_cm_fig = plot_confusion_matrix(val_cm, class_names=class_names, normalize=True)
        writer.add_figure('Confusion_Matrix_Final/test', val_cm_fig, 0)
        plt.close(val_cm_fig)
        
        return  # Exit early for test mode
    
    # --- Training mode: log time-series as scalars ---
    
    # Log metrics for each early exit
    for exit_idx in range(num_exits):
        exit_num = exit_idx + 1
        train_exit = train_results['exits'][exit_idx]
        val_exit = val_results['exits'][exit_idx]
        
        # Loss
        writer.add_scalar(f'Loss_Exit{exit_num}/train', train_exit['loss'], epoch)
        writer.add_scalar(f'Loss_Exit{exit_num}/val', val_exit['loss'], epoch)
        
        # Accuracy
        writer.add_scalar(f'Accuracy_Exit{exit_num}/train', train_exit['accuracy'], epoch)
        writer.add_scalar(f'Accuracy_Exit{exit_num}/val', val_exit['accuracy'], epoch)
        
        # Confusion matrices (every 5 epochs or last epoch)
        if (epoch + 1) % 5 == 0 or epoch == train_results.get('total_epochs', epoch + 1) - 1:
            # Training confusion matrix
            train_cm = calculate_confusion_matrix(
                train_exit['predictions'], 
                train_exit['labels'], 
                num_classes
            )
            train_cm_fig = plot_confusion_matrix(train_cm, class_names=class_names, normalize=True)
            writer.add_figure(f'Confusion_Matrix_Exit{exit_num}/train', train_cm_fig, epoch)
            plt.close(train_cm_fig)
            
            # Validation confusion matrix
            val_cm = calculate_confusion_matrix(
                val_exit['predictions'], 
                val_exit['labels'], 
                num_classes
            )
            val_cm_fig = plot_confusion_matrix(val_cm, class_names=class_names, normalize=True)
            writer.add_figure(f'Confusion_Matrix_Exit{exit_num}/val', val_cm_fig, epoch)
            plt.close(val_cm_fig)
    
    # Log metrics for final exit
    train_final = train_results['final']
    val_final = val_results['final']
    
    writer.add_scalar('Loss_Final/train', train_final['loss'], epoch)
    writer.add_scalar('Loss_Final/val', val_final['loss'], epoch)
    writer.add_scalar('Accuracy_Final/train', train_final['accuracy'], epoch)
    writer.add_scalar('Accuracy_Final/val', val_final['accuracy'], epoch)
    
    # Final confusion matrices (every 5 epochs or last epoch)
    if (epoch + 1) % 5 == 0 or epoch == train_results.get('total_epochs', epoch + 1) - 1:
        # Training confusion matrix
        train_cm = calculate_confusion_matrix(
            train_final['predictions'], 
            train_final['labels'], 
            num_classes
        )
        train_cm_fig = plot_confusion_matrix(train_cm, class_names=class_names, normalize=True)
        writer.add_figure('Confusion_Matrix_Final/train', train_cm_fig, epoch)
        plt.close(train_cm_fig)
        
        # Validation confusion matrix
        val_cm = calculate_confusion_matrix(
            val_final['predictions'], 
            val_final['labels'], 
            num_classes
        )
        val_cm_fig = plot_confusion_matrix(val_cm, class_names=class_names, normalize=True)
        writer.add_figure('Confusion_Matrix_Final/val', val_cm_fig, epoch)
        plt.close(val_cm_fig)
    
    # Log combined total loss
    writer.add_scalar('Loss_Total/train', train_results['total_loss'], epoch)
    writer.add_scalar('Loss_Total/val', val_results['total_loss'], epoch)

# def log_early_exits_to_tensorboard(writer, epoch, train_results, val_results, 
#                                     num_classes, class_names=None, memory_info=None, input_memory_info=None):
#     """
#     Log early exit metrics to TensorBoard.
    
#     Creates separate charts for each exit (including final) to track:
#     - Training and validation loss
#     - Training and validation accuracy
#     - Confusion matrices (logged every 5 epochs or last epoch)
#     - Memory requirements (if provided, logged once at epoch 0)
#     - Input data memory (if provided, logged once at epoch 0)
    
#     Args:
#         writer: TensorBoard SummaryWriter
#         epoch: Current epoch number
#         train_results: Training results dict with 'exits' and 'final' keys
#         val_results: Validation results dict from validate_with_early_exits()
#         num_classes: Number of classes
#         class_names: List of class names (optional)
#         memory_info: Dict from get_early_exit_memory() with memory requirements (optional)
#         input_memory_info: Dict from get_input_memory() with input memory info (optional)
#     """
#     num_exits = len(train_results['exits'])
    
#     # Log input memory (logged once at epoch 0)
#     if input_memory_info and epoch == 0:
#         for info in input_memory_info['shape_info']:
#             # Create a friendly name for the input
#             input_name = f"{info['location']}_{info['modality']}"
#             writer.add_scalar(f'Input_Memory_KB/{input_name}', info['memory'], 0)
#         writer.add_scalar('Input_Memory_KB/total', input_memory_info['total_memory'], 0)
    
#     # Log metrics for each early exit
#     for exit_idx in range(num_exits):
#         exit_num = exit_idx + 1
#         train_exit = train_results['exits'][exit_idx]
#         val_exit = val_results['exits'][exit_idx]
        
#         # Loss
#         writer.add_scalar(f'Loss_Exit{exit_num}/train', train_exit['loss'], epoch)
#         writer.add_scalar(f'Loss_Exit{exit_num}/val', val_exit['loss'], epoch)
        
#         # Accuracy
#         writer.add_scalar(f'Accuracy_Exit{exit_num}/train', train_exit['accuracy'], epoch)
#         writer.add_scalar(f'Accuracy_Exit{exit_num}/val', val_exit['accuracy'], epoch)
        
#         # Memory (logged once at epoch 0)
#         if memory_info and memory_info['has_early_exits'] and epoch == 0:
#             exit_mem = memory_info['exits'][exit_idx]
#             writer.add_scalar(f'Memory_KB_Exit{exit_num}/parameters', 
#                             exit_mem['parameter_memory'], 0)
#             writer.add_scalar(f'Memory_KB_Exit{exit_num}/activations', 
#                             exit_mem['activation_memory'], 0)
#             writer.add_scalar(f'Memory_KB_Exit{exit_num}/total', 
#                             exit_mem['total_memory'], 0)
        
#         # Confusion matrices (every 5 epochs or last epoch)
#         if (epoch + 1) % 5 == 0 or epoch == train_results.get('total_epochs', epoch + 1) - 1:
#             # Training confusion matrix
#             train_cm = calculate_confusion_matrix(
#                 train_exit['predictions'], 
#                 train_exit['labels'], 
#                 num_classes
#             )
#             train_cm_fig = plot_confusion_matrix(train_cm, class_names=class_names, normalize=True)
#             writer.add_figure(f'Confusion_Matrix_Exit{exit_num}/train', train_cm_fig, epoch)
#             plt.close(train_cm_fig)
            
#             # Validation confusion matrix
#             val_cm = calculate_confusion_matrix(
#                 val_exit['predictions'], 
#                 val_exit['labels'], 
#                 num_classes
#             )
#             val_cm_fig = plot_confusion_matrix(val_cm, class_names=class_names, normalize=True)
#             writer.add_figure(f'Confusion_Matrix_Exit{exit_num}/val', val_cm_fig, epoch)
#             plt.close(val_cm_fig)
    
#     # Log metrics for final exit
#     train_final = train_results['final']
#     val_final = val_results['final']
    
#     writer.add_scalar('Loss_Final/train', train_final['loss'], epoch)
#     writer.add_scalar('Loss_Final/val', val_final['loss'], epoch)
#     writer.add_scalar('Accuracy_Final/train', train_final['accuracy'], epoch)
#     writer.add_scalar('Accuracy_Final/val', val_final['accuracy'], epoch)
    
#     # Memory for final exit (logged once at epoch 0)
#     if memory_info and epoch == 0:
#         final_mem = memory_info['final']
#         writer.add_scalar('Memory_KB_Final/parameters', final_mem['parameter_memory'], 0)
#         writer.add_scalar('Memory_KB_Final/activations', final_mem['activation_memory'], 0)
#         writer.add_scalar('Memory_KB_Final/total', final_mem['total_memory'], 0)
    
#     # Final confusion matrices (every 5 epochs or last epoch)
#     if (epoch + 1) % 5 == 0 or epoch == train_results.get('total_epochs', epoch + 1) - 1:
#         # Training confusion matrix
#         train_cm = calculate_confusion_matrix(
#             train_final['predictions'], 
#             train_final['labels'], 
#             num_classes
#         )
#         train_cm_fig = plot_confusion_matrix(train_cm, class_names=class_names, normalize=True)
#         writer.add_figure('Confusion_Matrix_Final/train', train_cm_fig, epoch)
#         plt.close(train_cm_fig)
        
#         # Validation confusion matrix
#         val_cm = calculate_confusion_matrix(
#             val_final['predictions'], 
#             val_final['labels'], 
#             num_classes
#         )
#         val_cm_fig = plot_confusion_matrix(val_cm, class_names=class_names, normalize=True)
#         writer.add_figure('Confusion_Matrix_Final/val', val_cm_fig, epoch)
#         plt.close(val_cm_fig)
    
#     # Log combined total loss
#     writer.add_scalar('Loss_Total/train', train_results['total_loss'], epoch)
#     writer.add_scalar('Loss_Total/val', val_results['total_loss'], epoch)


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
    
    writer.close()
    
    # Return model, history, and best checkpoint path
    best_checkpoint_path = str(models_dir / "best_model.pth")
    return model, train_history, best_checkpoint_path


def train_with_early_exits(model, train_loader, val_loader, config, experiment_dir,
                           loss_fn, optimizer, scheduler, num_epochs,
                           augmenter=None, apply_augmentation_fn=None):
    """
    Train model with early exits, tracking metrics for each exit separately.
    
    Args:
        model: PyTorch model with early exits
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        experiment_dir: Path to experiment directory
        loss_fn: Early exit loss function (REQUIRED)
        optimizer: Pre-configured optimizer (REQUIRED)
        scheduler: Pre-configured scheduler (REQUIRED)
        num_epochs: Number of training epochs (REQUIRED)
        augmenter: Data augmenter object (optional)
        apply_augmentation_fn: Function to apply augmentation (optional)
    
    Returns:
        model: Trained model
        train_history: Dictionary with training history for all exits
        best_checkpoint_path: Path to best model checkpoint
    """
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if loss_fn is None:
        raise ValueError("loss_fn is required - must be passed explicitly")
    
    # Setup directories
    experiment_path = Path(experiment_dir)
    logs_dir = experiment_path / "logs"
    models_dir = experiment_path / "models"
    tensorboard_dir = experiment_path / "tensorboard"
    
    # Setup logging
    log_file = logs_dir / "train_early_exits.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger('train_early_exits')
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    # Setup TensorBoard
    writer = SummaryWriter(str(tensorboard_dir))
    
    # Training parameters
    num_classes = config['vehicle_classification']['num_classes']
    class_names = config['vehicle_classification']['class_names']
    
    # Training history
    train_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    # Best model tracking (based on final exit accuracy)
    best_val_acc = 0.0
    best_epoch = 0
    
    logger.info("=" * 80)
    logger.info("Starting Training with Early Exits")
    logger.info(f"Device: {device}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info("=" * 80)
    
    # ========================================================================
    # Calculate Memory Requirements (before training)
    # ========================================================================
    logger.info("\nCalculating per-exit memory requirements...")
    from models.create_models import get_early_exit_memory, log_memory_info, get_input_memory
    
    # Grab one batch and apply augmentation to get actual input format
    sample_batch_data = next(iter(train_loader))
    if len(sample_batch_data) == 3:
        sample_data, sample_labels, _ = sample_batch_data
    else:
        sample_data, sample_labels = sample_batch_data[0], sample_batch_data[1]
    
    # Apply augmentation to get the actual data format that goes to the model
    if augmenter is not None and apply_augmentation_fn is not None:
        sample_data, _ = apply_augmentation_fn(augmenter, sample_data, sample_labels)
    
    # Calculate input memory
    input_memory_info = get_input_memory(sample_data, unit='KB')
    
    # Calculate memory for each exit
    memory_info = get_early_exit_memory(model, sample_data, unit='KB')
    
    # Log memory info to console/file (including input info)
    log_memory_info(memory_info, input_memory_info=input_memory_info, logger=logger)
    logger.info("")
    
    for epoch in range(num_epochs):
        # ====================================================================
        # Training Phase
        # ====================================================================
        model.train()
        
        # Initialize tracking for exits
        train_exit_stats = None
        train_final_stats = None
        train_total_loss = 0.0
        train_total_samples = 0
        
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
                for loc in data:
                    for mod in data[loc]:
                        data[loc][mod] = data[loc][mod].to(device)
            else:
                data = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            
            # Initialize stats on first batch
            if train_exit_stats is None:
                num_early_exits = len(outputs['exits'])
                train_exit_stats = [
                    {'loss': 0.0, 'correct': 0, 'total': 0, 'predictions': [], 'labels': []}
                    for _ in range(num_early_exits)
                ]
                train_final_stats = {'loss': 0.0, 'correct': 0, 'total': 0, 'predictions': [], 'labels': []}
            
            # Handle one-hot labels
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels
            
            # Calculate combined loss
            loss = loss_fn(outputs, loss_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track total loss
            train_total_loss += loss.item() * labels.size(0)
            train_total_samples += labels.size(0)
            
            # Calculate individual metrics for each exit (for tracking only, not used in backprop)
            ce_loss_fn = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                # Process each early exit
                for idx, exit_logits in enumerate(outputs['exits']):
                    exit_loss = ce_loss_fn(exit_logits, loss_labels)
                    predictions = torch.argmax(exit_logits, dim=1)
                    
                    train_exit_stats[idx]['loss'] += exit_loss.item() * labels.size(0)
                    train_exit_stats[idx]['correct'] += (predictions == loss_labels).sum().item()
                    train_exit_stats[idx]['total'] += labels.size(0)
                    train_exit_stats[idx]['predictions'].extend(predictions.cpu().numpy())
                    train_exit_stats[idx]['labels'].extend(loss_labels.cpu().numpy())
                
                # Process final exit
                final_logits = outputs['logits']
                final_loss = ce_loss_fn(final_logits, loss_labels)
                predictions = torch.argmax(final_logits, dim=1)
                
                train_final_stats['loss'] += final_loss.item() * labels.size(0)
                train_final_stats['correct'] += (predictions == loss_labels).sum().item()
                train_final_stats['total'] += labels.size(0)
                train_final_stats['predictions'].extend(predictions.cpu().numpy())
                train_final_stats['labels'].extend(loss_labels.cpu().numpy())
        
        # Calculate epoch training metrics
        train_results = {
            'total_loss': train_total_loss / train_total_samples,
            'exits': [],
            'final': {},
            'total_epochs': num_epochs
        }
        
        for stats in train_exit_stats:
            train_results['exits'].append({
                'loss': stats['loss'] / stats['total'],
                'accuracy': stats['correct'] / stats['total'],
                'predictions': stats['predictions'],
                'labels': stats['labels']
            })
        
        train_results['final'] = {
            'loss': train_final_stats['loss'] / train_final_stats['total'],
            'accuracy': train_final_stats['correct'] / train_final_stats['total'],
            'predictions': train_final_stats['predictions'],
            'labels': train_final_stats['labels']
        }
        
        train_history['train_loss'].append(train_results['total_loss'])
        
        # ====================================================================
        # Validation Phase
        # ====================================================================
        val_results = validate_with_early_exits(
            model, val_loader, loss_fn, device, num_classes, 
            augmenter=augmenter, apply_augmentation_fn=apply_augmentation_fn
        )
        train_history['val_loss'].append(val_results['total_loss'])
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        train_history['learning_rates'].append(current_lr)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # ====================================================================
        # Logging
        # ====================================================================
        logger.info(f"\nEpoch [{epoch+1}/{num_epochs}]")
        logger.info(f"  Total Train Loss: {train_results['total_loss']:.4f}")
        logger.info(f"  Total Val Loss: {val_results['total_loss']:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # Log each exit
        for exit_idx in range(len(train_results['exits'])):
            logger.info(f"  Exit {exit_idx+1} - Train Acc: {train_results['exits'][exit_idx]['accuracy']:.4f}, "
                       f"Val Acc: {val_results['exits'][exit_idx]['accuracy']:.4f}")
        
        logger.info(f"  Final Exit - Train Acc: {train_results['final']['accuracy']:.4f}, "
                   f"Val Acc: {val_results['final']['accuracy']:.4f}")
        
        # TensorBoard logging
        log_early_exits_to_tensorboard(writer, epoch, train_results, val_results, num_classes, class_names, memory_info=memory_info, input_memory_info=input_memory_info)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # ====================================================================
        # Save Checkpoints (based on final exit accuracy)
        # ====================================================================
        epoch_val_acc = val_results['final']['accuracy']
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            best_model_path = models_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': val_results['total_loss'],
                'config': config,
                'exit_results': val_results  # Save all exit results
            }, best_model_path)
            logger.info(f"  Best model saved! (Final Exit Val Acc: {best_val_acc:.4f})")
        
        # Save last epoch
        last_model_path = models_dir / "last_epoch.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': epoch_val_acc,
            'val_loss': val_results['total_loss'],
            'config': config,
            'exit_results': val_results
        }, last_model_path)
    
    # Final summary
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Best final exit validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"TensorBoard logs: {tensorboard_dir}")
    logger.info("=" * 80)
    
    writer.close()
    
    # Return model, history, and best checkpoint path
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

