"""
Testing Script for Distillation Models

This script tests trained models from the distillation pipeline with support for:
- Automatic config loading from experiment directory
- Simplified checkpoint loading

Usage:
    # Test with best checkpoint (default)
    python test.py --experiment_dir ../experiments/20260214_173826_only_audio_resnet --gpu 0

    # Test with specific checkpoint
    python test.py --experiment_dir ../experiments/20260214_173826_only_audio_resnet \\
                   --checkpoint_path ../experiments/.../models/checkpoint_epoch_10.pth --gpu 0

    # Test on CPU
    python test.py --experiment_dir ../experiments/20260214_173826_only_audio_resnet --gpu -1

Output Structure:
    experiment_dir/
        └── test_YYYYMMDD_HHMMSS/
            ├── logs/
            │   └── test.log
            └── test_results.txt
"""

import sys
import logging
import torch
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import parse_test_args
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_augmenter import create_augmenter, apply_augmentation
from models.create_models import create_single_modal_model
from train_test.loss import get_loss_function
from train_test.train_test_utils import load_checkpoint
from train_test.normalize import setup_normalization

# Configure logging (console only initially, file handler added later)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def main():
    """Main testing function."""
    
    # ========================================================================
    # 1. Parse Arguments
    # ========================================================================
    logging.info("=" * 80)
    logging.info("TESTING SCRIPT - DISTILLATION MODELS")
    logging.info("=" * 80)
    
    args = parse_test_args()
    
    # ========================================================================
    # 2. Load Configuration from Experiment Directory
    # ========================================================================
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logging.info("\nLoading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logging.info(f"  Config loaded from: {config_path}")
    logging.info(f"  Experiment name: {config.get('experiment_name')}")
    logging.info(f"  Dataset config: {config.get('yaml_path')}")
    
    # ========================================================================
    # 3. Determine Checkpoint Path
    # ========================================================================
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        logging.info(f"\nUsing specified checkpoint: {args.checkpoint_path}")
    else:
        checkpoint_path = experiment_dir / "models" / "best_model.pth"
        logging.info(f"\nUsing default best checkpoint: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # ========================================================================
    # 4. Setup Device
    # ========================================================================
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logging.info(f"Device: GPU {args.gpu}")
    else:
        device = torch.device('cpu')
        logging.info("Device: CPU")
    
    # Update config with device
    config['device'] = str(device)
    
    # ========================================================================
    # 5. Create Test Directory
    # ========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = experiment_dir / f"test_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    logs_dir = test_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Setup file logging
    log_file = logs_dir / "test.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"\nTest directory: {test_dir}")
    logging.info(f"Log file: {log_file}")
    
    # ========================================================================
    # 6. Create Dataloaders (test set only)
    # ========================================================================
    logging.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    logging.info(f"  Test batches: {len(test_loader)}")

    # ========================================================================
    # 7. Extract Model Information (needed before normalization policy)
    # ========================================================================
    logging.info("\nExtracting model information...")

    experiment_name = config.get('experiment_name')
    if not experiment_name:
        raise ValueError("experiment_name not found in config")

    # The repo supports both "distillation"-style experiments and the newer
    # "experiments" + "training_configs" vanilla training flow.
    if "distillation" in config and experiment_name in config["distillation"]:
        experiment_config = config["distillation"][experiment_name]
        model_name = experiment_config["models"][0]  # First model (the one trained)
        loss_source_config = experiment_config["stages"][0]
    else:
        experiment_config = config["experiments"][experiment_name]
        model_name = experiment_config["model"]
        training_config_name = experiment_config["training"]
        loss_source_config = config["training_configs"][training_config_name]

    model_config = config["models"][model_name]

    logging.info(f"  Model: {model_name}")
    logging.info(f"  Architecture: {model_config['model_type']}")
    logging.info(f"  Modality: {model_config.get('active_modality', 'N/A')}")

    # ========================================================================
    # 8. Setup Normalization (mirror finetune path)
    # ========================================================================
    skip_normalization = False
    if "type" in loss_source_config and loss_source_config["type"] == "finetune":
        # finetune.py intentionally skips setup_normalization; keep test aligned.
        skip_normalization = True

    if skip_normalization:
        logging.info("\nSkipping normalization setup to match finetune.py behavior")
    else:
        logging.info("\nSetting up normalization...")
        train_loader, val_loader, test_loader = setup_normalization(
            train_loader, val_loader, test_loader, config
        )
        logging.info("Normalization setup complete")

    # ========================================================================
    # 9. Create Augmenter (disabled for deterministic evaluation)
    # ========================================================================
    logging.info(f"\nCreating augmenter (mode={args.augmentation_mode})...")
    augmenter = create_augmenter(
        config, augmentation_mode=args.augmentation_mode, experiment_config=experiment_config
    )
    logging.info(
        f"Augmenter created successfully (augmentation_mode={args.augmentation_mode})"
    )

    # ========================================================================
    # 10. Create Model
    # ========================================================================
    logging.info("\nCreating model...")
    config["models"][model_name]["pretrain_mode"] = False
    model = create_single_modal_model(config, model_name)
    logging.info("Model created successfully")
    
    # ========================================================================
    # 11. Load Checkpoint
    # ========================================================================
    logging.info("\nLoading checkpoint...")
    model = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()
    logging.info("Model loaded and set to eval mode")
    
    # ========================================================================
    # 11b. Calculate Memory Requirements
    # ========================================================================
    memory_info = None
    input_memory_info = None
    
    # ========================================================================
    # 12. Setup Loss Function
    # ========================================================================
    logging.info("\nSetting up loss function...")
    loss_fn, loss_fn_name = get_loss_function(loss_source_config)
    logging.info(f"  Loss function: {loss_fn_name}")
    
    # ========================================================================
    # 13. Run Testing
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STARTING TESTING")
    logging.info("=" * 80)
    
    # Test standard model
    logging.info("\nTesting standard model...")

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
            if augmenter is not None:
                data, labels = apply_augmentation(augmenter, data, labels)

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

            # Extract logits if dict output
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            # Handle one-hot labels
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels

            loss = loss_fn(outputs, loss_labels)

            test_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(logits, dim=1)
            test_correct += (predictions == loss_labels).sum().item()
            test_total += labels.size(0)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(loss_labels.cpu().numpy())

    test_loss /= test_total
    test_acc = test_correct / test_total

    # Build confusion matrix and per-class accuracy for better diagnostics.
    if "task_name" in config and config["task_name"] in config and "class_names" in config[config["task_name"]]:
        class_names = config[config["task_name"]]["class_names"]
    else:
        class_names = []

    inferred_num_classes = len(class_names)
    if inferred_num_classes == 0:
        inferred_num_classes = int(max(max(all_labels), max(all_preds))) + 1
        class_names = [f"class_{i}" for i in range(inferred_num_classes)]

    cm = np.zeros((inferred_num_classes, inferred_num_classes), dtype=np.int64)
    for true_label, pred_label in zip(all_labels, all_preds):
        cm[int(true_label), int(pred_label)] += 1

    per_class_accuracy = []
    for class_idx in range(inferred_num_classes):
        class_total = int(cm[class_idx].sum())
        class_correct = int(cm[class_idx, class_idx])
        class_acc = (class_correct / class_total) if class_total > 0 else 0.0
        per_class_accuracy.append(class_acc)

    test_results = {
        'loss': test_loss,
        'accuracy': test_acc,
        'predictions': all_preds,
        'labels': all_labels,
        'class_names': class_names,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_accuracy
    }

    # Log results
    logging.info("\n" + "-" * 80)
    logging.info("TEST RESULTS (Standard Model)")
    logging.info("-" * 80)
    logging.info(f"Loss: {test_results['loss']:.4f}")
    logging.info(f"Accuracy: {test_results['accuracy']:.4f}")
    logging.info("Per-class accuracy:")
    for class_idx, class_name in enumerate(test_results['class_names']):
        logging.info(f"  {class_name}: {test_results['per_class_accuracy'][class_idx]:.4f}")
    logging.info("Confusion matrix (rows=true, cols=pred):")
    logging.info(f"\n{test_results['confusion_matrix']}")
    logging.info("-" * 80)
    
    # ========================================================================
    # 16. Save Results to File
    # ========================================================================
    logging.info("\nSaving results to file...")
    results_file = test_dir / "test_results.txt"
    
    # Calculate total test samples
    test_samples = test_total
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEST RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Test samples: {test_samples}\n")
        f.write("\n")
        
        f.write(f"Loss: {test_results['loss']:.4f}\n")
        f.write(f"Accuracy: {test_results['accuracy']:.4f}\n")
        f.write("\nPer-class accuracy:\n")
        for class_idx, class_name in enumerate(test_results['class_names']):
            f.write(f"  {class_name}: {test_results['per_class_accuracy'][class_idx]:.4f}\n")
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(f"{test_results['confusion_matrix']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logging.info(f"  Results saved to: {results_file}")
    
    # ========================================================================
    # 17. Final Summary
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("TESTING COMPLETED SUCCESSFULLY")
    logging.info("=" * 80)
    logging.info(f"Test directory: {test_dir}")
    logging.info(f"  - Log file: {log_file}")
    logging.info(f"  - Results file: {results_file}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
