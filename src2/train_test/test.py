"""
Testing Script for Distillation Models

This script tests trained models from the distillation pipeline with support for:
- Automatic config loading from experiment directory
- Early exit testing with per-exit metrics
- TensorBoard logging for visualization
- Simplified checkpoint loading

Usage:
    # Test with best checkpoint (default)
    python test.py --experiment_dir ../experiments/20260214_173826_only_audio_resnet_early_exit --gpu 0
    
    # Test with specific checkpoint
    python test.py --experiment_dir ../experiments/20260214_173826_only_audio_resnet_early_exit \\
                   --checkpoint_path ../experiments/.../models/checkpoint_epoch_10.pth --gpu 0
    
    # Test on CPU
    python test.py --experiment_dir ../experiments/20260214_173826_only_audio_resnet_early_exit --gpu -1

Output Structure:
    experiment_dir/
        └── test_YYYYMMDD_HHMMSS/
            ├── logs/
            │   └── test.log
            ├── tensorboard/  # If early exits detected
            │   └── events.out.tfevents...
            └── test_results.txt
"""

import sys
import logging
import torch
import yaml
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import parse_test_args
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_augmenter import create_augmenter, apply_augmentation
from models.create_models import create_single_modal_model
from train_test.loss import get_loss_function
from train_test.train_test_utils import (
    load_checkpoint, validate_with_early_exits, 
    log_early_exits_to_tensorboard
)
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
    # 7. Setup Normalization
    # ========================================================================
    logging.info("\nSetting up normalization...")
    train_loader, val_loader, test_loader = setup_normalization(
        train_loader, val_loader, test_loader, config
    )
    logging.info("Normalization setup complete")
    
    # ========================================================================
    # 8. Create Augmenter
    # ========================================================================
    logging.info("\nCreating augmenter...")
    augmenter = create_augmenter(config, augmentation_mode="fixed")
    logging.info("Augmenter created successfully")
    
    # ========================================================================
    # 9. Extract Model Information and Create Model
    # ========================================================================
    logging.info("\nExtracting model information...")
    
    # Get experiment name and config
    experiment_name = config.get('experiment_name')
    if not experiment_name:
        raise ValueError("experiment_name not found in config")
    
    experiment_config = config['distillation'][experiment_name]
    model_name = experiment_config['models'][0]  # First model (the one trained)
    model_config = config['models'][model_name]
    
    logging.info(f"  Model: {model_name}")
    logging.info(f"  Architecture: {model_config['model_type']}")
    logging.info(f"  Modality: {model_config.get('active_modality', 'N/A')}")
    
    # ========================================================================
    # 10. Detect Early Exits
    # ========================================================================
    has_early_exits = len(model_config.get('early_exits', [])) > 0
    num_early_exits = len(model_config.get('early_exits', []))
    
    if has_early_exits:
        logging.info(f"  Early exits detected: {num_early_exits}")
        logging.info(f"  Exit locations: {model_config['early_exits']}")
    else:
        logging.info("  No early exits (standard model)")
    
    # ========================================================================
    # 11. Create Model
    # ========================================================================
    logging.info("\nCreating model...")
    temp_config = {model_name: model_config, **config}
    model = create_single_modal_model(temp_config, model_name)
    logging.info("Model created successfully")
    
    # ========================================================================
    # 12. Load Checkpoint
    # ========================================================================
    logging.info("\nLoading checkpoint...")
    model = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()
    logging.info("Model loaded and set to eval mode")
    
    # ========================================================================
    # 12b. Calculate Memory Requirements
    # ========================================================================
    if has_early_exits:
        logging.info("\nCalculating per-exit memory requirements...")
        from models.create_models import get_early_exit_memory, log_memory_info, get_input_memory
        
        # Grab one batch and apply augmentation to get actual input format
        sample_batch_data = next(iter(test_loader))
        if len(sample_batch_data) == 3:
            sample_data, sample_labels, _ = sample_batch_data
        else:
            sample_data, sample_labels = sample_batch_data[0], sample_batch_data[1]
        
        # Apply augmentation to get the actual data format that goes to the model
        if augmenter is not None:
            sample_data, _ = apply_augmentation(augmenter, sample_data, sample_labels)
        
        # Calculate input memory
        input_memory_info = get_input_memory(sample_data, unit='KB')
        
        # Calculate memory for each exit
        memory_info = get_early_exit_memory(model, sample_data, unit='KB')
        
        # Log memory info to console/file (including input info)
        log_memory_info(memory_info, input_memory_info=input_memory_info)
        logging.info("")
    else:
        memory_info = None
        input_memory_info = None
    
    # ========================================================================
    # 13. Setup Loss Function
    # ========================================================================
    logging.info("\nSetting up loss function...")
    stage_config = experiment_config['stages'][0]  # Use first stage config
    loss_fn, loss_fn_name = get_loss_function(stage_config, has_early_exits=has_early_exits)
    logging.info(f"  Loss function: {loss_fn_name}")
    if has_early_exits and 'exit_weights' in stage_config:
        logging.info(f"  Exit weights: {stage_config['exit_weights']}")
    
    # ========================================================================
    # 14. Run Testing
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STARTING TESTING")
    logging.info("=" * 80)
    
    if has_early_exits:
        # Test model with early exits
        logging.info("\nTesting model with early exits...")
        
        test_results = validate_with_early_exits(
            model=model,
            val_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            num_classes=config['vehicle_classification']['num_classes'],
            augmenter=augmenter,
            apply_augmentation_fn=apply_augmentation
        )
        
        # Log results
        logging.info("\n" + "-" * 80)
        logging.info("TEST RESULTS (Early Exit Model)")
        logging.info("-" * 80)
        logging.info(f"Total Loss: {test_results['total_loss']:.4f}")
        logging.info("\nPer-Exit Performance:")
        for i, exit_metrics in enumerate(test_results['exits']):
            logging.info(f"  Exit {i+1}: Loss={exit_metrics['loss']:.4f}, Accuracy={exit_metrics['accuracy']:.4f}")
        logging.info(f"  Final:  Loss={test_results['final']['loss']:.4f}, Accuracy={test_results['final']['accuracy']:.4f}")
        logging.info("-" * 80)
        
        # ========================================================================
        # 15. TensorBoard Logging (Early Exits)
        # ========================================================================
        logging.info("\nLogging to TensorBoard...")
        tensorboard_dir = test_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(str(tensorboard_dir))
        
        # Log test metrics for each exit
        # Note: Using test_results for both train and val since we only have test data
        log_early_exits_to_tensorboard(
            writer=writer,
            epoch=0,  # Use 0 for test (not training)
            train_results=test_results,  # Use test results
            val_results=test_results,     # Use test results
            num_classes=config['vehicle_classification']['num_classes'],
            memory_info=memory_info,
            input_memory_info=input_memory_info,
            test=True
        )
        writer.close()
        logging.info(f"  TensorBoard logs saved to: {tensorboard_dir}")
        
    else:
        # Test standard model (no early exits)
        logging.info("\nTesting standard model (no early exits)...")
        
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
        
        test_results = {
            'loss': test_loss,
            'accuracy': test_acc,
            'predictions': all_preds,
            'labels': all_labels
        }
        
        # Log results
        logging.info("\n" + "-" * 80)
        logging.info("TEST RESULTS (Standard Model)")
        logging.info("-" * 80)
        logging.info(f"Loss: {test_results['loss']:.4f}")
        logging.info(f"Accuracy: {test_results['accuracy']:.4f}")
        logging.info("-" * 80)
    
    # ========================================================================
    # 16. Save Results to File
    # ========================================================================
    logging.info("\nSaving results to file...")
    results_file = test_dir / "test_results.txt"
    
    # Calculate total test samples
    if has_early_exits:
        test_samples = len(test_results['exits'][0]['labels'])
    else:
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
        
        if has_early_exits:
            f.write(f"Total Loss: {test_results['total_loss']:.4f}\n")
            f.write("\nEarly Exits Performance:\n")
            f.write("-" * 40 + "\n")
            for i, exit_metrics in enumerate(test_results['exits']):
                f.write(f"Exit {i+1}:\n")
                f.write(f"  Loss: {exit_metrics['loss']:.4f}\n")
                f.write(f"  Accuracy: {exit_metrics['accuracy']:.4f}\n")
            f.write(f"\nFinal Exit:\n")
            f.write(f"  Loss: {test_results['final']['loss']:.4f}\n")
            f.write(f"  Accuracy: {test_results['final']['accuracy']:.4f}\n")
        else:
            f.write(f"Loss: {test_results['loss']:.4f}\n")
            f.write(f"Accuracy: {test_results['accuracy']:.4f}\n")
        
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
    if has_early_exits:
        logging.info(f"  - TensorBoard logs: {tensorboard_dir}")
        logging.info(f"\nView TensorBoard: tensorboard --logdir={tensorboard_dir}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
