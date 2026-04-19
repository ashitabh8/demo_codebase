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
from train_test.train_test_utils import (
    load_checkpoint,
    validate,
    validate_multilabel,
    validate_vanilla_supervised_contrastive,
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
    # 8. Extract Model Information (before augmenter: needs experiment_config)
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

    simple_model_training = False
    if "simple_model_training" in experiment_config:
        simple_model_training = bool(experiment_config["simple_model_training"])

    task_name = config["task_name"]
    num_classes = config[task_name]["num_classes"]

    logging.info(f"  Model: {model_name}")
    logging.info(f"  Architecture: {model_config['model_type']}")
    logging.info(f"  Modality: {model_config.get('active_modality', 'N/A')}")

    # ========================================================================
    # 9. Create Augmenter (uses experiment_config.fixed_augmenters only)
    # ========================================================================
    logging.info("\nCreating augmenter...")
    augmenter = create_augmenter(config, augmentation_mode="fixed", experiment_config=experiment_config)
    logging.info("Augmenter created successfully")

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
    # 12. Setup Loss Function
    # ========================================================================
    logging.info("\nSetting up loss function...")
    loss_fn, loss_fn_name = get_loss_function(loss_source_config)
    logging.info(f"  Loss function: {loss_fn_name}")
    
    # ========================================================================
    # 13. Run Testing (match train.py / finetune.py validation)
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STARTING TESTING")
    logging.info("=" * 80)

    logging.info("\nEvaluating on test set (same helper as training validation)...")
    if loss_fn_name == "ce_supcon":
        logging.info(
            "  validate_vanilla_supervised_contrastive: clean inputs "
            "(augmentation_mode='no' during forward, same as val in SupCon training)"
        )
        test_results = validate_vanilla_supervised_contrastive(
            model,
            test_loader,
            loss_fn,
            device,
            augmenter=augmenter,
            apply_augmentation_fn=apply_augmentation,
            num_classes=num_classes,
            collect_embeddings=False,
            simple_model_training=simple_model_training,
            model_name=model_name,
            config=config,
        )
    elif loss_fn_name == "bce_multilabel":
        logging.info(
            "  validate_multilabel: same augmenter usage as train() validation"
        )
        test_results = validate_multilabel(
            model,
            test_loader,
            loss_fn,
            device,
            loss_source_config,
            augmenter=augmenter,
            apply_augmentation_fn=apply_augmentation,
            simple_model_training=simple_model_training,
            model_name=model_name,
            config=config,
        )
    else:
        logging.info(
            "  validate: augmenter applied on each batch (same as val in train())"
        )
        test_results = validate(
            model,
            test_loader,
            loss_fn,
            device,
            augmenter=augmenter,
            apply_augmentation_fn=apply_augmentation,
            num_classes=num_classes,
            simple_model_training=simple_model_training,
            model_name=model_name,
            config=config,
        )

    # Log results
    logging.info("\n" + "-" * 80)
    logging.info("TEST RESULTS")
    logging.info("-" * 80)
    logging.info(f"Loss: {test_results['loss']:.4f}")
    if loss_fn_name == "bce_multilabel":
        logging.info(f"mAP: {test_results['mAP']:.4f}")
    else:
        logging.info(f"Accuracy: {test_results['accuracy']:.4f}")
    logging.info("-" * 80)
    
    # ========================================================================
    # 16. Save Results to File
    # ========================================================================
    logging.info("\nSaving results to file...")
    results_file = test_dir / "test_results.txt"

    if loss_fn_name == "bce_multilabel":
        test_samples = int(test_results["raw_probs"].shape[0])
    else:
        test_samples = len(test_results["predictions"])

    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEST RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Loss function: {loss_fn_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Test samples: {test_samples}\n")
        f.write("\n")

        f.write(f"Loss: {test_results['loss']:.4f}\n")
        if loss_fn_name == "bce_multilabel":
            f.write(f"mAP: {test_results['mAP']:.4f}\n")
        else:
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
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
