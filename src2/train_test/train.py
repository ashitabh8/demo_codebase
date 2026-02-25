"""
Training Script

This script orchestrates the training process:
1. Parse configuration and command-line arguments
2. Create dataloaders
3. Create model and augmenter
4. Setup experiment directory and logging
5. Initialize optimizer and scheduler
6. Train the model with checkpointing and logging
"""

import sys
import logging
import yaml
import torch
from pathlib import Path

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import get_config
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_augmenter import create_augmenter, apply_augmentation
from models.create_models import create_single_modal_model
from train_test.loss import get_loss_function
from train_test.train_test_utils import (
    setup_experiment_dir, train, train_with_early_exits, setup_optimizer, setup_scheduler, load_checkpoint
)
from train_test.normalize import setup_normalization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    """Main training function."""
    
    # ========================================================================
    # 1. Load Configuration
    # ========================================================================
    logging.info("=" * 80)
    logging.info("TRAINING SCRIPT")
    logging.info("=" * 80)
    
    config = get_config()
    logging.info("Configuration loaded successfully")
    
    # Get experiment name from config
    experiment_name = config.get('experiment_name')
    if experiment_name is None:
        raise ValueError("experiment_name not found in config. Please provide --experiment_name argument.")
    
    logging.info(f"  Experiment: {experiment_name}")
    logging.info(f"  Dataset: {config.get('yaml_path', 'Unknown')}")
    logging.info(f"  Device: {config.get('device', 'cpu')}")
    
    # Load distillation experiment configuration
    if 'distillation' not in config or not config['distillation'].get('enabled', False):
        raise ValueError("Distillation not enabled in config. Set distillation.enabled: true")
    
    if experiment_name not in config['distillation']:
        raise ValueError(f"Experiment '{experiment_name}' not found in distillation config. "
                        f"Available experiments: {list(config['distillation'].keys())}")
    
    experiment_config = config['distillation'][experiment_name]
    models_list = experiment_config['models']
    stages = experiment_config['stages']
    
    logging.info(f"  Models in cascade: {models_list}")
    logging.info(f"  Number of stages: {len(stages)}")
    
    # ========================================================================
    # 2. Create Dataloaders
    # ========================================================================
    logging.info("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    logging.info("  Train batches: {}".format(len(train_loader)))
    logging.info("  Val batches: {}".format(len(val_loader)))
    logging.info("  Test batches: {}".format(len(test_loader)))
    
    # ========================================================================
    # 2b. Setup Normalization
    # ========================================================================
    logging.info("\nSetting up normalization...")
    train_loader, val_loader, test_loader = setup_normalization(
        train_loader, val_loader, test_loader, config
    )
    logging.info("Normalization setup complete")
    
    # ========================================================================
    # 3. Create Augmenter
    # ========================================================================
    logging.info("\nCreating augmenter...")
    augmenter = create_augmenter(config, augmentation_mode="fixed")
    logging.info("Augmenter created successfully")
    
    # ========================================================================
    # 4. Setup Experiment Directory
    # ========================================================================
    logging.info("\nSetting up experiment directory...")
    experiment_dir, tensorboard_dir = setup_experiment_dir(config, experiment_name=experiment_name)
    
    # ========================================================================
    # 5. Setup File Logging
    # ========================================================================
    from pathlib import Path
    logs_dir = Path(experiment_dir) / "logs"
    log_file = logs_dir / "train.log"
    
    # Add file handler to root logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"Logging to file: {log_file}")
    
    # Log the command line
    import sys
    command_line = " ".join(sys.argv)
    logging.info("")
    logging.info("Command line used to run this script:")
    logging.info(f"  {command_line}")
    
    # ========================================================================
    # 6. Multi-Stage Training Loop
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STARTING MULTI-STAGE TRAINING")
    logging.info("=" * 80 + "\n")
    
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    stage_checkpoints = []  # Track checkpoints for each stage
    
    try:
        for stage_idx, stage_config in enumerate(stages):
            logging.info("\n" + "=" * 80)
            logging.info(f"STAGE {stage_idx + 1}/{len(stages)}")
            logging.info("=" * 80)
            
            # Get model name for this stage
            model_idx = stage_config['teacher_idx']
            model_name = models_list[model_idx]
            train_type = stage_config['train_type']
            stage_epochs = stage_config['epochs']
            loss_name = stage_config['loss_name']
            
            logging.info(f"  Model: {model_name}")
            logging.info(f"  Training type: {train_type}")
            logging.info(f"  Epochs: {stage_epochs}")
            logging.info(f"  Loss: {loss_name}")
            
            # ================================================================
            # Create Model
            # ================================================================
            logging.info("\nCreating model...")
            # Create a temporary config with the model definition at the top level
            # since create_single_modal_model expects config[model_name]
            temp_config = config.copy()
            temp_config[model_name] = config['models'][model_name]
            model = create_single_modal_model(temp_config, model_name)
            logging.info(f"Model created: {model_name}")
            
            # ================================================================
            # Load checkpoint if not first stage
            # ================================================================
            if stage_idx > 0:
                prev_checkpoint = stage_checkpoints[stage_idx - 1]
                logging.info(f"\nLoading teacher checkpoint from previous stage:")
                logging.info(f"  {prev_checkpoint}")
                model = load_checkpoint(model, prev_checkpoint, device)
            
            # ================================================================
            # Setup Loss Function
            # ================================================================
            # Detect if model has early exits
            model_config = config['models'][model_name]
            has_early_exits = len(model_config.get('early_exits', [])) > 0
            
            logging.info("\nSetting up loss function...")
            loss_fn, loss_fn_name = get_loss_function(stage_config, has_early_exits=has_early_exits)
            
            # ================================================================
            # Setup Optimizer and Scheduler
            # ================================================================
            logging.info("\nSetting up optimizer and scheduler...")
            optimizer = setup_optimizer(model, config, experiment_config=experiment_config)
            scheduler = setup_scheduler(optimizer, config, experiment_config=experiment_config, stage_config=stage_config)
            
            # ================================================================
            # Train based on train_type
            # ================================================================
            if train_type == 'vanilla_supervised':
                if has_early_exits:
                    logging.info(f"\nStarting vanilla supervised training with early exits...")
                    logging.info(f"  Number of early exits: {len(model_config['early_exits'])}")
                    model, train_history, best_checkpoint_path = train_with_early_exits(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        config=config,
                        experiment_dir=experiment_dir,
                        loss_fn=loss_fn,
                        augmenter=augmenter,
                        apply_augmentation_fn=apply_augmentation,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        num_epochs=stage_epochs,
                    )
                else:
                    logging.info("\nStarting vanilla supervised training...")
                    model, train_history, best_checkpoint_path = train(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        config=config,
                        experiment_dir=experiment_dir,
                        loss_fn=loss_fn,
                        val_fn=None,
                        augmenter=augmenter,
                        apply_augmentation_fn=apply_augmentation,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        num_epochs=stage_epochs
                    )
            else:
                raise ValueError(f"Unknown training type: {train_type}")
            
            # ================================================================
            # Store checkpoint for next stage
            # ================================================================
            stage_checkpoints.append(best_checkpoint_path)
            logging.info(f"\nStage {stage_idx + 1} complete!")
            logging.info(f"  Best checkpoint: {best_checkpoint_path}")
            
            # Update config with checkpoint path
            if 'models' not in config:
                config['models'] = {}
            if model_name not in config['models']:
                config['models'][model_name] = {}
            config['models'][model_name]['checkpoint_path'] = best_checkpoint_path
            
            # Save updated config
            config_path = Path(experiment_dir) / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        # ====================================================================
        # Training Complete
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("ALL STAGES COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)
        logging.info(f"\nExperiment directory: {experiment_dir}")
        logging.info(f"Checkpoints saved:")
        for i, ckpt in enumerate(stage_checkpoints):
            logging.info(f"  Stage {i+1}: {ckpt}")
        logging.info("=" * 80)
        
    except KeyboardInterrupt:
        logging.info("\n" + "=" * 80)
        logging.warning("Training interrupted by user")
        logging.info("=" * 80)
        logging.info(f"Experiment directory: {experiment_dir}")
        sys.exit(0)
    
    except Exception as e:
        logging.error("\n" + "=" * 80)
        logging.error("ERROR DURING TRAINING")
        logging.error("=" * 80)
        logging.error(f"Error: {e}")
        
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

