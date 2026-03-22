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
from models.create_models import create_single_modal_model, get_total_memory
from train_test.loss import get_loss_function
from train_test.train_test_utils import (
    setup_experiment_dir,
    setup_train_file_logging,
    train,
    train_vanilla_supervised_contrastive,
    setup_optimizer,
    setup_scheduler,
    apply_class_subset,
    validate_and_resolve_training_config,
)
from train_test.normalize import setup_normalization

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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

    (
        experiment_name,
        experiment_config,
        model_name,
        training_config_name,
        training_config,
        train_type,
        stage_epochs,
        loss_name,
    ) = validate_and_resolve_training_config(config)
    # breakpoint()

    apply_class_subset(config)

    # ========================================================================
    # 2. Create Dataloaders
    # ========================================================================
    train_loader, val_loader, test_loader = create_dataloaders(config=config)

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
    augmenter = create_augmenter(
        config, augmentation_mode="fixed", experiment_config=experiment_config
    )
    logging.info("Augmenter created successfully")

    # ========================================================================
    # 4. Setup Experiment Directory
    # ========================================================================
    logging.info("\nSetting up experiment directory...")
    experiment_dir, tensorboard_dir = setup_experiment_dir(
        config, experiment_name=experiment_name
    )
    tb_run_dir = Path(tensorboard_dir).resolve()
    logging.info(
        'TensorBoard (this run): tensorboard --logdir="%s" --port=6006',
        tb_run_dir,
    )
    logging.info(
        "To compare multiple runs, point --logdir at the parent experiments directory instead."
    )

    # ========================================================================
    # 5. Setup File Logging
    # ========================================================================
    setup_train_file_logging(experiment_dir, argv=sys.argv)

    # ========================================================================
    # 6. Training
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STARTING TRAINING")
    logging.info("=" * 80 + "\n")

    try:
        # ====================================================================
        # Create Student Model
        # ====================================================================
        # Factory pulls the model config from `config["models"][model_name]`
        model = create_single_modal_model(config, model_name)
        logging.info(f"Student model created: {model_name}")

        # Memory profile (B=1; divide parameter_memory by 4 for INT8 estimate)
        _location = config["location_names"][0]
        _modality = config["models"][model_name]["active_modality"]
        _in_ch = config["loc_mod_in_freq_channels"][_location][_modality]
        _n_segs = config.get("num_segments", 10)
        _spec_len = config["loc_mod_spectrum_len"][_location][_modality]
        _dummy = {_location: {_modality: torch.randn(1, _in_ch, _n_segs, _spec_len)}}
        _mem = get_total_memory(model, _dummy, unit="MB")
        logging.info("Memory profile (float32 / INT8 weight-only estimate):")
        logging.info(
            "  Parameters : %.2f MB  (INT8 ≈ %.2f MB)",
            _mem["parameter_memory"], _mem["parameter_memory"] / 4,
        )
        logging.info("  Peak activation (B=1): %.3f MB", _mem["activation_memory"])
        del _dummy, _mem

        # ====================================================================
        # Setup Loss Function
        # ====================================================================
        logging.info("\nSetting up loss function...")
        loss_fn, _ = get_loss_function(training_config)

        # ====================================================================
        # Setup Optimizer and Scheduler
        # ====================================================================
        logging.info("\nSetting up optimizer and scheduler...")
        optimizer = setup_optimizer(
            model, config, training_config=training_config
        )
        scheduler = setup_scheduler(optimizer, config, training_config)

        # ====================================================================
        # Train
        # ====================================================================
        if train_type == "vanilla_supervised":
            logging.info("\nStarting vanilla supervised training...")
            model, _, best_checkpoint_path = train(
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
                num_epochs=stage_epochs,
            )
        elif train_type == "vanilla_supervised_contrastive":
            logging.info(
                "\nStarting vanilla supervised contrastive training..."
            )
            model, _, best_checkpoint_path = (
                train_vanilla_supervised_contrastive(
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
                    num_epochs=stage_epochs,
                )
            )
        elif train_type == "distillation":
            raise NotImplementedError(
                "Distillation training loop not yet implemented. "
                "Wire up kd_loss and a distillation train function first."
            )
        else:
            raise ValueError(f"Unknown training type: {train_type}")

        # ====================================================================
        # Update config with checkpoint path and save
        # ====================================================================
        config["models"][model_name]["checkpoint_path"] = best_checkpoint_path

        config_path = Path(experiment_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # ====================================================================
        # Training Complete
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("TRAINING COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)
        logging.info(f"\nExperiment directory: {experiment_dir}")
        logging.info(f"Best checkpoint: {best_checkpoint_path}")
        logging.info(
            'TensorBoard (this run): tensorboard --logdir="%s" --port=6006',
            tb_run_dir,
        )
        logging.info(
            "To compare multiple runs, point --logdir at the parent experiments directory instead."
        )
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
