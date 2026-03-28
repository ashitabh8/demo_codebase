"""
SSL Pretraining Script

Standalone entry point for self-supervised contrastive pretraining with NT-Xent loss.
Does NOT touch train.py. No normalization, no val/test splits, no class subsetting.

Usage:
    python pretrain.py --experiment_name pretrain_audio_resnet --yaml_path ../data/Parkland.yaml --gpu 0
"""

import sys
import logging
import yaml
from pathlib import Path

# Add src2 to path for imports
src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from dataset_utils.parse_args_utils import get_config
from dataset_utils.MultiModalDataLoader import create_pretrain_dataloader
from data_augmenter import create_augmenter, apply_augmentation
from models.create_models import create_single_modal_model
from train_test.loss import get_loss_function
from train_test.train_test_utils import (
    setup_experiment_dir,
    setup_train_file_logging,
    setup_optimizer,
    setup_scheduler,
    validate_pretrain_config,
    pretrain,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main SSL pretraining function."""

    # ========================================================================
    # 1. Load Configuration
    # ========================================================================
    logging.info("=" * 80)
    logging.info("SSL PRETRAINING SCRIPT")
    logging.info("=" * 80)

    config = get_config()
    logging.info("Configuration loaded successfully")

    # ========================================================================
    # 2. Validate Pretrain Config
    # ========================================================================
    validate_pretrain_config(config)
    logging.info("Pretrain config validated successfully")

    # ========================================================================
    # 3. Extract Experiment Config (inline, no validate_and_resolve_training_config)
    # ========================================================================
    experiment_name = config["experiment_name"]
    experiment_config = config["experiments"][experiment_name]
    model_name = experiment_config["model"]
    training_config_name = experiment_config["training"]
    training_config = config["training_configs"][training_config_name]

    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Training config: {training_config_name}")

    # ========================================================================
    # 4. Create Pretrain Dataloader (no val/test splits)
    # ========================================================================
    logging.info("\nCreating pretrain dataloader...")
    train_loader = create_pretrain_dataloader(config=config)
    logging.info("Pretrain dataloader created successfully")

    # ========================================================================
    # 5. Create Augmenter
    # ========================================================================
    logging.info("\nCreating augmenter...")
    augmenter = create_augmenter(
        config, augmentation_mode="fixed", experiment_config=experiment_config
    )
    logging.info("Augmenter created successfully")

    # ========================================================================
    # 6. Setup Experiment Directory (with "pretrain_" prefix)
    # ========================================================================
    logging.info("\nSetting up experiment directory...")
    pretrain_experiment_name = f"pretrain_{experiment_name}"
    experiment_dir, tensorboard_dir = setup_experiment_dir(
        config, experiment_name=pretrain_experiment_name
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
    # 7. Setup File Logging
    # ========================================================================
    setup_train_file_logging(experiment_dir, argv=sys.argv)

    # ========================================================================
    # 8. Pretraining
    # ========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("STARTING SSL PRETRAINING")
    logging.info("=" * 80 + "\n")

    try:
        # ====================================================================
        # Create Model
        # ====================================================================
        config["models"][model_name]["pretrain_mode"] = True
        model = create_single_modal_model(config, model_name)
        logging.info(f"Model created: {model_name}")

        # ====================================================================
        # Setup Loss Function
        # ====================================================================
        logging.info("\nSetting up loss function...")
        loss_fn, _ = get_loss_function(training_config)

        # ====================================================================
        # Setup Optimizer and Scheduler
        # ====================================================================
        logging.info("\nSetting up optimizer and scheduler...")
        optimizer = setup_optimizer(model, config, training_config=training_config)
        scheduler = setup_scheduler(optimizer, config, training_config)

        # ====================================================================
        # Pretrain
        # ====================================================================
        logging.info("\nStarting SSL pretraining...")
        model, best_loss, best_checkpoint_path = pretrain(
            model=model,
            train_loader=train_loader,
            config=config,
            experiment_dir=experiment_dir,
            loss_fn=loss_fn,
            augmenter=augmenter,
            apply_augmentation_fn=apply_augmentation,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=training_config["epochs"],
            model_name=model_name,
        )

        # ====================================================================
        # Update config with checkpoint path and save
        # ====================================================================
        config["models"][model_name]["checkpoint_path"] = best_checkpoint_path

        config_path = Path(experiment_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # ====================================================================
        # Pretraining Complete
        # ====================================================================
        logging.info("\n" + "=" * 80)
        logging.info("SSL PRETRAINING COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)
        logging.info(f"\nExperiment directory: {experiment_dir}")
        logging.info(f"Best checkpoint: {best_checkpoint_path}")
        logging.info(f"Best loss: {best_loss:.6f}")
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
        logging.warning("Pretraining interrupted by user")
        logging.info("=" * 80)
        logging.info(f"Experiment directory: {experiment_dir}")
        sys.exit(0)

    except Exception as e:
        logging.error("\n" + "=" * 80)
        logging.error("ERROR DURING PRETRAINING")
        logging.error("=" * 80)
        logging.error(f"Error: {e}")

        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
