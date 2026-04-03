"""
Supervised fine-tuning from a pretrained (SSL) checkpoint.

Loads backbone weights from a pretrain .pth (projection head keys skipped),
optionally freezes early layers, then runs the standard CE training loop.

Usage:
    python finetune.py --experiment_name finetune_audio_resnet_from_pretrain \\
        --yaml_path ../data/Parkland.yaml --gpu 0

Set training_configs.finetune_ce.checkpoint_path to your best_pretrain_model.pth
(or set models.<name>.checkpoint_path) before running.
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
from models.W8A8Quant import calibrate_w8a8, has_w8a8_layers, export_quantized_checkpoint
from train_test.loss import get_loss_function
from train_test.train_test_utils import (
    setup_experiment_dir,
    setup_train_file_logging,
    train,
    train_vanilla_supervised_contrastive,
    setup_optimizer,
    setup_scheduler,
    apply_class_subset,
    validate_finetune_config,
    load_pretrained_backbone,
    apply_finetune_backbone_freeze,
    freeze_backbone_bn,
)
from train_test.normalize import setup_normalization

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def select_finetune_train_fn(loss_name):
    """Route finetune runs to the correct training loop for the configured loss."""
    if loss_name == "cross_entropy":
        return train
    if loss_name == "bce_multilabel":
        return train
    if loss_name == "ce_supcon":
        return train_vanilla_supervised_contrastive
    raise ValueError(
        f"Unsupported finetune loss_name '{loss_name}'. "
        "Expected 'cross_entropy', 'bce_multilabel', or 'ce_supcon'."
    )


def main():
    logging.info("=" * 80)
    logging.info("FINETUNE SCRIPT")
    logging.info("=" * 80)

    config = get_config()
    logging.info("Configuration loaded successfully")

    (
        experiment_name,
        experiment_config,
        model_name,
        training_config_name,
        training_config,
        pretrained_checkpoint_path,
        num_epochs,
    ) = validate_finetune_config(config)

    apply_class_subset(config)

    simple_model_training = False
    if "simple_model_training" in experiment_config:
        simple_model_training = bool(experiment_config["simple_model_training"])

    train_loader, val_loader, test_loader = create_dataloaders(config=config)

    logging.info("\nSetting up normalization...")
    train_loader, val_loader, test_loader = setup_normalization(
        train_loader, val_loader, test_loader, config
    )
    logging.info("Normalization setup complete")

    logging.info("\nCreating augmenter...")
    augmenter = create_augmenter(
        config, augmentation_mode="fixed", experiment_config=experiment_config
    )
    logging.info("Augmenter created successfully")

    finetune_experiment_name = f"finetune_{experiment_name}"
    logging.info("\nSetting up experiment directory...")
    experiment_dir, tensorboard_dir = setup_experiment_dir(
        config, experiment_name=finetune_experiment_name
    )
    tb_run_dir = Path(tensorboard_dir).resolve()
    logging.info(
        'TensorBoard (this run): tensorboard --logdir="%s" --port=6006',
        tb_run_dir,
    )
    logging.info(
        "To compare multiple runs, point --logdir at the parent experiments directory instead."
    )

    setup_train_file_logging(experiment_dir, argv=sys.argv)

    logging.info("\n" + "=" * 80)
    logging.info("STARTING FINE-TUNING")
    logging.info("=" * 80 + "\n")

    try:
        config["models"][model_name]["pretrain_mode"] = False
        model = create_single_modal_model(config, model_name)
        logging.info(f"Model created: {model_name} (pretrain_mode=False)")

        load_pretrained_backbone(model, pretrained_checkpoint_path)

        if training_config["freeze_backbone"]:
            apply_finetune_backbone_freeze(model)
        elif "backbone_lr_scale" in training_config:
            freeze_backbone_bn(model)

        _location = config["location_names"][0]
        _model_cfg = config["models"][model_name]
        _modality = _model_cfg["active_modality"]
        _in_ch = _model_cfg["in_channels"] if "in_channels" in _model_cfg else config["loc_mod_in_freq_channels"][_location][_modality]
        _n_segs = config["num_segments"]
        _spec_len = (
            _model_cfg["in_spectrum_len"]
            if "in_spectrum_len" in _model_cfg
            else config["loc_mod_spectrum_len"][_location][_modality]
        )
        _dummy = {_location: {_modality: torch.randn(1, _in_ch, _n_segs, _spec_len)}}
        if _model_cfg["model_type"] == "deepsense_dw_simple":
            _mem = get_total_memory(model, _dummy[_location][_modality], unit="MB")
        else:
            _mem = get_total_memory(model, _dummy, unit="MB")
        logging.info("Memory profile (float32 / INT8 weight-only estimate):")
        logging.info(
            "  Parameters : %.2f MB  (INT8 ≈ %.2f MB)",
            _mem["parameter_memory"],
            _mem["parameter_memory"] / 4,
        )
        logging.info("  Peak activation (B=1): %.3f MB", _mem["activation_memory"])
        del _dummy, _mem

        # W8A8 calibration: run before optimizer/loss setup so scales are
        # locked in before the first QAT gradient update.
        if has_w8a8_layers(model):
            _device = torch.device(
                f"cuda:{config.get('gpu', 0)}" if torch.cuda.is_available() else "cpu"
            )
            n_calib = int(training_config.get("w8a8_calib_batches", 50))
            logging.info(
                "\nRunning W8A8 activation calibration (%d batches)...", n_calib
            )
            calibrate_model = model
            if simple_model_training:
                _location_name = config["location_names"][0]
                _modality_name = config["models"][model_name]["active_modality"]

                class _SimpleInputWrapper(torch.nn.Module):
                    def __init__(self, base_model, location_name, modality_name):
                        super().__init__()
                        self.base_model = base_model
                        self.location_name = location_name
                        self.modality_name = modality_name

                    def forward(self, inputs):
                        return self.base_model(
                            inputs[self.location_name][self.modality_name]
                        )

                calibrate_model = _SimpleInputWrapper(
                    model, _location_name, _modality_name
                )
            calibrate_w8a8(
                calibrate_model, train_loader, _device,
                n_batches=n_calib,
                augmenter=augmenter,
                apply_augmentation_fn=apply_augmentation,
            )
            logging.info("W8A8 calibration done — QAT fake-quant enabled.\n")

        logging.info("\nSetting up loss function...")
        loss_fn, _ = get_loss_function(training_config)

        logging.info("\nSetting up optimizer and scheduler...")
        optimizer = setup_optimizer(model, config, training_config=training_config)
        scheduler = setup_scheduler(optimizer, config, training_config)

        selected_train_fn = select_finetune_train_fn(training_config["loss_name"])
        logging.info(
            f"\nStarting supervised fine-tuning with loss function: {training_config['loss_name']}..."
        )
        train_kwargs = {
            "model": model,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "config": config,
            "experiment_dir": experiment_dir,
            "loss_fn": loss_fn,
            "val_fn": None,
            "augmenter": augmenter,
            "apply_augmentation_fn": apply_augmentation,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "num_epochs": num_epochs,
            "model_name": model_name,
            "training_config": training_config,
        }
        if selected_train_fn is train:
            train_kwargs["simple_model_training"] = simple_model_training

        model, _, best_checkpoint_path = selected_train_fn(
            **train_kwargs
        )

        if has_w8a8_layers(model) and best_checkpoint_path is not None:
            quantized_path = str(Path(best_checkpoint_path).parent / "best_model_quantized.pth")
            logging.info("\nExporting quantized checkpoint (int8 weights + scales)...")
            export_quantized_checkpoint(model, best_checkpoint_path, quantized_path)
            logging.info(f"Quantized checkpoint: {quantized_path}")

        config["models"][model_name]["checkpoint_path"] = best_checkpoint_path
        config["models"][model_name]["pretrained_checkpoint_path"] = (
            pretrained_checkpoint_path
        )

        config_path = Path(experiment_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logging.info("\n" + "=" * 80)
        logging.info("FINE-TUNING COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)
        logging.info(f"\nExperiment directory: {experiment_dir}")
        logging.info(f"Pretrained checkpoint: {pretrained_checkpoint_path}")
        logging.info(f"Best finetune checkpoint: {best_checkpoint_path}")
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
        logging.warning("Fine-tuning interrupted by user")
        logging.info("=" * 80)
        logging.info(f"Experiment directory: {experiment_dir}")
        sys.exit(0)

    except Exception as e:
        logging.error("\n" + "=" * 80)
        logging.error("ERROR DURING FINE-TUNING")
        logging.error("=" * 80)
        logging.error(f"Error: {e}")

        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
