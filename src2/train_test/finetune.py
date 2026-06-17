"""
Supervised fine-tuning from a pretrained checkpoint, or from random init.

If experiments[..].checkpoint_path (or models.<name>.checkpoint_path) resolves
to a non-empty path, loads backbone weights from that .pth (projection head keys
skipped), optionally freezes early layers, then runs the configured training loop.

If both paths are empty or missing, skips weight loading and uses the weights
from create_single_modal_model (requires freeze_backbone: false in the finetune
training_config).

Usage:
    python finetune.py --experiment_name finetune_audio_resnet_from_pretrain \\
        --yaml_path ../data/ACIDS.yaml --gpu 0

Set checkpoint_path to your best_pretrain_model.pth for pretrain-based finetune,
or to \"\" on the experiment (and omit or clear the model entry) for from-scratch.
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


def export_int8_weight_checkpoint(float_checkpoint_path, output_path):
    """
    Export an int8-packed checkpoint from a float/fake-quant checkpoint.

    For each floating-point weight tensor (keys ending with '.weight', ndim>=2),
    stores:
      - int8 tensor in model_state_dict_int8[key]
      - per-tensor symmetric scale in weight_scales[key]

    Dequantization rule:
      weight_fp32 ~= weight_int8.float() * scale
    """
    ckpt = torch.load(float_checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(
            "Expected checkpoint with 'model_state_dict' for int8 export"
        )

    fp32_sd = ckpt["model_state_dict"]
    qsd = {}
    scales = {}
    quantized_count = 0

    for key, value in fp32_sd.items():
        if (
            torch.is_tensor(value)
            and value.is_floating_point()
            and key.endswith(".weight")
            and value.ndim >= 2
        ):
            max_abs = float(value.detach().abs().max().item())
            scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
            q = torch.clamp(torch.round(value / scale), -127, 127).to(torch.int8)
            qsd[key] = q.cpu()
            scales[key] = scale
            quantized_count += 1
        elif torch.is_tensor(value):
            qsd[key] = value.detach().cpu()
        else:
            qsd[key] = value

    out = {
        "model_state_dict_int8": qsd,
        "weight_scales": scales,
        "source_checkpoint": str(float_checkpoint_path),
        "format": "symmetric_per_tensor_int8_v1",
        "quantized_weight_tensors": quantized_count,
    }
    torch.save(out, output_path)
    return quantized_count


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

    train_loader, val_loader, test_loader = create_dataloaders(config=config)

    # logging.info("\nSetting up normalization...")
    # train_loader, val_loader, test_loader = setup_normalization(
    #     train_loader, val_loader, test_loader, config
    # )
    # logging.info("Normalization setup complete")

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

        if pretrained_checkpoint_path is not None:
            load_pretrained_backbone(model, pretrained_checkpoint_path)
        else:
            logging.info(
                "No checkpoint_path: using randomly initialized weights from "
                "create_single_modal_model (load_pretrained_backbone skipped)."
            )

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
        _mem = get_total_memory(model, _dummy, unit="MB")
        logging.info("Memory profile (float32 / INT8 weight-only estimate):")
        logging.info(
            "  Parameters : %.2f MB  (INT8 ≈ %.2f MB)",
            _mem["parameter_memory"],
            _mem["parameter_memory"] / 4,
        )
        logging.info("  Peak activation (B=1): %.3f MB", _mem["activation_memory"])
        del _dummy, _mem

        logging.info("\nSetting up loss function...")
        loss_fn, _ = get_loss_function(training_config)

        logging.info("\nSetting up optimizer and scheduler...")
        optimizer = setup_optimizer(model, config, training_config=training_config)
        scheduler = setup_scheduler(optimizer, config, training_config)

        selected_train_fn = select_finetune_train_fn(training_config["loss_name"])
        logging.info(
            f"\nStarting supervised fine-tuning with loss function: {training_config['loss_name']}..."
        )
        model, _, best_checkpoint_path = selected_train_fn(
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
            num_epochs=num_epochs,
            model_name=model_name,
            training_config=training_config,
        )

        _model_cfg = config["models"][model_name]
        if _model_cfg.get("w8a8") or _model_cfg.get("w8a16"):
            quantized_ckpt_path = str(
                Path(best_checkpoint_path).with_name("best_model_quantized.pth")
            )
            n_q = export_int8_weight_checkpoint(
                best_checkpoint_path, quantized_ckpt_path
            )
            logging.info(
                "Exported int8-packed checkpoint: %s (%d weight tensors quantized)",
                quantized_ckpt_path,
                n_q,
            )

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
        if pretrained_checkpoint_path is not None:
            logging.info(f"Pretrained checkpoint: {pretrained_checkpoint_path}")
        else:
            logging.info("Pretrained checkpoint: (none — started from random init)")
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
