#!/usr/bin/env python3
"""
Export W8 DeepSense model to C via Tiny-NN-in-C.

Loads a finetune checkpoint, calibrates conv activation scales on ACIDS train
data (n_fft=25 mel), applies mixed quantization (conv static int8 + linear
weight-only int8 per-channel), and writes model.c / model.h / weights.h.

Usage:
  python3 export_w8_model.py \\
    --ckpt_path ../src2/experiments/20260706_183051_finetune_finetune_audio_deepsense_dw_large_mel_5class_nfft25/models/best_model.pth \\
    --output_dir .

  python3 export_w8_model.py --help
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

PKG = Path(__file__).resolve().parent
SRC2 = PKG.parent / "src2"
DEFAULT_TINY_NN = Path.home() / "Tiny-NN-in-C"
DEFAULT_OUTPUT = PKG.parent / "ARL_deploy"

DEFAULT_CKPT = (
    SRC2
    / "experiments/20260706_183051_finetune_finetune_audio_deepsense_dw_large_mel_5class_nfft25/models/best_model.pth"
)


class DeepSenseCompileWrapper(nn.Module):
    """Compiler-facing wrapper; IR node names use wrapped_backbone_* prefix."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.wrapped_backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.wrapped_backbone(x)
        return out["logits"]


def _ir_name_to_stats_key(ir_name: str, layer_stats: dict) -> str:
    for key in layer_stats:
        if key.replace(".", "_") == ir_name:
            return key
    return ir_name.replace("_", ".")


def build_mixed_quant_rules(stats, ir_graph):
    from src.pytorch_to_c.quantization import (
        Int8WeightOnlyLinearRule,
        StaticPerChannelConvQuantRule,
    )

    rules = []
    for node in ir_graph.nodes:
        if node.op_type not in ("conv2d", "conv1d", "linear"):
            continue
        pattern = f"^{re.escape(node.name)}$"
        if node.op_type in ("conv2d", "conv1d"):
            stats_key = _ir_name_to_stats_key(node.name, stats.layer_stats)
            in_scale = stats.get_input_scale(stats_key, "int8")
            out_scale = stats.get_output_scale(stats_key, "int8")
            rules.append(
                StaticPerChannelConvQuantRule(
                    pattern=pattern,
                    dtype="int8",
                    input_scale=in_scale,
                    input_offset=0,
                    output_scale=out_scale,
                    output_offset=0,
                )
            )
        elif node.op_type == "linear":
            rules.append(Int8WeightOnlyLinearRule(pattern=pattern))
    if not rules:
        raise RuntimeError("No quantizable conv/linear nodes found in IR graph")
    return rules


def load_backbone_1ch_from_checkpoint(
    backbone: nn.Module, ckpt_path: Path, channel_index: int = 0
) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"Expected model_state_dict in {ckpt_path}")
    sd = ckpt["model_state_dict"]
    mapped = {}
    for key, value in sd.items():
        if not key.startswith("backbone."):
            continue
        local_key = key[len("backbone.") :]
        if local_key == "freq_stack.0.depthwise.weight" and value.shape[0] > 1:
            mapped[local_key] = value[channel_index : channel_index + 1].clone()
        elif (
            local_key == "freq_stack.0.pointwise.weight"
            and value.ndim == 4
            and value.shape[1] > 1
        ):
            mapped[local_key] = value[:, channel_index : channel_index + 1, :, :].clone()
        else:
            mapped[local_key] = value
    missing, unexpected = backbone.load_state_dict(mapped, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys loading backbone: {unexpected}")
    allowed_missing = {
        "projection_head.0.weight",
        "projection_head.0.bias",
        "projection_head.2.weight",
        "projection_head.2.bias",
    }
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing:
        raise RuntimeError(f"Missing backbone keys: {bad_missing}")


def make_calib_iterator(train_loader, augmenter, device, channel_index: int):
    for batch in train_loader:
        data, labels, _idx = batch
        freq_data, _ = augmenter.forward_noaug(data, labels)
        x = freq_data["shake"]["audio"]
        x = x[:, channel_index : channel_index + 1, :, :]
        yield x.to(device)


def export_w8_model(
    ckpt_path: Path,
    output_dir: Path,
    tiny_nn_root: Path,
    max_calib_batches: int,
    channel_index: int,
    device: str,
) -> None:
    # Import demo_codebase modules before Tiny-NN-in-C (which also has a `models` pkg).
    sys.path.insert(0, str(SRC2))
    from data_augmenter.augmenter_utils import create_augmenter
    from dataset_utils.MultiModalDataLoader import create_dataloaders
    from models.DeepSenseDepthwise import DeepSenseDepthwiseBackbone
    from train_test.train_test_utils import apply_class_subset

    sys.path.insert(0, str(tiny_nn_root))
    from src.pytorch_to_c.calibration import calibrate
    from src.pytorch_to_c.codegen.c_printer import CPrinter
    from src.pytorch_to_c.compiler import compile_model
    from src.pytorch_to_c.quantization import QuantizationTransform

    experiment_dir = ckpt_path.parent.parent
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if "include_classes_mapping" not in config:
        apply_class_subset(config)
    experiment_name = config["experiment_name"]
    experiment_config = config["experiments"][experiment_name]
    model_cfg = config["models"][experiment_config["model"]]

    num_classes = config[config["task_name"]]["num_classes"]
    backbone = DeepSenseDepthwiseBackbone(
        in_channels=1,
        in_spectrum_len=model_cfg["in_spectrum_len"],
        num_classes=num_classes,
        channels_freq=model_cfg["channels_freq"],
        kernel_sizes_freq=model_cfg["kernel_sizes_freq"],
        strides_freq=model_cfg["strides_freq"],
        temporal_channels=model_cfg["temporal_channels"],
        num_temporal_layers=model_cfg["num_temporal_layers"],
        temporal_kernel=model_cfg["temporal_kernel"],
        fc_dim=model_cfg["fc_dim"],
        dropout_ratio=0.0,
        pretrain_mode=False,
        proj_hidden_dim=model_cfg["proj_hidden_dim"],
        proj_out_dim=model_cfg["proj_out_dim"],
    )
    load_backbone_1ch_from_checkpoint(backbone, ckpt_path, channel_index)
    backbone.eval()

    wrapper = DeepSenseCompileWrapper(backbone)
    wrapper.eval()

    example_input = torch.randn(1, 1, 7, 80)
    print(f"Compiling IR graph (example input {list(example_input.shape)})...")
    ir_graph = compile_model(wrapper, example_input, return_ir=True, verbose=False)

    config["device"] = device
    train_loader, _, _ = create_dataloaders(config=config)
    augmenter = create_augmenter(
        config, augmentation_mode="no", experiment_config=experiment_config
    )

    print(f"Calibrating activation scales ({max_calib_batches} train batches)...")
    calib_data = make_calib_iterator(
        train_loader, augmenter, device, channel_index
    )
    stats = calibrate(
        wrapper.to(device),
        calib_data,
        max_batches=max_calib_batches,
    )
    print(f"  Calibrated {stats.num_batches} batches, {len(stats.layer_stats)} layers")

    rules = build_mixed_quant_rules(stats, ir_graph)
    print(f"Applying {len(rules)} quantization rules...")
    ir_graph = QuantizationTransform(rules).apply(ir_graph)

    output_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = output_dir / "_tiny_nn_gen"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    print(f"Generating C code -> {gen_dir}")
    CPrinter(ir_graph).generate_all(str(gen_dir))

    for name in ("model.c", "model.h", "weights.h", "nn_ops_float.h", "nn_ops_int8.h", "nn_ops_int16.h"):
        src = gen_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            print(f"  copied {name}")
    shutil.rmtree(gen_dir)
    print(f"Done. W8 C artifacts in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export W8 DeepSense checkpoint to C via Tiny-NN-in-C"
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=DEFAULT_CKPT,
        help="Path to float32 best_model.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for model.c, model.h, weights.h (default: deploy package)",
    )
    parser.add_argument(
        "--tiny_nn_root",
        type=Path,
        default=DEFAULT_TINY_NN,
        help="Path to Tiny-NN-in-C repository",
    )
    parser.add_argument(
        "--max_calib_batches",
        type=int,
        default=32,
        help="ACIDS train batches for conv activation scale calibration",
    )
    parser.add_argument(
        "--channel_index",
        type=int,
        default=0,
        help="Audio channel index for 1-ch C deployment (0/1/2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for calibration forward passes",
    )
    args = parser.parse_args()

    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    if not args.tiny_nn_root.exists():
        raise FileNotFoundError(
            f"Tiny-NN-in-C not found at {args.tiny_nn_root}. "
            "Set --tiny_nn_root to your clone."
        )

    export_w8_model(
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        tiny_nn_root=args.tiny_nn_root,
        max_calib_batches=args.max_calib_batches,
        channel_index=args.channel_index,
        device=args.device,
    )


if __name__ == "__main__":
    main()
