#!/usr/bin/env python3
"""Compile W8A8 simple DeepSense model to C with per-channel conv scales."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC2_ROOT = REPO_ROOT / "src2"
COMPILER_ROOT = REPO_ROOT / "Tiny-NN-in-C"

DEFAULT_YAML = SRC2_ROOT / "data" / "ACIDS.yaml"
DEFAULT_MODEL_NAME = "student_audio_deepsense_dw_simple_tiny_w8a8"
DEFAULT_FLOAT_MODEL_NAME = "student_audio_deepsense_dw_simple_tiny"
DEFAULT_EXPERIMENT_NAME = "finetune_audio_deepsense_dw_simple_tiny_w8a8"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated"
DEFAULT_SCALES_JSON = Path(__file__).resolve().parent / "quant_scales.json"
DEFAULT_EXAMPLE_SHAPE = (1, 6, 7, 256)
DEFAULT_CALIB_BATCHES = 50

sys.path.insert(0, str(SRC2_ROOT))
sys.path.insert(0, str(COMPILER_ROOT))

from models.QuantBase import QuantDWConv2d, INT8_MAX  # noqa: E402
from data_augmenter import create_augmenter, apply_augmentation  # noqa: E402
from dataset_utils.MultiModalDataLoader import create_dataloaders  # noqa: E402
from models.create_models import create_single_modal_model  # noqa: E402
from train_test.normalize import setup_normalization  # noqa: E402
from src.pytorch_to_c.codegen.c_printer import CPrinter  # noqa: E402
from src.pytorch_to_c.compiler import compile_model  # noqa: E402
from src.pytorch_to_c.quantization.graph_transform import QuantizationTransform  # noqa: E402
from src.pytorch_to_c.quantization.rules import (  # noqa: E402
    QuantRule,
    StaticQuantRule,
    QATStaticDepthwiseConvRule,
    QATStaticPointwiseConvRule,
)


def _latest_simple_w8a8_checkpoint() -> Path:
    experiments_dir = SRC2_ROOT / "experiments"
    candidates = sorted(
        experiments_dir.glob("*finetune_finetune_audio_deepsense_dw_simple_tiny_w8a8")
    )
    for exp_dir in reversed(candidates):
        ckpt = exp_dir / "models" / "best_model.pth"
        if ckpt.exists():
            return ckpt
    raise FileNotFoundError(
        "No simple W8A8 best_model.pth found. Train with "
        "finetune_audio_deepsense_dw_simple_tiny_w8a8 first."
    )


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def _clone_with_task_name(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg["task_name"] = "vehicle_classification"
    return cfg


def _load_quant_model(config: dict, model_name: str, checkpoint_path: Path) -> torch.nn.Module:
    model = create_single_modal_model(config, model_name)
    state_dict = _load_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[compile] missing keys when loading quant model: {len(missing)}")
    if unexpected:
        print(f"[compile] unexpected keys when loading quant model: {len(unexpected)}")
    model.eval()
    return model


def _load_float_model_from_quant_weights(
    config: dict,
    float_model_name: str,
    quant_state_dict: dict[str, torch.Tensor],
) -> torch.nn.Module:
    model = create_single_modal_model(config, float_model_name)
    filtered = {}
    for key, tensor in quant_state_dict.items():
        if key.endswith("act_scale"):
            continue
        filtered[key] = tensor
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[compile] missing keys when loading float model: {len(missing)}")
    if unexpected:
        print(f"[compile] unexpected keys when loading float model: {len(unexpected)}")
    model.eval()
    return model


def _per_channel_scale(weight: torch.Tensor) -> np.ndarray:
    out_ch = weight.shape[0]
    flattened = weight.detach().float().reshape(out_ch, -1)
    scales = flattened.abs().max(dim=1).values.clamp(min=1e-8) / INT8_MAX
    return scales.cpu().numpy().astype(np.float32)


def _collect_conv_stats(
    model: torch.nn.Module,
    data_loader,
    augmenter,
    location_name: str,
    modality_name: str,
    calib_batches: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    hooks = []

    def _make_hook(module_name: str):
        def _hook(_module, inputs, output):
            input_tensor = inputs[0].detach()
            output_tensor = output.detach()
            node_name = module_name.replace(".", "_")
            if node_name not in stats:
                stats[node_name] = {"input_absmax": 0.0, "output_absmax": 0.0}
            stats[node_name]["input_absmax"] = max(
                stats[node_name]["input_absmax"], float(input_tensor.abs().max().item())
            )
            stats[node_name]["output_absmax"] = max(
                stats[node_name]["output_absmax"], float(output_tensor.abs().max().item())
            )
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(_make_hook(name)))

    model.to(device)
    model.eval()
    batches_seen = 0
    with torch.no_grad():
        for batch in data_loader:
            if batches_seen >= calib_batches:
                break
            if len(batch) == 2:
                data, labels = batch
            elif len(batch) == 3:
                data, labels, _ = batch
            else:
                data = batch[0]
                labels = batch[1]

            data = {
                loc: {mod: t.to(device) for mod, t in mods.items()}
                for loc, mods in data.items()
            }
            labels = labels.to(device)
            data, _ = apply_augmentation(augmenter, data, labels)
            x = data[location_name][modality_name].to(device)
            model(x)
            batches_seen += 1

    for hook in hooks:
        hook.remove()

    scales = {}
    for node_name, node_stats in stats.items():
        input_scale = max(node_stats["input_absmax"] / INT8_MAX, 1e-8)
        output_scale = max(node_stats["output_absmax"] / INT8_MAX, 1e-8)
        scales[node_name] = {
            "input_scale": float(input_scale),
            "output_scale": float(output_scale),
            "input_absmax": float(node_stats["input_absmax"]),
            "output_absmax": float(node_stats["output_absmax"]),
        }
    return scales


def _build_conv_quant_plan(
    ir_graph,
    quant_model: torch.nn.Module,
    trained_input_scales: dict[str, float],
    activation_scales: dict[str, dict[str, float]],
) -> tuple[list[QuantRule], dict[str, np.ndarray], list[dict]]:
    weight_scales_by_weight_name: dict[str, np.ndarray] = {}
    quant_layer_map = {}
    for name, layer in quant_model.named_modules():
        if isinstance(layer, QuantDWConv2d):
            quant_layer_map[name.replace(".", "_")] = layer

    rules: list[QuantRule] = []
    report_rows: list[dict] = []

    for node in ir_graph.nodes:
        if node.op_type != "conv2d":
            continue

        node_name = node.name
        if node_name not in activation_scales:
            raise KeyError(f"Missing activation scales for conv node '{node_name}'")

        prefix = node_name.rsplit("_", 1)[0]
        if prefix not in quant_layer_map:
            raise KeyError(f"Missing quant layer mapping for conv node '{node_name}'")
        layer = quant_layer_map[prefix]

        if node_name.endswith("_depthwise"):
            per_channel = _per_channel_scale(layer.depthwise.weight)
            if node_name not in trained_input_scales:
                raise KeyError(
                    f"Missing trained act_scale for depthwise node '{node_name}'"
                )
            input_scale = trained_input_scales[node_name]
            rule = QATStaticDepthwiseConvRule(
                pattern=f"^{re.escape(node_name)}$",
                dtype="int8",
                input_scale=input_scale,
                input_offset=0,
                weight_scale=per_channel,
                weight_offset=0,
                output_scale=activation_scales[node_name]["output_scale"],
                output_offset=0,
            )
            input_scale_source = "trained_act_scale"
        elif node_name.endswith("_pointwise"):
            per_channel = _per_channel_scale(layer.pointwise.weight)
            depthwise_node_name = node_name.replace("_pointwise", "_depthwise")
            if depthwise_node_name not in activation_scales:
                raise KeyError(
                    f"Missing depthwise output scale for pointwise node '{node_name}'"
                )
            # Pointwise receives the depthwise output tensor; preserve that link.
            input_scale = activation_scales[depthwise_node_name]["output_scale"]
            rule = QATStaticPointwiseConvRule(
                pattern=f"^{re.escape(node_name)}$",
                dtype="int8",
                input_scale=input_scale,
                input_offset=0,
                weight_scale=per_channel,
                weight_offset=0,
                output_scale=activation_scales[node_name]["output_scale"],
                output_offset=0,
            )
            input_scale_source = "depthwise_output_scale"
        else:
            # Legacy fallback: keep previous static quant path unchanged.
            per_channel = _per_channel_scale(layer.pointwise.weight)
            input_scale = activation_scales[node_name]["input_scale"]
            rule = StaticQuantRule(
                pattern=f"^{re.escape(node_name)}$",
                dtype="int8",
                input_scale=input_scale,
                input_offset=0,
                weight_scale=per_channel,
                weight_offset=0,
                output_scale=activation_scales[node_name]["output_scale"],
                output_offset=0,
            )
            input_scale_source = "calibration_input_scale"

        weight_name = node.metadata["weight_name"]
        weight_scales_by_weight_name[weight_name] = per_channel

        rules.append(rule)

        report_rows.append(
            {
                "node_name": node_name,
                "weight_name": weight_name,
                "input_scale": input_scale,
                "output_scale": activation_scales[node_name]["output_scale"],
                "input_scale_source": input_scale_source,
                "rule_type": type(rule).__name__,
                "weight_scale_min": float(np.min(per_channel)),
                "weight_scale_max": float(np.max(per_channel)),
                "weight_scale_len": int(per_channel.shape[0]),
            }
        )

    return rules, weight_scales_by_weight_name, report_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile W8A8 simple model to C.")
    parser.add_argument("--yaml_path", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--float_model_name", type=str, default=DEFAULT_FLOAT_MODEL_NAME)
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scales_json_path", type=Path, default=DEFAULT_SCALES_JSON)
    parser.add_argument("--calib_batches", type=int, default=DEFAULT_CALIB_BATCHES)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = _latest_simple_w8a8_checkpoint()

    config = _clone_with_task_name(_load_yaml(args.yaml_path))
    config["experiment_name"] = args.experiment_name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    experiment_config = config["experiments"][args.experiment_name]
    model_cfg = config["models"][args.model_name]
    location_name = config["location_names"][0]
    modality_name = model_cfg["active_modality"]

    print(f"[compile] checkpoint: {checkpoint_path}")
    print(f"[compile] device: {device}")

    quant_model = _load_quant_model(config, args.model_name, checkpoint_path)
    quant_state_dict = _load_state_dict(checkpoint_path)
    float_model = _load_float_model_from_quant_weights(
        config, args.float_model_name, quant_state_dict
    )

    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    train_loader, val_loader, test_loader = setup_normalization(
        train_loader, val_loader, test_loader, config
    )
    augmenter = create_augmenter(
        config, augmentation_mode="fixed", experiment_config=experiment_config
    )

    activation_scales = _collect_conv_stats(
        float_model,
        train_loader,
        augmenter,
        location_name,
        modality_name,
        args.calib_batches,
        device,
    )
    print(f"[compile] collected activation stats for {len(activation_scales)} conv nodes")

    trained_input_scales: dict[str, float] = {}
    for name, layer in quant_model.named_modules():
        if isinstance(layer, QuantDWConv2d):
            node_name = f"{name.replace('.', '_')}_depthwise"
            trained_input_scales[node_name] = float(layer.act_scale.item())
    print(
        f"[compile] loaded trained act_scales for {len(trained_input_scales)} depthwise nodes"
    )

    example_input = torch.randn(*DEFAULT_EXAMPLE_SHAPE)
    ir_graph = compile_model(model=float_model.cpu(), example_input=example_input, return_ir=True)

    rules, weight_scales_by_weight_name, report_rows = _build_conv_quant_plan(
        ir_graph, quant_model, trained_input_scales, activation_scales
    )
    print(f"[compile] built {len(rules)} static quant rules")

    ir_graph = QuantizationTransform(rules).apply(ir_graph)

    for weight_name, scales in weight_scales_by_weight_name.items():
        ir_graph.parameters[f"{weight_name}_scale"] = scales.astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    CPrinter(ir_graph).generate_all(str(args.output_dir))
    print(f"[compile] generated C files in: {args.output_dir}")

    args.scales_json_path.parent.mkdir(parents=True, exist_ok=True)
    with args.scales_json_path.open("w", encoding="utf-8") as handle:
        json.dump(report_rows, handle, indent=2)
    print(f"[compile] wrote scale report: {args.scales_json_path}")


if __name__ == "__main__":
    main()
