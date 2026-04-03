#!/usr/bin/env python3
"""Export deterministic inputs and PyTorch W8A8 reference outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC2_ROOT = REPO_ROOT / "src2"

DEFAULT_YAML = SRC2_ROOT / "data" / "ACIDS.yaml"
DEFAULT_MODEL_NAME = "student_audio_deepsense_dw_simple_tiny_w8a8"
DEFAULT_EXPERIMENT_NAME = "finetune_audio_deepsense_dw_simple_tiny_w8a8"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "test_data"
DEFAULT_HEADER = Path(__file__).resolve().parent / "test_inputs.h"
DEFAULT_EXPECTED = Path(__file__).resolve().parent / "expected_outputs.txt"

sys.path.insert(0, str(SRC2_ROOT))

from data_augmenter import create_augmenter, apply_augmentation  # noqa: E402
from dataset_utils.MultiModalDataLoader import create_dataloaders  # noqa: E402
from models.W8A8Quant import freeze_w8a8  # noqa: E402
from models.create_models import create_single_modal_model  # noqa: E402
from train_test.normalize import setup_normalization  # noqa: E402


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
        "No simple W8A8 best_model.pth found. Train finetune_audio_deepsense_dw_simple_tiny_w8a8 first."
    )


def _load_config(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["task_name"] = "vehicle_classification"
    return cfg


def _load_state_dict(checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def _load_w8a8_model(yaml_path: Path, checkpoint_path: Path, model_name: str) -> torch.nn.Module:
    config = _load_config(yaml_path)
    model = create_single_modal_model(config, model_name)
    state_dict = _load_state_dict(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    freeze_w8a8(model)
    model.eval()
    return model


def _collect_real_samples(
    config: dict,
    experiment_name: str,
    model_name: str,
    num_samples: int,
    split: str,
) -> list[torch.Tensor]:
    config["experiment_name"] = experiment_name
    experiment_config = config["experiments"][experiment_name]
    model_cfg = config["models"][model_name]
    location_name = config["location_names"][0]
    modality_name = model_cfg["active_modality"]

    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    train_loader, val_loader, test_loader = setup_normalization(
        train_loader, val_loader, test_loader, config
    )
    augmenter = create_augmenter(
        config, augmentation_mode="fixed", experiment_config=experiment_config
    )

    if split == "train":
        loader = train_loader
    elif split == "val":
        loader = val_loader
    else:
        loader = test_loader

    samples: list[torch.Tensor] = []
    for batch in loader:
        if len(batch) == 2:
            data, labels = batch
        elif len(batch) == 3:
            data, labels, _ = batch
        else:
            data = batch[0]
            labels = batch[1]

        data, _ = apply_augmentation(augmenter, data, labels)
        x = data[location_name][modality_name]
        for i in range(x.shape[0]):
            samples.append(x[i : i + 1].detach().cpu())
            if len(samples) >= num_samples:
                return samples

    if len(samples) < num_samples:
        raise ValueError(
            f"Requested {num_samples} samples but only found {len(samples)} in split='{split}'"
        )
    return samples


def _write_test_header(header_path: Path, samples_nhwc: list[np.ndarray], output_size: int) -> None:
    input_size = int(samples_nhwc[0].size)
    with header_path.open("w", encoding="utf-8") as handle:
        handle.write("// Auto-generated W8A8 test input arrays\n")
        handle.write("#pragma once\n\n")
        handle.write(f"#define NUM_TEST_SAMPLES {len(samples_nhwc)}\n")
        handle.write(f"#define TEST_INPUT_SIZE {input_size}\n")
        handle.write(f"#define TEST_OUTPUT_SIZE {output_size}\n\n")

        for idx, sample in enumerate(samples_nhwc):
            values = ", ".join(f"{float(v):.9e}f" for v in sample.reshape(-1))
            handle.write(f"static const float sample_{idx}[TEST_INPUT_SIZE] = {{{values}}};\n")
        handle.write("\n")

        ptrs = ", ".join(f"sample_{idx}" for idx in range(len(samples_nhwc)))
        handle.write(f"static const float* const TEST_INPUTS[NUM_TEST_SAMPLES] = {{{ptrs}}};\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export W8A8 test data and references.")
    parser.add_argument("--yaml_path", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--split", type=str, choices=("train", "val", "test"), default="val")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--header_path", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--expected_outputs_path", type=Path, default=DEFAULT_EXPECTED)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = _latest_simple_w8a8_checkpoint()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.header_path.parent.mkdir(parents=True, exist_ok=True)
    args.expected_outputs_path.parent.mkdir(parents=True, exist_ok=True)

    model = _load_w8a8_model(args.yaml_path, checkpoint_path, args.model_name)
    config = _load_config(args.yaml_path)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_inputs = _collect_real_samples(
        config,
        args.experiment_name,
        args.model_name,
        args.num_samples,
        args.split,
    )

    nhwc_samples: list[np.ndarray] = []
    expected_lines: list[str] = []
    output_size = 0

    for idx, x_nchw in enumerate(model_inputs):
        with torch.no_grad():
            logits = model(x_nchw).detach().cpu().numpy().reshape(-1)
        output_size = int(logits.size)

        x_nhwc = (
            x_nchw[0]
            .permute(1, 2, 0)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        nhwc_samples.append(x_nhwc)

        np.savetxt(args.data_dir / f"test_input_{idx}.txt", x_nhwc, fmt="%.9g")
        np.savetxt(args.data_dir / f"pytorch_output_{idx}.txt", logits, fmt="%.9g")
        expected_lines.append(
            "sample_"
            + str(idx)
            + " "
            + " ".join(f"{float(v):.9g}" for v in logits.tolist())
        )

    _write_test_header(args.header_path, nhwc_samples, output_size)
    args.expected_outputs_path.write_text("\n".join(expected_lines) + "\n", encoding="utf-8")

    print(f"[export] checkpoint: {checkpoint_path}")
    print(f"[export] split: {args.split}")
    print(f"[export] samples: {args.num_samples}")
    print(f"[export] header: {args.header_path}")
    print(f"[export] expected outputs: {args.expected_outputs_path}")


if __name__ == "__main__":
    main()
