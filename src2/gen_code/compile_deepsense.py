#!/usr/bin/env python3
"""Compile trained DeepSenseDWSimpleBackbone to C code."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC2_ROOT = REPO_ROOT / "src2"
COMPILER_ROOT = REPO_ROOT / "Tiny-NN-in-C"
DEFAULT_YAML = SRC2_ROOT / "data" / "ACIDS.yaml"
DEFAULT_CKPT = (
    SRC2_ROOT
    / "experiments"
    / "20260402_005734_only_audio_deepsense_dw_simple_tiny"
    / "models"
    / "best_model.pth"
)
DEFAULT_OUTPUT = SRC2_ROOT / "gen_code" / "generated"
DEFAULT_MODEL_NAME = "student_audio_deepsense_dw_simple_tiny"
INPUT_SHAPE = (1, 6, 7, 256)

sys.path.insert(0, str(SRC2_ROOT))
sys.path.insert(0, str(COMPILER_ROOT))

from models.create_models import create_single_modal_model  # noqa: E402
from src.pytorch_to_c.compiler import compile_model  # noqa: E402
from src.pytorch_to_c.codegen.c_printer import CPrinter  # noqa: E402


def load_model(yaml_path: Path, checkpoint_path: Path, model_name: str) -> torch.nn.Module:
    with yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Needed by create_single_modal_model() for supervised head width resolution.
    config["task_name"] = "vehicle_classification"

    model = create_single_modal_model(config, model_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile DeepSenseDWSimpleBackbone to C.")
    parser.add_argument("--yaml_path", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.yaml_path, args.checkpoint_path, args.model_name)
    example_input = torch.randn(*INPUT_SHAPE)

    with torch.no_grad():
        output = model(example_input)
    print(f"Loaded model: {args.model_name}")
    print(f"Input shape: {list(example_input.shape)}")
    print(f"PyTorch output shape: {list(output.shape)}")

    ir_graph = compile_model(model=model, example_input=example_input, return_ir=True)
    print(f"IR nodes: {len(ir_graph.nodes)}")

    printer = CPrinter(ir_graph)
    printer.generate_all(str(args.output_dir))
    print(f"Generated C sources in: {args.output_dir}")
    print(ir_graph.print_graph())


if __name__ == "__main__":
    main()
