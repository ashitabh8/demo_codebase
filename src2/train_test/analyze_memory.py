"""
Print memory profile for all models defined in ACIDS.yaml.
Usage: python analyze_memory.py --yaml_path ../data/ACIDS.yaml
"""
import argparse
import sys
import torch
import yaml
from pathlib import Path

src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from models.create_models import (
    create_single_modal_model,
    get_total_memory,
    get_input_memory,
    log_memory_info,
)


def make_dummy_input(config, model_config_key, batch_size=4):
    model_cfg = config["models"][model_config_key]
    active_modality = model_cfg["active_modality"]
    location_name = config["location_names"][0]
    in_channels = config["loc_mod_in_freq_channels"][location_name][active_modality]
    # Use spectrum length for spatial dims if available, else default 128
    spectrum_len = config.get("loc_mod_spectrum_len", {}).get(location_name, {}).get(active_modality, 128)
    # num_segments is the model's temporal input dimension; seq_len is for contrastive learning
    num_intervals = config.get("num_segments", config.get("seq_len", 128))
    tensor = torch.randn(batch_size, in_channels, num_intervals, spectrum_len)
    return {location_name: {active_modality: tensor}}


def analyze_model(config, model_name, unit="KB"):
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")

    config["models"][model_name]["pretrain_mode"] = False
    model = create_single_modal_model(config, model_name)
    model.eval()

    input_dict = make_dummy_input(config, model_name)
    input_mem = get_input_memory(input_dict, unit=unit)
    memory_info = get_total_memory(model, input_dict, unit=unit)
    print(f"  Parameters: {memory_info['parameter_memory']:.2f} {unit}")
    print(f"  Activations (per sample): {memory_info['activation_memory']:.2f} {unit}")
    print(f"  Total (per sample): {memory_info['total_memory']:.2f} {unit}")


def main():
    parser = argparse.ArgumentParser(description="Analyze model memory profiles")
    parser.add_argument("--yaml_path", default="../data/ACIDS.yaml")
    parser.add_argument("--unit", default="KB", choices=["B", "KB", "MB"])
    parser.add_argument("--model", default=None, help="Analyze a specific model key (default: all)")
    args = parser.parse_args()

    with open(args.yaml_path) as f:
        config = yaml.safe_load(f)

    model_keys = list(config["models"].keys())
    if args.model:
        if args.model not in model_keys:
            print(f"Error: '{args.model}' not found. Available: {model_keys}")
            sys.exit(1)
        model_keys = [args.model]

    import logging
    logging.disable(logging.CRITICAL)  # suppress verbose model creation logs

    for model_name in model_keys:
        analyze_model(config, model_name, unit=args.unit)


if __name__ == "__main__":
    main()