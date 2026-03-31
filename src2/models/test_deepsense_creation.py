"""
Test: SingleModalDeepSense model creation and single-batch inference.

Run from repo root:
    cd /home/misra8/demo_codebase
    python src2/models/test_deepsense_creation.py

Expected: all three models pass shape checks and print "All tests passed!"
"""
import sys
from pathlib import Path

src2_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src2_dir))

import torch
import yaml

from models.create_models import create_single_modal_model


def make_dummy_batch(config, modality, location, batch_size=4):
    C = config["loc_mod_in_freq_channels"][location][modality]
    S = config["loc_mod_spectrum_len"][location][modality]
    I = config["num_segments"]
    return {location: {modality: torch.randn(batch_size, C, I, S)}}


def test_model(config, model_key, batch_size=4):
    print(f"\n{'='*60}\nTesting: {model_key}\n{'='*60}")

    model = create_single_modal_model(config, model_key)
    model.eval()

    model_cfg = config["models"][model_key]
    modality = model_cfg["active_modality"]
    location = config["location_names"][0]

    dummy_input = make_dummy_batch(config, modality, location, batch_size)
    tensor_shape = dummy_input[location][modality].shape
    print(f"Input: {location}/{modality} -> {tuple(tensor_shape)}")

    with torch.no_grad():
        output = model(dummy_input)

    assert isinstance(output, dict), f"Expected dict, got {type(output)}"
    assert "logits" in output and "features" in output, f"Missing keys: {output.keys()}"

    num_classes = config[config["task_name"]]["num_classes"]
    fc_dim = model_cfg["fc_dim"]

    assert output["logits"].shape == (batch_size, num_classes), \
        f"logits shape {output['logits'].shape} != ({batch_size}, {num_classes})"
    assert output["features"].shape == (batch_size, fc_dim), \
        f"features shape {output['features'].shape} != ({batch_size}, {fc_dim})"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.3f}M)")
    print(f"logits:   {tuple(output['logits'].shape)}   ✓")
    print(f"features: {tuple(output['features'].shape)}  ✓")
    print("PASSED")


def main():
    yaml_path = src2_dir / "data" / "Parkland.yaml"
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    config["task_name"] = "fine_tune_vehicle_classification"

    test_model(config, "student_audio_deepsense_small")
    test_model(config, "student_audio_deepsense")
    test_model(config, "student_audio_deepsense_dw")
    test_model(config, "student_audio_deepsense_dw_large")
    test_model(config, "student_audio_resnet")   # sanity: existing model unbroken

    print(f"\n{'='*60}\nAll tests passed!\n{'='*60}")


if __name__ == "__main__":
    main()
