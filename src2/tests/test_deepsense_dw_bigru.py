"""
Regression: DeepSense DW + optional BiGRU + output_dims MLP head.
"""
import sys
from pathlib import Path

import torch
import yaml
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "train_test"))

from models.create_models import create_single_modal_model


@pytest.fixture
def parkland_config():
    yaml_path = Path(__file__).parent.parent / "data" / "Parkland.yaml"
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _mel_dummy(config, batch_size=2):
    loc = config["location_names"][0]
    mod = "audio"
    n_seg = config["num_segments"]
    x = torch.randn(batch_size, 1, n_seg, 80)
    return {loc: {mod: x}}


def test_dw_large_mel_bigru_supervised_shapes(parkland_config):
    cfg = parkland_config
    name = "student_audio_deepsense_dw_large_mel_bigru"
    cfg["models"][name]["pretrain_mode"] = False
    model = create_single_modal_model(cfg, name)
    model.eval()
    num_classes = cfg["vehicle_classification"]["num_classes"]
    out_dims = cfg["models"][name]["output_dims"]
    embed_dim = out_dims[-1]
    with torch.no_grad():
        o = model(_mel_dummy(cfg))
    assert o["logits"].shape == (2, num_classes)
    assert o["features"].shape == (2, embed_dim)


def test_dw_large_mel_bigru_pretrain_shapes(parkland_config):
    cfg = parkland_config
    name = "student_audio_deepsense_dw_large_mel_bigru"
    cfg["models"][name]["pretrain_mode"] = True
    model = create_single_modal_model(cfg, name)
    model.eval()
    proj_dim = cfg["models"][name]["proj_out_dim"]
    out_dims = cfg["models"][name]["output_dims"]
    embed_dim = out_dims[-1]
    with torch.no_grad():
        o = model(_mel_dummy(cfg))
    assert "projection" in o
    assert o["features"].shape == (2, embed_dim)
    assert o["projection"].shape == (2, proj_dim)


def test_dw_large_mel_legacy_unchanged(parkland_config):
    cfg = parkland_config
    name = "student_audio_deepsense_dw_large_mel"
    cfg["models"][name]["pretrain_mode"] = False
    model = create_single_modal_model(cfg, name)
    model.eval()
    num_classes = cfg["vehicle_classification"]["num_classes"]
    fc_dim = cfg["models"][name]["fc_dim"]
    with torch.no_grad():
        o = model(_mel_dummy(cfg))
    assert o["logits"].shape == (2, num_classes)
    assert o["features"].shape == (2, fc_dim)
