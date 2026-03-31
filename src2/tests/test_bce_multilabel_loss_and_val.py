"""Tests for BCE multilabel loss and validate_multilabel."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "train_test"))

import numpy as np
from loss import BCEWithLogitsMultilabelForDictOutput, get_loss_function
from train_test_utils import validate_multilabel, find_optimal_per_class_thresholds


def test_bce_multilabel_loss_dict_output():
    loss_fn = BCEWithLogitsMultilabelForDictOutput()
    logits = torch.tensor([[0.0, 2.0], [-1.0, 1.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    out = {"logits": logits, "features": torch.zeros(2, 4)}
    loss = loss_fn(out, target)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_get_loss_function_bce_multilabel_pos_weight():
    cfg = {
        "loss_name": "bce_multilabel",
        "bce_pos_weight": [1.0, 2.0, 3.0],
    }
    loss_fn, name = get_loss_function(cfg)
    assert name == "bce_multilabel"
    assert isinstance(loss_fn, BCEWithLogitsMultilabelForDictOutput)


def test_get_loss_function_bce_pos_weight_must_be_list():
    cfg = {"loss_name": "bce_multilabel", "bce_pos_weight": "bad"}
    with pytest.raises(ValueError, match="bce_pos_weight must be a list"):
        get_loss_function(cfg)


class _HighLogitsModel(nn.Module):
    """Constant high logits so sigmoid > 0.5 for all entries."""

    def forward(self, x):
        b = x.shape[0]
        return {"logits": torch.full((b, 3), 10.0, device=x.device, dtype=x.dtype)}


def test_validate_multilabel_perfect_when_targets_all_ones():
    loss_fn = BCEWithLogitsMultilabelForDictOutput()
    x = torch.zeros(4, 1)
    y = torch.ones(4, 3)
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    model = _HighLogitsModel()
    training_config = {}
    r = validate_multilabel(
        model, loader, loss_fn, torch.device("cpu"), training_config
    )
    assert r["mAP"] == 1.0
    assert r["confusion_matrix"] is None
    assert r["raw_probs"].shape == (4, 3)
    assert r["raw_labels"].shape == (4, 3)
    assert np.all(r["raw_probs"] > 0.5)


def test_validate_multilabel_low_logits_zero_targets():
    """Logits negative -> probs near 0; all-zero targets -> mAP undefined (0)."""
    class _LowLogitsModel(nn.Module):
        def forward(self, x):
            b = x.shape[0]
            return {"logits": torch.full((b, 3), -10.0, device=x.device, dtype=x.dtype)}

    loss_fn = BCEWithLogitsMultilabelForDictOutput()
    x = torch.zeros(2, 1)
    y = torch.zeros(2, 3)
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    model = _LowLogitsModel()
    training_config = {}
    r = validate_multilabel(
        model, loader, loss_fn, torch.device("cpu"), training_config
    )
    assert r["raw_probs"].shape == (2, 3)
    assert np.all(r["raw_probs"] < 0.5)
    assert r["mAP"] == 0.0


def test_find_optimal_thresholds_perfect_separation():
    """When probs perfectly separate, optimal threshold should give F1=1."""
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    y_prob = np.array([
        [0.9, 0.1, 0.05],
        [0.1, 0.95, 0.1],
        [0.85, 0.8, 0.05],
        [0.05, 0.1, 0.9],
    ], dtype=np.float64)
    result = find_optimal_per_class_thresholds(
        y_true, y_prob, ["polaris", "warthog", "truck"]
    )
    assert len(result["per_class"]) == 3
    for cls in result["per_class"]:
        assert cls["best_f1"] == 1.0
    assert result["global_macro_f1_at_best"] == 1.0
    assert result["global_subset_acc_at_best"] == 1.0


def test_find_optimal_thresholds_returns_per_class_curves():
    y_true = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
    y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.9]], dtype=np.float64)
    result = find_optimal_per_class_thresholds(
        y_true, y_prob, ["A", "B"]
    )
    assert "best_thresholds" in result
    assert set(result["best_thresholds"].keys()) == {"A", "B"}
    for cls in result["per_class"]:
        assert len(cls["curve"]) > 0
        for pt in cls["curve"]:
            assert "threshold" in pt
            assert "precision" in pt
            assert "recall" in pt
            assert "f1" in pt
