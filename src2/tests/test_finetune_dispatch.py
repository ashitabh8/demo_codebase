import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "train_test"))

import train_test.finetune as finetune_module


def test_select_finetune_train_fn_uses_supcon_loop_for_ce_supcon():
    train_fn = finetune_module.select_finetune_train_fn("ce_supcon")
    assert train_fn is finetune_module.train_vanilla_supervised_contrastive


def test_select_finetune_train_fn_uses_standard_loop_for_cross_entropy():
    train_fn = finetune_module.select_finetune_train_fn("cross_entropy")
    assert train_fn is finetune_module.train


def test_select_finetune_train_fn_uses_standard_loop_for_bce_multilabel():
    train_fn = finetune_module.select_finetune_train_fn("bce_multilabel")
    assert train_fn is finetune_module.train


def test_select_finetune_train_fn_rejects_unknown_loss():
    with pytest.raises(ValueError, match="Unsupported finetune loss_name"):
        finetune_module.select_finetune_train_fn("mystery_loss")
