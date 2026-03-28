import torch
import pytest

from train_test.train_test_utils import validate_finetune_config


def _base_finetune_config(checkpoint_path, loss_name):
    return {
        "experiment_name": "finetune_exp",
        "experiments": {
            "enabled": True,
            "finetune_exp": {
                "model": "student_audio_deepsense_dw_large_mel",
                "training": "finetune_cfg",
                "checkpoint_path": str(checkpoint_path),
            },
        },
        "models": {
            "student_audio_deepsense_dw_large_mel": {
                "checkpoint_path": str(checkpoint_path),
            }
        },
        "training_configs": {
            "finetune_cfg": {
                "type": "finetune",
                "epochs": 3,
                "loss_name": loss_name,
                "freeze_backbone": True,
            }
        },
    }


def test_validate_finetune_config_accepts_ce_supcon(tmp_path):
    checkpoint_path = tmp_path / "dummy_checkpoint.pth"
    torch.save({"model_state_dict": {}}, checkpoint_path)

    config = _base_finetune_config(checkpoint_path, loss_name="ce_supcon")
    _, _, _, _, training_cfg, resolved_checkpoint, _ = validate_finetune_config(config)

    assert training_cfg["loss_name"] == "ce_supcon"
    assert resolved_checkpoint == str(checkpoint_path)


def test_validate_finetune_config_rejects_unknown_loss(tmp_path):
    checkpoint_path = tmp_path / "dummy_checkpoint.pth"
    torch.save({"model_state_dict": {}}, checkpoint_path)

    config = _base_finetune_config(checkpoint_path, loss_name="not_a_valid_loss")
    with pytest.raises(ValueError, match="Finetune expects loss_name"):
        validate_finetune_config(config)
