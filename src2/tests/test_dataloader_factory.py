"""Fail-fast resolution for experiment -> dataloader_configs routing."""

import pytest

from dataset_utils.dataloader_factory import create_dataloaders


def _minimal_supervised_config():
    return {
        "batch_size": 2,
        "num_workers": 0,
        "use_balanced_sampling": False,
        "experiment_name": "exp1",
        "task_name": "my_task",
        "my_task": {
            "num_classes": 2,
            "class_names": ["a", "b"],
            "train_index_file": "/nonexistent/train.txt",
            "val_index_file": "/nonexistent/val.txt",
            "test_index_file": "/nonexistent/test.txt",
        },
    }


def test_create_dataloaders_raises_without_experiment_dataloader():
    config = _minimal_supervised_config()
    config["experiments"] = {"exp1": {"model": "m", "training": "t"}}
    config["dataloader_configs"] = {"dl": {"type": "legacy_multiclass"}}
    with pytest.raises(ValueError, match="must set 'dataloader'"):
        create_dataloaders(config)


def test_create_dataloaders_raises_unknown_dataloader_key():
    config = _minimal_supervised_config()
    config["experiments"] = {"exp1": {"dataloader": "missing_key"}}
    config["dataloader_configs"] = {"other": {"type": "legacy_multiclass"}}
    with pytest.raises(ValueError, match="not in dataloader_configs"):
        create_dataloaders(config)


def test_create_dataloaders_raises_without_dataloader_type():
    config = _minimal_supervised_config()
    config["experiments"] = {"exp1": {"dataloader": "dl"}}
    config["dataloader_configs"] = {"dl": {}}
    with pytest.raises(ValueError, match="must set 'type'"):
        create_dataloaders(config)


def test_create_dataloaders_raises_unknown_type():
    config = _minimal_supervised_config()
    config["experiments"] = {"exp1": {"dataloader": "dl"}}
    config["dataloader_configs"] = {"dl": {"type": "not_a_real_loader"}}
    with pytest.raises(ValueError, match="Must be one of"):
        create_dataloaders(config)
