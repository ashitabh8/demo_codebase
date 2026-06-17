"""
YAML-driven dataloader factory.

experiments.<name>.dataloader must name a key in dataloader_configs (fail-fast).
Each dataloader_configs entry must set type to one of the registered builders.
"""

import logging

from dataset_utils.loaders import legacy_multiclass_loader
from dataset_utils.loaders import multilabel_distance_loader
from dataset_utils.loaders import single_label_loader
from dataset_utils.loaders import single_label_seismic_loader

_BUILDERS = {
    "legacy_multiclass": legacy_multiclass_loader.create_dataloaders,
    "multilabel_distance": multilabel_distance_loader.create_dataloaders,
    "single_label_only": single_label_loader.create_dataloaders,
    "single_label_seismic_only": single_label_seismic_loader.create_dataloaders,
}


def _resolve_dataloader_entry(config):
    experiment_name = config["experiment_name"]
    if "experiments" not in config:
        raise KeyError("config must contain top-level 'experiments'")
    experiments = config["experiments"]
    if experiment_name not in experiments:
        available = [k for k in experiments if k != "enabled"]
        raise ValueError(
            f"Experiment '{experiment_name}' not found in config['experiments']. "
            f"Available: {available}"
        )
    exp_cfg = experiments[experiment_name]
    if "dataloader" not in exp_cfg:
        raise ValueError(
            f"experiments['{experiment_name}'] must set 'dataloader' to a key in "
            f"dataloader_configs (explicit selection required; no default)."
        )
    dl_key = exp_cfg["dataloader"]
    if "dataloader_configs" not in config:
        raise KeyError(
            "config must contain top-level 'dataloader_configs' when using "
            "experiment dataloader routing"
        )
    dataloader_configs = config["dataloader_configs"]
    if dl_key not in dataloader_configs:
        allowed = sorted(k for k in dataloader_configs.keys())
        raise ValueError(
            f"experiments['{experiment_name}']['dataloader'] is '{dl_key}' but that key "
            f"is not in dataloader_configs. Allowed dataloader config keys: {allowed}"
        )
    dl_cfg = dataloader_configs[dl_key]
    if "type" not in dl_cfg:
        raise ValueError(
            f"dataloader_configs['{dl_key}'] must set 'type' to one of: "
            f"{sorted(_BUILDERS.keys())}"
        )
    return dl_key, dl_cfg


def create_dataloaders(config):
    """
    Create train, validation, and test dataloaders from configuration.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logging.info("\nCreating dataloaders...")

    required_top_level = ["batch_size", "num_workers", "use_balanced_sampling"]
    for key in required_top_level:
        if key not in config:
            raise ValueError(f"Missing required config key '{key}'")

    task_name = config["task_name"]
    if task_name not in config:
        raise KeyError(
            f"task_name '{task_name}' not found in config (expected task block at top level)"
        )
    task_config = config[task_name]

    dl_key, dl_cfg = _resolve_dataloader_entry(config)
    dl_type = dl_cfg["type"]
    if dl_type not in _BUILDERS:
        raise ValueError(
            f"dataloader_configs['{dl_key}']['type'] is '{dl_type}'. "
            f"Must be one of: {sorted(_BUILDERS.keys())}"
        )

    logging.info(
        "Using dataloader config '%s' (type=%s)",
        dl_key,
        dl_type,
    )

    builder = _BUILDERS[dl_type]
    return builder(config, task_config, dl_cfg)
