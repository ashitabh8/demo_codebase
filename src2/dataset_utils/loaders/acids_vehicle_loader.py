"""ACIDS .pt layout: sample['label'] is a dict; vehicle class is at label[label_subkey]."""

import logging

from dataset_utils.loaders.loader_common import (
    assemble_supervised_dataloaders,
    make_multimodal_triple,
)


def create_dataloaders(config, task_config, dl_cfg):
    """
    dl_cfg must contain:
      type: acids_vehicle_classification
      label_subkey: str  — e.g. vehicle_type for ACIDS vehicle_classification .pt files

    Task must set num_classes, class_names, and index files.
    """
    if dl_cfg["type"] != "acids_vehicle_classification":
        raise ValueError(
            f"acids_vehicle_loader expected type 'acids_vehicle_classification', "
            f"got {dl_cfg['type']!r}"
        )
    if "label_subkey" not in dl_cfg:
        raise ValueError(
            "dataloader_configs entry for acids_vehicle_classification must set "
            "'label_subkey' (e.g. vehicle_type)"
        )
    allowed_keys = {"type", "label_subkey"}
    extra = [k for k in dl_cfg if k not in allowed_keys]
    if extra:
        raise ValueError(
            f"dataloader_configs entry for acids_vehicle_classification only allows "
            f"{sorted(allowed_keys)}; unexpected keys: {extra}"
        )

    if "num_classes" not in task_config:
        raise ValueError("Task config must explicitly set num_classes")
    cn = task_config["class_names"]
    if not isinstance(cn, list) or len(cn) == 0:
        raise ValueError(
            "acids_vehicle_classification requires non-empty class_names in the task config"
        )
    nc = task_config["num_classes"]
    if nc != len(cn):
        raise ValueError(
            f"num_classes ({nc}) must equal len(class_names) ({len(cn)}) "
            "for acids_vehicle_classification"
        )

    subkey = dl_cfg["label_subkey"]
    logging.info(
        "Creating datasets (acids_vehicle_classification, label_subkey=%r)...",
        subkey,
    )
    ds_kw = {
        "num_classes": nc,
        "multilabel_distance_targets": False,
        "single_label_only": True,
        "class_names": cn,
        "label_subkey": subkey,
    }

    train_ds, val_ds, test_ds = make_multimodal_triple(task_config, ds_kw)

    return assemble_supervised_dataloaders(
        config,
        train_ds,
        val_ds,
        test_ds,
        is_multilabel_distance=False,
    )
