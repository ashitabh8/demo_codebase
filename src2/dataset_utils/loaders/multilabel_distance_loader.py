"""Per-class binary targets from class names + distance threshold."""

import logging

import torch

from dataset_utils.multimodal_core import balance_background_indices, resolve_balance_background
from dataset_utils.loaders.loader_common import (
    assemble_supervised_dataloaders,
    make_multimodal_triple,
)


def create_dataloaders(config, task_config, dl_cfg):
    """
    dl_cfg must contain:
      type: multilabel_distance
      distance_threshold_m: float
      distance_key: str

    Task must set num_classes, class_names matching num_classes, and index files.
    """
    if dl_cfg["type"] != "multilabel_distance":
        raise ValueError(
            f"multilabel_distance_loader expected type 'multilabel_distance', got {dl_cfg['type']!r}"
        )
    for req in ("distance_threshold_m", "distance_key"):
        if req not in dl_cfg:
            raise ValueError(
                f"dataloader_configs multilabel_distance must set '{req}' explicitly"
            )
    allowed_keys = {"type", "distance_threshold_m", "distance_key"}
    extra = set(dl_cfg.keys()) - allowed_keys
    if extra:
        raise ValueError(
            f"dataloader_configs multilabel_distance has unexpected keys: {sorted(extra)}"
        )

    if "num_classes" not in task_config:
        raise ValueError("Task config must explicitly set num_classes")
    cn = task_config["class_names"]
    if not isinstance(cn, list) or len(cn) == 0:
        raise ValueError(
            "multilabel_distance requires non-empty class_names in the task config"
        )
    nc = task_config["num_classes"]
    if nc != len(cn):
        raise ValueError(
            f"num_classes ({nc}) must equal len(class_names) ({len(cn)}) "
            "for multilabel_distance"
        )

    logging.info("Creating datasets (multilabel_distance)...")
    ds_kw = {
        "num_classes": nc,
        "multilabel_distance_targets": True,
        "single_label_only": False,
        "class_names": cn,
        "distance_threshold_m": float(dl_cfg["distance_threshold_m"]),
        "distance_key": str(dl_cfg["distance_key"]),
    }

    train_ds, val_ds, test_ds = make_multimodal_triple(task_config, ds_kw)

    if resolve_balance_background(config):
        kept = balance_background_indices(train_ds)
        train_ds = torch.utils.data.Subset(train_ds, kept)
        logging.info(
            "After background balancing: %d training samples",
            len(train_ds),
        )

    return assemble_supervised_dataloaders(
        config,
        train_ds,
        val_ds,
        test_ds,
        is_multilabel_distance=True,
    )
