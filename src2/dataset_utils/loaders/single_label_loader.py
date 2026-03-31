"""Single-label only: drop multi-label samples; map string labels to class indices."""

import logging

from dataset_utils.loaders.loader_common import (
    assemble_supervised_dataloaders,
    make_multimodal_triple,
)


def create_dataloaders(config, task_config, dl_cfg):
    """
    dl_cfg must contain:
      type: single_label_only

    Task must set num_classes, class_names (non-empty list), and index files.
    """
    if dl_cfg["type"] != "single_label_only":
        raise ValueError(
            f"single_label_loader expected type 'single_label_only', got {dl_cfg['type']!r}"
        )
    if len(dl_cfg) != 1:
        extra = [k for k in dl_cfg if k != "type"]
        raise ValueError(
            f"dataloader_configs entry for single_label_only must only contain 'type'; "
            f"unexpected keys: {extra}"
        )

    if "num_classes" not in task_config:
        raise ValueError("Task config must explicitly set num_classes")
    cn = task_config["class_names"]
    if not isinstance(cn, list) or len(cn) == 0:
        raise ValueError(
            "single_label_only requires non-empty class_names in the task config"
        )
    nc = task_config["num_classes"]
    if nc != len(cn):
        raise ValueError(
            f"num_classes ({nc}) must equal len(class_names) ({len(cn)}) "
            "for single_label_only"
        )

    logging.info("Creating datasets (single_label_only)...")
    ds_kw = {
        "num_classes": nc,
        "multilabel_distance_targets": False,
        "single_label_only": True,
        "class_names": cn,
    }

    train_ds, val_ds, test_ds = make_multimodal_triple(task_config, ds_kw)

    return assemble_supervised_dataloaders(
        config,
        train_ds,
        val_ds,
        test_ds,
        is_multilabel_distance=False,
    )
