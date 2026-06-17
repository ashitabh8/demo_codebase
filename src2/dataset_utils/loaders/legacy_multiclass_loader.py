"""Raw-label multiclass: return labels as stored in .pt (weights_only load)."""

import logging

from dataset_utils.loaders.loader_common import (
    assemble_supervised_dataloaders,
    make_multimodal_triple,
)


def create_dataloaders(config, task_config, dl_cfg):
    """
    dl_cfg must contain:
      type: legacy_multiclass

    No extra keys. Task must set num_classes and index files; class_names optional.
    """
    if dl_cfg["type"] != "legacy_multiclass":
        raise ValueError(
            f"legacy_multiclass_loader expected type 'legacy_multiclass', got {dl_cfg['type']!r}"
        )
    if len(dl_cfg) != 1:
        extra = [k for k in dl_cfg if k != "type"]
        raise ValueError(
            f"dataloader_configs entry for legacy_multiclass must only contain 'type'; "
            f"unexpected keys: {extra}"
        )

    if "num_classes" not in task_config:
        raise ValueError("Task config must explicitly set num_classes")

    logging.info("Creating datasets (legacy_multiclass)...")
    ds_kw = {
        "num_classes": task_config["num_classes"],
        "multilabel_distance_targets": False,
        "single_label_only": False,
    }

    loc_names = config["location_names"] if "location_names" in config else None
    loc_mods = config["loc_modalities"] if "loc_modalities" in config else None
    train_ds, val_ds, test_ds = make_multimodal_triple(
        task_config,
        ds_kw,
        location_names=loc_names,
        loc_modalities=loc_mods,
    )

    return assemble_supervised_dataloaders(
        config,
        train_ds,
        val_ds,
        test_ds,
        is_multilabel_distance=False,
    )
