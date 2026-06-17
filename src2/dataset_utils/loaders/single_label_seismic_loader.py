"""Single-label seismic-only loader (copied from single_label_loader)."""

import logging

from dataset_utils.loaders.loader_common import (
    assemble_supervised_dataloaders,
    make_multimodal_triple,
)


def create_dataloaders(config, task_config, dl_cfg):
    """
    dl_cfg must contain:
      type: single_label_seismic_only

    Task must set num_classes, class_names (non-empty list), and index files.
    """
    if dl_cfg["type"] != "single_label_seismic_only":
        raise ValueError(
            "single_label_seismic_loader expected type "
            f"'single_label_seismic_only', got {dl_cfg['type']!r}"
        )
    if len(dl_cfg) != 1:
        extra = [k for k in dl_cfg if k != "type"]
        raise ValueError(
            "dataloader_configs entry for single_label_seismic_only must only "
            f"contain 'type'; unexpected keys: {extra}"
        )

    if "num_classes" not in task_config:
        raise ValueError("Task config must explicitly set num_classes")
    cn = task_config["class_names"]
    if not isinstance(cn, list) or len(cn) == 0:
        raise ValueError(
            "single_label_seismic_only requires non-empty class_names in the task config"
        )
    nc = task_config["num_classes"]
    if nc != len(cn):
        raise ValueError(
            f"num_classes ({nc}) must equal len(class_names) ({len(cn)}) "
            "for single_label_seismic_only"
        )

    if "loc_modalities" not in config or "shake" not in config["loc_modalities"]:
        raise ValueError(
            "single_label_seismic_only requires config['loc_modalities']['shake']"
        )
    if "seismic" not in config["loc_modalities"]["shake"]:
        raise ValueError(
            "single_label_seismic_only requires 'seismic' in loc_modalities['shake']"
        )

    logging.info("Creating datasets (single_label_seismic_only)...")
    ds_kw = {
        "num_classes": nc,
        "multilabel_distance_targets": False,
        "single_label_only": True,
        "class_names": cn,
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
