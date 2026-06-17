"""Shared validation and DataLoader assembly for supervised multimodal splits."""

import logging
import os

import torch
from torch.utils.data import DataLoader

from dataset_utils.multimodal_core import MultiModalDataset, SubsetClassDataset


def validate_task_index_files(task_config):
    train_index_file = task_config["train_index_file"]
    val_index_file = task_config["val_index_file"]
    test_index_file = task_config["test_index_file"]

    if not train_index_file or not os.path.exists(train_index_file):
        raise FileNotFoundError(
            f"Train index file not found: {train_index_file}"
        )
    if not val_index_file or not os.path.exists(val_index_file):
        raise FileNotFoundError(f"Val index file not found: {val_index_file}")
    if not test_index_file or not os.path.exists(test_index_file):
        raise FileNotFoundError(f"Test index file not found: {test_index_file}")

    return train_index_file, val_index_file, test_index_file


def assemble_supervised_dataloaders(
    config,
    train_dataset,
    val_dataset,
    test_dataset,
    *,
    is_multilabel_distance,
):
    """
    Apply optional class subset, balanced sampling, and build DataLoaders.

    Batches are always (data, labels, idx).
    """
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    use_balanced_sampling = config["use_balanced_sampling"]

    if use_balanced_sampling and is_multilabel_distance:
        raise ValueError(
            "use_balanced_sampling cannot be used with multilabel distance "
            "targets; set use_balanced_sampling to false in config for this task"
        )

    include = config.get("include_classes")
    label_map = config.get("include_classes_mapping")
    if include and label_map:
        if is_multilabel_distance:
            raise ValueError(
                "include_classes / include_classes_mapping is not supported with "
                "multilabel distance targets in this codebase version"
            )
        train_dataset = SubsetClassDataset(train_dataset, include, label_map)
        val_dataset = SubsetClassDataset(val_dataset, include, label_map)
        test_dataset = SubsetClassDataset(test_dataset, include, label_map)

    logging.info("Creating dataloaders...")

    if use_balanced_sampling:
        train_dataset.compute_sample_weights_for_balanced_sampling()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        logging.info("Using balanced sampling for training")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logging.info(
        f"Train samples: {len(train_dataset)}, batches: {len(train_loader)}"
    )
    logging.info(f"Val samples: {len(val_dataset)}, batches: {len(val_loader)}")
    logging.info(
        f"Test samples: {len(test_dataset)}, batches: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader


def make_multimodal_triple(
    task_config,
    dataset_kwargs,
    *,
    location_names=None,
    loc_modalities=None,
):
    train_f, val_f, test_f = validate_task_index_files(task_config)
    kw = dict(dataset_kwargs)
    if "label_subkey" in task_config:
        kw["label_subkey"] = task_config["label_subkey"]
    if location_names is not None:
        kw["location_names"] = list(location_names)
    if loc_modalities is not None:
        kw["loc_modalities"] = loc_modalities
    train_ds = MultiModalDataset(index_file=train_f, **kw)
    val_ds = MultiModalDataset(index_file=val_f, **kw)
    test_ds = MultiModalDataset(index_file=test_f, **kw)
    return train_ds, val_ds, test_ds
