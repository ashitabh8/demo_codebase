"""
Multimodal supervised and pretrain dataloaders.

Supervised loading is delegated to dataloader_factory.create_dataloaders (YAML router).
Core dataset and helpers live in multimodal_core.py and are re-exported here
for backward compatibility with tests and scripts.
"""

import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset_utils.dataloader_factory import create_dataloaders
from dataset_utils.multimodal_core import (
    MultiModalDataset,
    SubsetClassDataset,
    balance_background_indices,
    build_multilabel_distance_target,
    flatten_distance_map,
    present_class_names_from_label,
    resolve_balance_background,
    single_label_to_class_index_tensor,
)

# Deprecated aliases for tests
_balance_background_indices = balance_background_indices
_resolve_balance_background = resolve_balance_background


# ============================================================================
# SSL Pretraining Dataset and Dataloader
# ============================================================================


class PretrainDataset(Dataset):
    """
    Dataset for SSL pretraining. Returns (data, label, idx).
    label is extracted from sample['label'] via .item() for use in visualization only.
    Labels are never used in the SSL loss computation.
    """

    def __init__(self, sample_paths):
        self.sample_files = sample_paths

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample = torch.load(self.sample_files[idx], weights_only=False)
        data = sample["data"]
        label = sample["label"]
        if isinstance(label, torch.Tensor):
            label = label.item()
        label = torch.tensor(label, dtype=torch.long)
        return data, label, idx


def create_pretrain_dataloader(config):
    """
    Builds a dataloader from pretrain_index_file with optional global ratio subset.
    No val/test splits. No balanced sampling. No normalization.
    Labels are returned for visualization only, not used in loss.

    Args:
        config: Full config dict. Reads keys:
            pretrain_index_file (required): path to text file listing .pt sample paths
            pretrain_subset_ratio (float, default 1.0): fraction of samples to use
            pretrain_subset_mode (str, default "global"): "global" implemented; "stratified" raises NotImplementedError
            pretrain_seed (int, default 42): RNG seed for reproducible subset selection
            batch_size_pretrain (int): batch size for pretraining; falls back to batch_size if not set
            num_workers (int, default 4): number of worker processes

    Returns:
        DataLoader with drop_last=True (required for NT-Xent consistent batch size)
    """
    pretrain_index_file = config["pretrain_index_file"]
    if not pretrain_index_file or not os.path.exists(pretrain_index_file):
        raise FileNotFoundError(
            f"pretrain_index_file not found: {pretrain_index_file}"
        )

    experiment_name = config["experiment_name"]
    experiment_config = config["experiments"][experiment_name]
    training_config_name = experiment_config["training"]
    training_configs = config["training_configs"]

    subset_ratio = training_configs[training_config_name][
        "pretrain_subset_ratio"
    ]
    subset_mode = training_configs[training_config_name]["pretrain_subset_mode"]
    seed = training_configs[training_config_name]["pretrain_seed"]
    batch_size = config["batch_size_pretrain"]
    num_workers = config["num_workers"]

    all_paths = list(np.loadtxt(pretrain_index_file, dtype=str))

    if subset_ratio < 1.0:
        if subset_mode == "global":
            rng = np.random.default_rng(seed)
            n = max(1, int(len(all_paths) * subset_ratio))
            idx = rng.choice(len(all_paths), size=n, replace=False)
            idx.sort()
            all_paths = [all_paths[i] for i in idx]
        elif subset_mode == "stratified":
            raise NotImplementedError(
                "Stratified pretrain subset is configured but not implemented yet."
            )
        else:
            raise ValueError(f"Unknown pretrain_subset_mode: {subset_mode}")

    dataset = PretrainDataset(all_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logging.info(
        f"Pretrain dataset: {len(dataset)} samples, {len(loader)} batches (batch_size={batch_size}, drop_last=True)"
    )
    return loader
