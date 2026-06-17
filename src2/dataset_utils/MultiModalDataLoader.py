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
    normalize_sample_data_layout,
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
    label is converted to a class index for visualization only (not used in NT-Xent).
    Supports ACIDS flat dict labels via label_subkey + class_names.
    """

    def __init__(
        self,
        sample_paths,
        location_names=None,
        loc_modalities=None,
        label_subkey=None,
        class_names=None,
    ):
        self.sample_files = sample_paths
        self.location_names = (
            list(location_names) if location_names is not None else None
        )
        self.loc_modalities = loc_modalities
        self.label_subkey = label_subkey
        self.class_names = (
            list(class_names) if class_names is not None else None
        )

    def _label_for_viz(self, sample, sample_path):
        if "label" not in sample:
            return torch.tensor(0, dtype=torch.long)
        lab = sample["label"]
        if isinstance(lab, dict):
            if "label" in lab:
                raw = lab["label"]
            elif self.label_subkey is not None:
                if self.label_subkey not in lab:
                    raise KeyError(
                        f"sample['label'] has no key {self.label_subkey!r} in {sample_path}. "
                        f"Available: {list(lab.keys())}"
                    )
                raw = lab[self.label_subkey]
            else:
                return torch.tensor(0, dtype=torch.long)
            if self.class_names is not None:
                return single_label_to_class_index_tensor(raw, self.class_names)
            return torch.tensor(0, dtype=torch.long)
        if isinstance(lab, torch.Tensor):
            if lab.ndim == 0 or lab.numel() == 1:
                return torch.tensor(int(lab.reshape(-1)[0].item()), dtype=torch.long)
            return torch.tensor(0, dtype=torch.long)
        return torch.tensor(int(lab), dtype=torch.long)

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample_path = self.sample_files[idx]
        sample = torch.load(sample_path, weights_only=False)
        normalize_sample_data_layout(
            sample, self.location_names, self.loc_modalities
        )
        data = sample["data"]
        label = self._label_for_viz(sample, sample_path)
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

    loc_names = config["location_names"] if "location_names" in config else None
    loc_mods = config["loc_modalities"] if "loc_modalities" in config else None
    label_subkey = None
    class_names = None
    if "task_name" in config and config["task_name"] in config:
        task_cfg = config[config["task_name"]]
        if "label_subkey" in task_cfg:
            label_subkey = task_cfg["label_subkey"]
        if "class_names" in task_cfg:
            class_names = task_cfg["class_names"]
    elif "vehicle_classification" in config:
        vc = config["vehicle_classification"]
        if "label_subkey" in vc:
            label_subkey = vc["label_subkey"]
        if "class_names" in vc:
            class_names = vc["class_names"]
    dataset = PretrainDataset(
        all_paths,
        location_names=loc_names,
        loc_modalities=loc_mods,
        label_subkey=label_subkey,
        class_names=class_names,
    )
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
