"""
Core multimodal dataset, label helpers, and subset utilities.

Used by dataloader builders in dataset_utils/loaders/ and re-exported from
MultiModalDataLoader for backward compatibility.
"""

import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def flatten_distance_map(obj):
    """
    Extract class_name -> distance (float) from a distance dict.

    Supports two formats only:
      - flat:   {"warthog": 5.0, ...}
      - nested: {"shake": {"warthog": 5.0, ...}, ...}

    Raises TypeError if obj is not a dict.
    """
    if not isinstance(obj, dict):
        raise TypeError(f"Expected distance dict, got {type(obj)}")
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            for inner_k, inner_v in v.items():
                out[str(inner_k)] = float(inner_v)
        else:
            out[str(k)] = float(v)
    return out


def present_class_names_from_label(label, class_names):
    """
    Return the set of class name strings present in a sample label.

    Expected format: np.ndarray with dtype=object containing class name strings,
    e.g. np.array(["polaris", "warthog"], dtype=object).

    Also accepts a list of strings or integer indices into class_names for
    convenience in tests.

    Raises TypeError for any other format.
    """
    cn = list(class_names)

    if isinstance(label, np.ndarray):
        if label.dtype != object:
            raise TypeError(
                f"Label ndarray must have dtype=object (got {label.dtype}). "
                "Store class name strings, not numeric indices."
            )
        return {str(x) for x in label.flat if str(x).strip()}

    if isinstance(label, (list, tuple)):
        names = set()
        for x in label:
            if isinstance(x, (int, np.integer)):
                names.add(cn[int(x)])
            else:
                names.add(str(x))
        return names

    raise TypeError(
        f"Unsupported label type {type(label)}. "
        "Expected np.ndarray with dtype=object."
    )


def build_multilabel_distance_target(
    class_names, present_names, distance_map, threshold_m
):
    """
    Per-class binary target: 1.0 if class is in present_names, has a distance entry,
    and distance < threshold_m; else 0.0.

    If the sample lists a class but it is missing from distance_map, that class is 0.0.
    """
    num = len(class_names)
    t = torch.zeros(num, dtype=torch.float32)
    dist = distance_map if isinstance(distance_map, dict) else {}
    for i, c in enumerate(class_names):
        if c not in present_names:
            continue
        if c not in dist:
            continue
        if dist[c] < threshold_m:
            t[i] = 1.0
    return t


def single_label_cardinality(label):
    """Return how many labels are present in a raw sample label."""
    if isinstance(label, np.ndarray):
        return int(label.size)
    if isinstance(label, (list, tuple)):
        return len(label)
    if isinstance(label, torch.Tensor):
        if label.ndim == 0:
            return 1
        return int(label.numel())
    if isinstance(label, (str, int, np.integer)):
        return 1
    raise TypeError(f"Unsupported label type {type(label)}")


def single_label_to_class_index_tensor(label, class_names):
    """
    Convert a single-label raw value to class-index tensor for CE training.

    Supports numeric labels directly. For string labels, class_names must be
    provided and the label must exist in class_names.
    """
    if isinstance(label, torch.Tensor):
        if label.ndim == 0 or label.numel() == 1:
            return torch.tensor(int(label.reshape(-1)[0].item()), dtype=torch.long)
        raise ValueError(
            f"Expected a single label, got tensor with shape {tuple(label.shape)}"
        )

    if isinstance(label, np.ndarray):
        if int(label.size) != 1:
            raise ValueError(
                f"Expected a single label, got ndarray with size {int(label.size)}"
            )
        return single_label_to_class_index_tensor(label.flat[0], class_names)

    if isinstance(label, (list, tuple)):
        if len(label) != 1:
            raise ValueError(
                f"Expected a single label, got sequence of length {len(label)}"
            )
        return single_label_to_class_index_tensor(label[0], class_names)

    if isinstance(label, (int, np.integer)):
        return torch.tensor(int(label), dtype=torch.long)

    if isinstance(label, str):
        if class_names is None:
            raise ValueError(
                "String labels require class_names to map class name -> class index"
            )
        class_to_idx = {str(name): i for i, name in enumerate(class_names)}
        if label not in class_to_idx:
            raise ValueError(
                f"Label '{label}' not found in class_names: {list(class_names)}"
            )
        return torch.tensor(class_to_idx[label], dtype=torch.long)

    raise TypeError(
        f"Unsupported label type {type(label)} for single-label conversion"
    )


class MultiModalDataset(Dataset):
    """
    PyTorch Dataset for multi-modal sensing data (classification).

    Loads individual samples from .pt files as needed (lazy loading) to avoid memory overflow.
    Each .pt file contains a dictionary with 'data' and 'label' keys.

    Data structure:
        - data: dict[location][modality] = Tensor
        - label: dict with classification label key(s)
    """

    def __init__(
        self,
        index_file,
        num_classes=None,
        multilabel_distance_targets=False,
        single_label_only=False,
        class_names=None,
        distance_threshold_m=None,
        distance_key=None,
        label_subkey=None,
    ):
        self.num_classes = num_classes
        self.multilabel_distance_targets = multilabel_distance_targets
        self.single_label_only = single_label_only
        self.class_names = (
            list(class_names) if class_names is not None else None
        )
        self.distance_threshold_m = distance_threshold_m
        self.distance_key = distance_key
        self.label_subkey = label_subkey

        if self.multilabel_distance_targets:
            if not self.class_names:
                raise ValueError(
                    "multilabel_distance_targets requires non-empty class_names"
                )
            if self.distance_threshold_m is None:
                raise ValueError(
                    "multilabel_distance_targets requires distance_threshold_m"
                )
            if not self.distance_key:
                raise ValueError(
                    "multilabel_distance_targets requires distance_key"
                )
        if self.multilabel_distance_targets and self.single_label_only:
            raise ValueError(
                "single_label_only cannot be enabled with multilabel_distance_targets"
            )

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")

        idx_arr = np.loadtxt(index_file, dtype=str, ndmin=1)
        idx_arr = np.atleast_1d(idx_arr)
        self.sample_files = [
            str(x).strip() for x in idx_arr.ravel() if str(x).strip()
        ]
        logging.info(
            f"Initialized dataset with {len(self.sample_files)} samples"
        )

        if self.single_label_only:
            self._filter_to_single_label_samples()

    def _extract_label_field(self, sample, sample_path):
        if "label" not in sample:
            raise KeyError(
                f"Sample missing required top-level 'label' key: {sample_path}"
            )
        raw = sample["label"]
        if isinstance(raw, dict):
            if self.label_subkey is not None:
                if self.label_subkey not in raw:
                    raise KeyError(
                        f"sample['label'] is a dict but missing required key "
                        f"{self.label_subkey!r}. Available keys: {list(raw.keys())}"
                    )
                return raw[self.label_subkey]
            if "label" not in raw:
                raise KeyError(
                    "sample['label'] is a dict but missing required key 'label'. "
                    f"Available keys: {list(raw.keys())}"
                )
            return raw["label"]
        return raw

    def _filter_to_single_label_samples(self):
        kept = []
        dropped = 0
        for sample_path in self.sample_files:
            sample = torch.load(sample_path, weights_only=False)
            raw_label = self._extract_label_field(sample, sample_path)
            n_labels = single_label_cardinality(raw_label)
            if n_labels == 1:
                kept.append(sample_path)
            else:
                dropped += 1
        self.sample_files = kept
        logging.info(
            "single_label_only enabled: kept %d samples, dropped %d multi-label samples",
            len(self.sample_files),
            dropped,
        )

    def compute_sample_weights_for_balanced_sampling(self):
        if self.multilabel_distance_targets:
            raise ValueError(
                "Balanced sampling is not supported when multilabel_distance_targets is enabled"
            )
        if self.num_classes is None:
            raise ValueError(
                "num_classes must be provided for balanced sampling"
            )

        sample_labels = []
        label_count = [0 for _ in range(self.num_classes)]

        logging.info("Computing sample weights for balanced sampling...")
        for idx in range(len(self.sample_files)):
            _, label, _ = self.__getitem__(idx)
            label_idx = label.item() if hasattr(label, "item") else int(label)
            sample_labels.append(label_idx)
            label_count[label_idx] += 1

        self.sample_weights = []
        for sample_label in sample_labels:
            self.sample_weights.append(1.0 / label_count[sample_label])

        logging.info(f"Label distribution: {label_count}")
        logging.info("Sample weights computed for balanced sampling")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample_path = self.sample_files[idx]

        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample file not found: {sample_path}")

        wo = not self.multilabel_distance_targets and not self.single_label_only
        sample = torch.load(sample_path, weights_only=wo)

        data = sample["data"]
        label = self._extract_label_field(sample, sample_path)

        if self.multilabel_distance_targets:
            present = present_class_names_from_label(label, self.class_names)
            if self.distance_key not in sample:
                raise KeyError(
                    f"Sample missing '{self.distance_key}' required for multilabel_distance_targets"
                )
            dist_raw = sample[self.distance_key]
            dist_map = flatten_distance_map(dist_raw)
            label_tensor = build_multilabel_distance_target(
                self.class_names,
                present,
                dist_map,
                self.distance_threshold_m,
            )
            return data, label_tensor, idx

        if self.single_label_only:
            label_tensor = single_label_to_class_index_tensor(
                label, self.class_names
            )
            return data, label_tensor, idx

        return data, label, idx


def balance_background_indices(dataset, seed=42):
    """
    For a multilabel_distance_targets dataset, identify background (all-zeros
    target) vs positive samples, then downsample background so its count is
    the average of single-positive and multi-positive sample counts.

    Returns a sorted list of dataset indices to keep.
    """
    bg_indices = []
    single_pos_indices = []
    multi_pos_indices = []

    logging.info("Scanning training samples for background balancing...")
    for idx in range(len(dataset)):
        _, label, _ = dataset[idx]
        n_positive = int((label > 0).sum().item())
        if n_positive == 0:
            bg_indices.append(idx)
        elif n_positive == 1:
            single_pos_indices.append(idx)
        else:
            multi_pos_indices.append(idx)

    n_single = len(single_pos_indices)
    n_multi = len(multi_pos_indices)
    n_bg = len(bg_indices)
    target_bg = (n_single + n_multi) // 2

    logging.info(
        "Background balancing: %d background, %d single-label, %d multi-label "
        "-> keeping %d background samples",
        n_bg,
        n_single,
        n_multi,
        target_bg,
    )

    if target_bg >= n_bg:
        return list(range(len(dataset)))

    rng = np.random.default_rng(seed)
    kept_bg = rng.choice(bg_indices, size=target_bg, replace=False).tolist()
    return sorted(single_pos_indices + multi_pos_indices + kept_bg)


def resolve_balance_background(config):
    """
    Return whether the current experiment has balance_background enabled.

    Requires experiment_name and experiments[experiment_name] to be present in config.
    Raises KeyError if either is missing.
    Returns False if balance_background key is absent from the experiment config.
    """
    exp_name = config["experiment_name"]
    exp_cfg = config["experiments"][exp_name]
    return bool(exp_cfg.get("balance_background", False))


class SubsetClassDataset(Dataset):
    """
    Wraps a base MultiModalDataset and:
    - keeps only samples whose original label is in allowed_classes
    - remaps labels using label_map (old_idx -> new_idx in 0..K-1)
    - supports balanced sampling via its own compute_sample_weights_for_balanced_sampling()
    """

    def __init__(self, base_dataset, allowed_classes, label_map):
        self.base = base_dataset
        self.allowed = set(allowed_classes)
        self.label_map = label_map

        self.indices = []
        for base_idx in range(len(self.base)):
            _, label, _ = self.base[base_idx]
            label_idx = int(label.item() if hasattr(label, "item") else label)
            if label_idx in self.allowed:
                self.indices.append(base_idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = self.indices[i]
        data, label, _ = self.base[base_idx]
        label_idx = int(label.item() if hasattr(label, "item") else label)

        new_label = self.label_map[label_idx]
        return data, torch.tensor(new_label, dtype=torch.long), base_idx

    def compute_sample_weights_for_balanced_sampling(self):
        K = len(set(self.label_map.values()))
        label_count = [0 for _ in range(K)]
        sample_labels = []

        for i in range(len(self.indices)):
            _, label, _ = self[i]
            label_idx = int(label.item())
            sample_labels.append(label_idx)
            label_count[label_idx] += 1

        self.sample_weights = []
        for lbl in sample_labels:
            self.sample_weights.append(1.0 / label_count[lbl])
