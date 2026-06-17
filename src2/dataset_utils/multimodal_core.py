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


def normalize_sample_data_layout(sample, location_names, loc_modalities):
    """
    Ensure ``sample`` has the nested ``data`` dict expected by loaders and augmenters.

    Parkland-style files already set ``sample['data'][location][modality]`` to tensors.
    West Point-style files may expose flat time-domain modalities such as:
      - ``sample['mic']`` for audio
      - ``sample['geo']`` for seismic/geophone
    In that case, we nest the tensors under the first configured location using the
    configured modality names.

    Mutates ``sample`` in place when wrapping flat ``mic`` layout.
    """
    if "data" in sample:
        return sample
    if "mic" not in sample and "geo" not in sample:
        raise KeyError(
            "Sample dict must contain 'data' (nested location/modality tensors) "
            "or flat modality keys like 'mic'/'geo'. Missing all. "
            f"Keys present: {sorted(sample.keys())!r}"
        )
    if location_names is None or loc_modalities is None:
        raise KeyError(
            "Flat 'mic' samples require config keys location_names and loc_modalities "
            "to build nested 'data'."
        )
    loc = location_names[0]
    if loc not in loc_modalities:
        raise KeyError(
            f"location_names[0] is {loc!r} but loc_modalities has no entry for it. "
            f"loc_modalities keys: {sorted(loc_modalities.keys())!r}"
        )
    mods = loc_modalities[loc]
    mic = None
    if "mic" in sample:
        raw_mic = sample["mic"]
        if isinstance(raw_mic, np.ndarray):
            mic = torch.from_numpy(raw_mic.astype(np.float32, copy=False))
        elif isinstance(raw_mic, torch.Tensor):
            mic = raw_mic.float()
        else:
            mic = torch.as_tensor(raw_mic, dtype=torch.float32)
    geo = None
    if "geo" in sample:
        raw_geo = sample["geo"]
        if isinstance(raw_geo, np.ndarray):
            geo = torch.from_numpy(raw_geo.astype(np.float32, copy=False))
        elif isinstance(raw_geo, torch.Tensor):
            geo = raw_geo.float()
        else:
            geo = torch.as_tensor(raw_geo, dtype=torch.float32)
    inner = {}
    if "audio" in mods and mic is not None:
        inner["audio"] = mic
    if "seismic" in mods and geo is not None:
        inner["seismic"] = geo
    if "geo" in mods and geo is not None:
        inner["geo"] = geo
    if len(inner) == 0:
        if mic is not None:
            inner[mods[0]] = mic
        elif geo is not None:
            inner[mods[0]] = geo
    for m in mods:
        if m not in inner:
            if mic is not None:
                inner[m] = torch.zeros_like(mic)
            elif geo is not None:
                inner[m] = torch.zeros_like(geo)
            else:
                raise RuntimeError("Expected at least one flat modality tensor")
    sample["data"] = {loc: inner}
    return sample


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

    If a sample has zero labels (empty ndarray/list/tuple), this maps it to the
    "background" class index when "background" exists in class_names.
    """
    def _background_index():
        if class_names is None:
            return None
        class_to_idx = {str(name): i for i, name in enumerate(class_names)}
        return class_to_idx.get("background")

    if isinstance(label, torch.Tensor):
        if label.ndim == 0 or label.numel() == 1:
            return torch.tensor(int(label.reshape(-1)[0].item()), dtype=torch.long)
        if int(label.numel()) == 0:
            bg_idx = _background_index()
            if bg_idx is not None:
                return torch.tensor(bg_idx, dtype=torch.long)
            raise ValueError(
                "Got empty tensor label with no 'background' class configured"
            )
        raise ValueError(
            f"Expected a single label, got tensor with shape {tuple(label.shape)}"
        )

    if isinstance(label, np.ndarray):
        if int(label.size) == 0:
            bg_idx = _background_index()
            if bg_idx is not None:
                return torch.tensor(bg_idx, dtype=torch.long)
            raise ValueError(
                "Got empty ndarray label with no 'background' class configured"
            )
        if int(label.size) != 1:
            raise ValueError(
                f"Expected a single label, got ndarray with size {int(label.size)}"
            )
        return single_label_to_class_index_tensor(label.flat[0], class_names)

    if isinstance(label, (list, tuple)):
        if len(label) == 0:
            bg_idx = _background_index()
            if bg_idx is not None:
                return torch.tensor(bg_idx, dtype=torch.long)
            raise ValueError(
                "Got empty sequence label with no 'background' class configured"
            )
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
        - label: dict with classification label key(s), or nested label['label']

    If label is a flat dict (e.g. ACIDS: vehicle_type, terrain, speed, distance),
    set label_subkey on the task config (e.g. vehicle_type) so that field is used
    as the raw label for filtering and training.
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
        location_names=None,
        loc_modalities=None,
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
        self.location_names = (
            list(location_names) if location_names is not None else None
        )
        self.loc_modalities = loc_modalities

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
        if self.multilabel_distance_targets and self.label_subkey is not None:
            raise ValueError(
                "label_subkey is not supported with multilabel_distance_targets"
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
        lab = sample["label"]
        if isinstance(lab, dict):
            if "label" in lab:
                return lab["label"]
            if self.label_subkey is not None:
                if self.label_subkey not in lab:
                    raise KeyError(
                        f"sample['label'] has no key {self.label_subkey!r}. "
                        f"Available keys: {list(lab.keys())}"
                    )
                return lab[self.label_subkey]
            raise KeyError(
                "sample['label'] is a dict but missing required key 'label', "
                "and no label_subkey was configured on the dataset. "
                f"Available keys: {list(lab.keys())}"
            )
        return lab

    def _filter_to_single_label_samples(self):
        kept = []
        dropped_multilabel = 0
        dropped_oov = 0
        allowed = (
            {str(c) for c in self.class_names}
            if self.class_names is not None
            else None
        )
        for sample_path in self.sample_files:
            sample = torch.load(sample_path, weights_only=False)
            raw_label = self._extract_label_field(sample, sample_path)
            n_labels = single_label_cardinality(raw_label)
            if n_labels != 1:
                dropped_multilabel += 1
                continue
            if allowed is not None and isinstance(raw_label, str):
                if raw_label not in allowed:
                    dropped_oov += 1
                    continue
            kept.append(sample_path)
        self.sample_files = kept
        logging.info(
            "single_label_only enabled: kept %d samples, dropped %d multi-label, "
            "dropped %d samples with label not in class_names",
            len(self.sample_files),
            dropped_multilabel,
            dropped_oov,
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
        normalize_sample_data_layout(
            sample, self.location_names, self.loc_modalities
        )

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
