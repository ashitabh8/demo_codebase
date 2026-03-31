"""Tests for multilabel distance targets in MultiModalDataset."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dataset_utils.MultiModalDataLoader import (
    MultiModalDataset,
    _balance_background_indices,
    _resolve_balance_background,
    build_multilabel_distance_target,
    create_dataloaders,
    flatten_distance_map,
    present_class_names_from_label,
    single_label_to_class_index_tensor,
)


CLASS_NAMES = ["polaris", "warthog", "truck"]
THRESHOLD = 10.0


def test_flatten_distance_map_nested():
    d = {"shake": {"warthog": 5.5}}
    assert flatten_distance_map(d) == {"warthog": 5.5}


def test_flatten_distance_map_flat():
    assert flatten_distance_map({"warthog": 17.9}) == {"warthog": 17.9}


def test_build_multilabel_distance_target_all_far():
    present = {"warthog"}
    dist = {"warthog": 15.0}
    t = build_multilabel_distance_target(CLASS_NAMES, present, dist, THRESHOLD)
    assert t.shape == (3,)
    assert torch.allclose(t, torch.zeros(3))


def test_build_multilabel_distance_target_near_single():
    present = {"warthog"}
    dist = {"warthog": 5.0}
    t = build_multilabel_distance_target(CLASS_NAMES, present, dist, THRESHOLD)
    expected = torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(t, expected)


def test_build_multilabel_distance_target_mixed_two_vehicles():
    present = {"polaris", "warthog"}
    dist = {"polaris": 5.0, "warthog": 20.0}
    t = build_multilabel_distance_target(CLASS_NAMES, present, dist, THRESHOLD)
    expected = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(t, expected)


def test_build_multilabel_distance_target_both_near():
    present = {"polaris", "warthog"}
    dist = {"polaris": 5.0, "warthog": 8.0}
    t = build_multilabel_distance_target(CLASS_NAMES, present, dist, THRESHOLD)
    expected = torch.tensor([1.0, 1.0, 0.0])
    assert torch.allclose(t, expected)


def test_build_missing_distance_entry_is_zero():
    present = {"warthog"}
    dist = {}
    t = build_multilabel_distance_target(CLASS_NAMES, present, dist, THRESHOLD)
    assert torch.allclose(t, torch.zeros(3))


def test_present_class_names_numpy_object_array():
    arr = np.array(["polaris", "warthog"], dtype=object)
    s = present_class_names_from_label(arr, CLASS_NAMES)
    assert s == {"polaris", "warthog"}


def test_present_class_names_int_indices():
    s = present_class_names_from_label([0, 2], CLASS_NAMES)
    assert s == {"polaris", "truck"}


@pytest.fixture()
def tmp_index_and_samples(tmp_path):
    """Two tiny .pt samples and an index file listing their paths."""
    p1 = tmp_path / "a.pt"
    p2 = tmp_path / "b.pt"
    idx_path = tmp_path / "index.txt"

    torch.save(
        {
            "data": {"shake": {"audio": torch.zeros(1, 1, 1)}},
            "label": np.array(["warthog"], dtype=object),
            "distance": {"warthog": 5.0},
        },
        p1,
    )
    torch.save(
        {
            "data": {"shake": {"audio": torch.zeros(1, 1, 1)}},
            "label": np.array(["polaris", "warthog"], dtype=object),
            "distance": {"polaris": 5.0, "warthog": 25.0},
        },
        p2,
    )
    idx_path.write_text(f"{p1}\n{p2}\n")
    return str(idx_path)


def test_multimodal_dataset_returns_float_vector(tmp_index_and_samples):
    ds = MultiModalDataset(
        index_file=tmp_index_and_samples,
        num_classes=3,
        multilabel_distance_targets=True,
        class_names=CLASS_NAMES,
        distance_threshold_m=THRESHOLD,
        distance_key="distance",
    )
    _, y0, _ = ds[0]
    assert y0.dtype == torch.float32
    assert torch.allclose(y0, torch.tensor([0.0, 1.0, 0.0]))

    _, y1, _ = ds[1]
    assert torch.allclose(y1, torch.tensor([1.0, 0.0, 0.0]))


def test_multimodal_dataset_legacy_returns_raw_label(tmp_path):
    p = tmp_path / "one.pt"
    torch.save(
        {
            "data": {"shake": {"audio": torch.zeros(1)}},
            "label": torch.tensor(2, dtype=torch.long),
        },
        p,
    )
    idx = tmp_path / "index.txt"
    idx.write_text(f"{p}\n")
    ds = MultiModalDataset(
        index_file=str(idx),
        num_classes=3,
        multilabel_distance_targets=False,
    )
    _, lab, _ = ds[0]
    assert int(lab.item()) == 2


def test_single_label_to_class_index_tensor_string_label():
    y = single_label_to_class_index_tensor("warthog", CLASS_NAMES)
    assert int(y.item()) == 1


def test_multimodal_dataset_single_label_only_filters_and_maps(tmp_index_and_samples):
    ds = MultiModalDataset(
        index_file=tmp_index_and_samples,
        num_classes=3,
        multilabel_distance_targets=False,
        single_label_only=True,
        class_names=CLASS_NAMES,
    )
    assert len(ds) == 1
    _, lab, _ = ds[0]
    assert lab.dtype == torch.long
    assert int(lab.item()) == 1


def test_dataloader_batch_shape(tmp_index_and_samples):
    ds = MultiModalDataset(
        index_file=tmp_index_and_samples,
        num_classes=3,
        multilabel_distance_targets=True,
        class_names=CLASS_NAMES,
        distance_threshold_m=THRESHOLD,
        distance_key="distance",
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    _, labels, _ = batch
    assert labels.shape == (2, 3)


def test_multimodal_dataset_raises_without_distance_key(tmp_path):
    p = tmp_path / "x.pt"
    torch.save(
        {
            "data": {"shake": {"audio": torch.zeros(1)}},
            "label": np.array(["warthog"], dtype=object),
        },
        p,
    )
    idx = tmp_path / "index.txt"
    idx.write_text(f"{p}\n")
    ds = MultiModalDataset(
        index_file=str(idx),
        num_classes=3,
        multilabel_distance_targets=True,
        class_names=CLASS_NAMES,
        distance_threshold_m=THRESHOLD,
        distance_key="distance",
    )
    with pytest.raises(KeyError, match="distance"):
        _ = ds[0]


def test_balanced_sampling_rejects_multilabel(tmp_index_and_samples):
    ds = MultiModalDataset(
        index_file=tmp_index_and_samples,
        num_classes=3,
        multilabel_distance_targets=True,
        class_names=CLASS_NAMES,
        distance_threshold_m=THRESHOLD,
        distance_key="distance",
    )
    with pytest.raises(ValueError, match="Balanced sampling"):
        ds.compute_sample_weights_for_balanced_sampling()


# ---------------------------------------------------------------------------
# balance_background tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def bg_heavy_index(tmp_path):
    """
    10 samples: 6 background (all far), 2 single-positive, 2 multi-positive.
    After balancing, target_bg = (2 + 2) // 2 = 2, so 6 kept total.
    """
    paths = []
    for i in range(6):
        p = tmp_path / f"bg_{i}.pt"
        torch.save(
            {
                "data": {"shake": {"audio": torch.zeros(1, 1, 1)}},
                "label": np.array(["warthog"], dtype=object),
                "distance": {"warthog": 50.0},
            },
            p,
        )
        paths.append(str(p))
    for i in range(2):
        p = tmp_path / f"single_{i}.pt"
        torch.save(
            {
                "data": {"shake": {"audio": torch.zeros(1, 1, 1)}},
                "label": np.array(["polaris"], dtype=object),
                "distance": {"polaris": 5.0},
            },
            p,
        )
        paths.append(str(p))
    for i in range(2):
        p = tmp_path / f"multi_{i}.pt"
        torch.save(
            {
                "data": {"shake": {"audio": torch.zeros(1, 1, 1)}},
                "label": np.array(["polaris", "warthog"], dtype=object),
                "distance": {"polaris": 3.0, "warthog": 7.0},
            },
            p,
        )
        paths.append(str(p))
    idx_path = tmp_path / "index.txt"
    idx_path.write_text("\n".join(paths) + "\n")
    return str(idx_path)


def test_balance_background_indices_downsamples(bg_heavy_index):
    ds = MultiModalDataset(
        index_file=bg_heavy_index,
        num_classes=3,
        multilabel_distance_targets=True,
        class_names=CLASS_NAMES,
        distance_threshold_m=THRESHOLD,
        distance_key="distance",
    )
    kept = _balance_background_indices(ds, seed=0)
    assert len(kept) == 6
    assert all(i in kept for i in [6, 7, 8, 9])


def test_balance_background_noop_when_few_bg(tmp_path):
    """If background count is already <= target, all indices are returned."""
    paths = []
    p = tmp_path / "bg.pt"
    torch.save(
        {
            "data": {"shake": {"audio": torch.zeros(1, 1, 1)}},
            "label": np.array(["truck"], dtype=object),
            "distance": {"truck": 99.0},
        },
        p,
    )
    paths.append(str(p))
    for i in range(4):
        p = tmp_path / f"pos_{i}.pt"
        torch.save(
            {
                "data": {"shake": {"audio": torch.zeros(1, 1, 1)}},
                "label": np.array(["polaris"], dtype=object),
                "distance": {"polaris": 2.0},
            },
            p,
        )
        paths.append(str(p))
    idx = tmp_path / "index.txt"
    idx.write_text("\n".join(paths) + "\n")
    ds = MultiModalDataset(
        index_file=str(idx),
        num_classes=3,
        multilabel_distance_targets=True,
        class_names=CLASS_NAMES,
        distance_threshold_m=THRESHOLD,
        distance_key="distance",
    )
    kept = _balance_background_indices(ds)
    assert kept == list(range(5))


def test_resolve_balance_background_true():
    config = {
        "experiment_name": "exp_a",
        "experiments": {"exp_a": {"balance_background": True}},
    }
    assert _resolve_balance_background(config) is True


def test_resolve_balance_background_missing_key():
    config = {
        "experiment_name": "exp_a",
        "experiments": {"exp_a": {"model": "foo"}},
    }
    assert _resolve_balance_background(config) is False


def test_resolve_balance_background_no_experiment_name():
    config = {"experiments": {"exp_a": {"balance_background": True}}}
    with pytest.raises(KeyError):
        _resolve_balance_background(config)


def test_create_dataloaders_with_balance_background(bg_heavy_index, tmp_path):
    """End-to-end: create_dataloaders with balance_background reduces train set."""
    val_idx = tmp_path / "val_index.txt"
    test_idx = tmp_path / "test_index.txt"

    p = tmp_path / "val.pt"
    torch.save(
        {
            "data": {"shake": {"audio": torch.zeros(1, 1, 1)}},
            "label": np.array(["polaris"], dtype=object),
            "distance": {"polaris": 5.0},
        },
        p,
    )
    val_idx.write_text(f"{p}\n")
    test_idx.write_text(f"{p}\n")

    config = {
        "batch_size": 2,
        "num_workers": 0,
        "use_balanced_sampling": False,
        "experiment_name": "test_exp",
        "experiments": {
            "test_exp": {
                "balance_background": True,
                "dataloader": "test_multilabel",
            }
        },
        "dataloader_configs": {
            "test_multilabel": {
                "type": "multilabel_distance",
                "distance_threshold_m": THRESHOLD,
                "distance_key": "distance",
            }
        },
        "task_name": "fine_tune_vehicle_classification",
        "fine_tune_vehicle_classification": {
            "num_classes": 3,
            "class_names": ["polaris", "warthog", "truck"],
            "train_index_file": bg_heavy_index,
            "val_index_file": str(val_idx),
            "test_index_file": str(test_idx),
        },
    }
    train_loader, _, _ = create_dataloaders(config)
    assert len(train_loader.dataset) == 6


def test_create_dataloaders_no_balance_background(bg_heavy_index, tmp_path):
    """Without balance_background, all 10 samples are in the train set."""
    val_idx = tmp_path / "val_index.txt"
    test_idx = tmp_path / "test_index.txt"

    p = tmp_path / "val.pt"
    torch.save(
        {
            "data": {"shake": {"audio": torch.zeros(1, 1, 1)}},
            "label": np.array(["polaris"], dtype=object),
            "distance": {"polaris": 5.0},
        },
        p,
    )
    val_idx.write_text(f"{p}\n")
    test_idx.write_text(f"{p}\n")

    config = {
        "batch_size": 2,
        "num_workers": 0,
        "use_balanced_sampling": False,
        "experiment_name": "test_exp",
        "experiments": {
            "test_exp": {
                "balance_background": False,
                "dataloader": "test_multilabel",
            }
        },
        "dataloader_configs": {
            "test_multilabel": {
                "type": "multilabel_distance",
                "distance_threshold_m": THRESHOLD,
                "distance_key": "distance",
            }
        },
        "task_name": "fine_tune_vehicle_classification",
        "fine_tune_vehicle_classification": {
            "num_classes": 3,
            "class_names": ["polaris", "warthog", "truck"],
            "train_index_file": bg_heavy_index,
            "val_index_file": str(val_idx),
            "test_index_file": str(test_idx),
        },
    }
    train_loader, _, _ = create_dataloaders(config)
    assert len(train_loader.dataset) == 10


def test_create_dataloaders_single_label_only_filters_multi(tmp_index_and_samples, tmp_path):
    val_idx = tmp_path / "val_index.txt"
    test_idx = tmp_path / "test_index.txt"
    index_text = tmp_path / "train_index_copy.txt"
    index_text.write_text(Path(tmp_index_and_samples).read_text())
    val_idx.write_text(index_text.read_text())
    test_idx.write_text(index_text.read_text())

    config = {
        "batch_size": 2,
        "num_workers": 0,
        "use_balanced_sampling": False,
        "experiment_name": "test_exp",
        "experiments": {
            "test_exp": {
                "balance_background": False,
                "dataloader": "test_single",
            }
        },
        "dataloader_configs": {
            "test_single": {"type": "single_label_only"},
        },
        "task_name": "fine_tune_vehicle_classification",
        "fine_tune_vehicle_classification": {
            "num_classes": 3,
            "class_names": CLASS_NAMES,
            "train_index_file": str(index_text),
            "val_index_file": str(val_idx),
            "test_index_file": str(test_idx),
        },
    }

    train_loader, val_loader, test_loader = create_dataloaders(config)
    assert len(train_loader.dataset) == 1
    assert len(val_loader.dataset) == 1
    assert len(test_loader.dataset) == 1
