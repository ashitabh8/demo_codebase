#!/usr/bin/env python3
"""Export balanced mel-audio demo samples (CSV + metadata) for device inference."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC2_ROOT = REPO_ROOT / "src2"
if str(SRC2_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC2_ROOT))

from dataset_utils.dataloader_factory import create_dataloaders  # noqa: E402
from dataset_utils.parse_args_utils import load_yaml_config  # noqa: E402
from data_augmenter.augmenter_utils import apply_augmentation, create_augmenter  # noqa: E402
from train_test.train_test_utils import apply_class_subset  # noqa: E402


DEFAULT_YAML = SRC2_ROOT / "data" / "Parkland.yaml"
DEFAULT_OUTPUT_DIR = SRC2_ROOT / "gen_code" / "demo_data"
DEFAULT_EXPERIMENT = "only_audio_deepsense_dw_large_mel"
DEFAULT_TASK = "vehicle_classification"
DEFAULT_DATALOADER = "parkland_legacy_multiclass"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export mel-audio demo samples from Parkland.")
    parser.add_argument("--yaml_path", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task_name", type=str, default=DEFAULT_TASK)
    parser.add_argument("--dataloader_key", type=str, default=DEFAULT_DATALOADER)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--num_samples", type=int, default=90)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target_classes", nargs="+", default=["Polaris", "Warhog", "Truck"])
    parser.add_argument("--location", type=str, default="shake")
    parser.add_argument("--modality", type=str, default="audio")
    parser.add_argument(
        "--export_header",
        action="store_true",
        help="Also export C header with flattened feature arrays.",
    )
    return parser.parse_args()


def _resolve_include_class_indices(task_cfg: dict[str, Any], target_names: list[str]) -> list[int]:
    class_names = [str(name) for name in task_cfg["class_names"]]
    lookup = {name.lower(): idx for idx, name in enumerate(class_names)}
    include = []
    for name in target_names:
        key = name.lower()
        if key not in lookup:
            raise ValueError(f"Target class '{name}' not in task class_names={class_names}")
        include.append(lookup[key])
    return sorted(set(include))


def _select_counts(available: dict[int, int], requested: int, n_classes: int) -> dict[int, int]:
    base = requested // n_classes
    rem = requested % n_classes
    chosen = {}
    for cls in range(n_classes):
        need = base + (1 if cls < rem else 0)
        chosen[cls] = min(need, available.get(cls, 0))
    picked = sum(chosen.values())
    spare = {cls: max(0, available.get(cls, 0) - chosen[cls]) for cls in range(n_classes)}
    while picked < requested:
        progressed = False
        for cls in range(n_classes):
            if spare[cls] > 0:
                chosen[cls] += 1
                spare[cls] -= 1
                picked += 1
                progressed = True
                if picked >= requested:
                    break
        if not progressed:
            break
    return chosen


def _write_csv(
    samples_path: Path,
    labels_path: Path,
    selected: list[dict[str, Any]],
    feature_size: int,
) -> None:
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    feature_headers = [f"feature_{i}" for i in range(feature_size)]
    with samples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "target"] + feature_headers)
        for row in selected:
            writer.writerow([row["sample_id"], row["target"]] + row["features"])

    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "target", "class_name", "source_dataset_index"])
        for row in selected:
            writer.writerow([row["sample_id"], row["target"], row["class_name"], row["source_dataset_index"]])


def _write_header(path: Path, selected: list[dict[str, Any]], feature_size: int, n_classes: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("// Auto-generated demo sample header\n")
        handle.write("#pragma once\n\n")
        handle.write(f"#define DEMO_NUM_SAMPLES {len(selected)}\n")
        handle.write(f"#define DEMO_FEATURE_SIZE {feature_size}\n")
        handle.write(f"#define DEMO_NUM_CLASSES {n_classes}\n\n")
        handle.write("static const float DEMO_SAMPLES[DEMO_NUM_SAMPLES][DEMO_FEATURE_SIZE] = {\n")
        for row in selected:
            values = ", ".join(f"{float(v):.9g}f" for v in row["features"])
            handle.write(f"  {{{values}}},\n")
        handle.write("};\n\n")
        labels = ", ".join(str(int(row["target"])) for row in selected)
        handle.write(f"static const int DEMO_TARGETS[DEMO_NUM_SAMPLES] = {{{labels}}};\n")


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_yaml_config(str(args.yaml_path))
    config["experiment_name"] = args.experiment_name
    config["device"] = "cpu"
    config["num_workers"] = 0
    config.setdefault("batch_size", 32)

    if args.experiment_name not in config["experiments"]:
        raise ValueError(f"Experiment '{args.experiment_name}' not found in YAML.")
    exp_cfg = config["experiments"][args.experiment_name]
    exp_cfg["task_name"] = args.task_name
    exp_cfg["dataloader"] = args.dataloader_key
    exp_cfg["preprocess_mode"] = "mel"
    config["task_name"] = args.task_name

    task_cfg = config[config["task_name"]]
    include_classes = _resolve_include_class_indices(task_cfg, args.target_classes)
    config["include_classes"] = include_classes
    apply_class_subset(config)

    train_loader, val_loader, test_loader = create_dataloaders(config)
    split_to_loader = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = split_to_loader[args.split]
    class_names = list(config[config["task_name"]]["class_names"])

    augmenter = create_augmenter(config, augmentation_mode="no", experiment_config=exp_cfg)
    per_class_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    feature_shape = None

    for data, labels, indices in loader:
        with torch.no_grad():
            freq_data, labels = apply_augmentation(augmenter, data, labels)

        if args.location not in freq_data or args.modality not in freq_data[args.location]:
            raise KeyError(
                f"Could not find location/modality '{args.location}/{args.modality}' in batch keys={list(freq_data.keys())}"
            )

        audio_tensor = freq_data[args.location][args.modality]
        if feature_shape is None:
            feature_shape = list(audio_tensor.shape[1:])

        bsz = int(audio_tensor.shape[0])
        for i in range(bsz):
            target = int(labels[i].item())
            feats = audio_tensor[i].detach().cpu().reshape(-1).numpy().astype(np.float32)
            per_class_rows[target].append(
                {
                    "target": target,
                    "class_name": class_names[target],
                    "source_dataset_index": int(indices[i].item()),
                    "features": feats.tolist(),
                }
            )

    num_classes = len(class_names)
    available = {cls: len(per_class_rows.get(cls, [])) for cls in range(num_classes)}
    chosen_counts = _select_counts(available, args.num_samples, num_classes)
    selected: list[dict[str, Any]] = []
    for cls in range(num_classes):
        rows = per_class_rows.get(cls, [])
        random.shuffle(rows)
        selected.extend(rows[: chosen_counts[cls]])

    random.shuffle(selected)
    for i, row in enumerate(selected):
        row["sample_id"] = i

    if not selected:
        raise RuntimeError("No samples selected. Check split, class filters, and dataset availability.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_csv = output_dir / "demo_samples.csv"
    labels_csv = output_dir / "demo_labels.csv"
    metadata_json = output_dir / "demo_metadata.json"
    feature_size = len(selected[0]["features"])

    _write_csv(samples_csv, labels_csv, selected, feature_size)
    if args.export_header:
        _write_header(output_dir / "demo_samples.h", selected, feature_size, num_classes)

    counts = {name: 0 for name in class_names}
    for row in selected:
        counts[row["class_name"]] += 1

    metadata = {
        "yaml_path": str(args.yaml_path),
        "experiment_name": args.experiment_name,
        "task_name": args.task_name,
        "dataloader_key": args.dataloader_key,
        "split": args.split,
        "seed": args.seed,
        "num_selected": len(selected),
        "requested_num_samples": args.num_samples,
        "location": args.location,
        "modality": args.modality,
        "class_names": class_names,
        "selected_class_counts": counts,
        "available_class_counts": available,
        "feature_shape_per_sample": feature_shape,
        "feature_size_flat": feature_size,
        "flatten_order": "row-major over [channels, segments, mel_bins]",
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved samples: {samples_csv}")
    print(f"Saved labels: {labels_csv}")
    print(f"Saved metadata: {metadata_json}")
    print(f"Selected class counts: {counts}")


if __name__ == "__main__":
    main()
