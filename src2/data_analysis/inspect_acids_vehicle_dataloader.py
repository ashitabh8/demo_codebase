"""
Smoke-test ACIDS vehicle dataloader: MultiModalDataset with label_subkey=vehicle_type.

Run from src2:
  python data_analysis/inspect_acids_vehicle_dataloader.py \\
    --yaml_path data/ACIDS.yaml --experiment_name only_audio_deepsense_dw_tiny

Or with explicit dataloader key (must exist under dataloader_configs):
  python data_analysis/inspect_acids_vehicle_dataloader.py \\
    --yaml_path data/ACIDS.yaml --task_name vehicle_classification --dataloader_key acids_vehicle
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SRC2_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC2_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC2_ROOT))

import torch

from dataset_utils.dataloader_factory import _resolve_dataloader_entry
from dataset_utils.multimodal_core import MultiModalDataset
from dataset_utils.parse_args_utils import load_yaml_config


def read_index_paths(index_file: Path) -> list[str]:
    with open(index_file) as f:
        return [line.strip() for line in f if line.strip()]


def dataset_kwargs_acids_vehicle(task_config: dict, dl_cfg: dict) -> dict:
    nc = task_config["num_classes"]
    cn = task_config["class_names"]
    if "label_subkey" not in dl_cfg:
        raise ValueError("acids_vehicle dataloader_configs entry needs label_subkey")
    return {
        "num_classes": nc,
        "multilabel_distance_targets": False,
        "single_label_only": True,
        "class_names": list(cn),
        "label_subkey": dl_cfg["label_subkey"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect ACIDS vehicle_type labels and MultiModalDataset batches"
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=str(_SRC2_ROOT / "data/ACIDS.yaml"),
        help="Path to ACIDS (or other) YAML",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment under config['experiments'] (resolves dataloader + task)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Task block key if not using --experiment_name",
    )
    parser.add_argument(
        "--dataloader_key",
        type=str,
        default=None,
        help="Override dataloader_configs key (requires --task_name)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Which task index file to use",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Max dataset rows to print from __getitem__",
    )
    parser.add_argument(
        "--run_batch",
        action="store_true",
        help="Also build train DataLoader and fetch one batch (full pipeline)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    yaml_path = Path(args.yaml_path)
    if not yaml_path.is_file():
        print(f"Error: YAML not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    config = load_yaml_config(str(yaml_path))

    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
        exp_cfg = config["experiments"][args.experiment_name]
        task_name = exp_cfg["task_name"]
        config["task_name"] = task_name
        dl_key, dl_cfg = _resolve_dataloader_entry(config)
    elif args.task_name:
        task_name = args.task_name
        config["task_name"] = task_name
        if args.dataloader_key is None:
            print(
                "Error: with --task_name, pass --dataloader_key (e.g. acids_vehicle)",
                file=sys.stderr,
            )
            sys.exit(1)
        dl_key = args.dataloader_key
        if "dataloader_configs" not in config:
            print("Error: config missing dataloader_configs", file=sys.stderr)
            sys.exit(1)
        dcs = config["dataloader_configs"]
        if dl_key not in dcs:
            print(f"Error: unknown dataloader key {dl_key!r}", file=sys.stderr)
            sys.exit(1)
        dl_cfg = dcs[dl_key]
    else:
        print(
            "Error: provide --experiment_name or (--task_name and --dataloader_key)",
            file=sys.stderr,
        )
        sys.exit(1)

    if dl_cfg["type"] != "acids_vehicle_classification":
        print(
            f"Warning: dataloader type is {dl_cfg['type']!r}, "
            f"not acids_vehicle_classification (script is for ACIDS vehicle .pt layout).",
            file=sys.stderr,
        )

    if task_name not in config:
        print(f"Error: unknown task {task_name!r}", file=sys.stderr)
        sys.exit(1)

    task_config = config[task_name]
    split_key = {
        "train": "train_index_file",
        "val": "val_index_file",
        "test": "test_index_file",
    }[args.split]
    index_file = Path(task_config[split_key])

    print("=== Resolution ===\n")
    if args.experiment_name:
        print(f"experiment_name:     {args.experiment_name}")
    print(f"dataloader_configs:  {dl_key} -> type={dl_cfg['type']}")
    if "label_subkey" in dl_cfg:
        print(f"label_subkey:        {dl_cfg['label_subkey']!r}")
    print(f"task_name:           {task_name}")
    print(f"num_classes:         {task_config['num_classes']}")
    print(f"class_names:         {task_config['class_names']}")
    print(f"index ({args.split}): {index_file}")
    print()

    if not index_file.is_file():
        print(
            f"Error: index file not found: {index_file}\n"
            "Point YAML task paths at your ACIDS index files.",
            file=sys.stderr,
        )
        sys.exit(1)

    paths = read_index_paths(index_file)
    print(f"index lines: {len(paths)}\n")

    kw = dataset_kwargs_acids_vehicle(task_config, dl_cfg)
    print("=== Raw torch.load (first path, label dict) ===\n")
    for p in paths[:3]:
        pt = Path(p)
        if not pt.is_file():
            print(f"  skip missing: {p}")
            continue
        sample = torch.load(p, weights_only=False)
        lab = sample["label"]
        print(f"--- {p}")
        if isinstance(lab, dict):
            print(f"  label keys: {sorted(lab.keys())}")
            sk = dl_cfg["label_subkey"]
            if sk in lab:
                print(f"  label[{sk!r}]: {lab[sk]!r} (type {type(lab[sk]).__name__})")
        else:
            print(f"  label (not dict): {lab!r}")
        print()
        break

    print("=== MultiModalDataset (filtered single-label, class index) ===\n")
    ds = MultiModalDataset(str(index_file), **kw)
    print(f"len(dataset) after init: {len(ds)}\n")
    n = min(args.max_samples, len(ds))
    for i in range(n):
        data, label, idx = ds[i]
        print(f"--- dataset[{i}] idx={idx} label_tensor={label} (class idx)")
        if isinstance(data, dict):
            for loc, mods in data.items():
                print(f"  data[{loc!r}]:")
                if isinstance(mods, dict):
                    for mname, t in mods.items():
                        if isinstance(t, torch.Tensor):
                            print(
                                f"    {mname}: shape={tuple(t.shape)} dtype={t.dtype}"
                            )
                        else:
                            print(f"    {mname}: {type(t).__name__}")
        print()

    if args.run_batch:
        from dataset_utils.dataloader_factory import create_dataloaders

        if not args.experiment_name:
            print(
                "Error: --run_batch requires --experiment_name (full config resolution)",
                file=sys.stderr,
            )
            sys.exit(1)
        print("=== create_dataloaders (one train batch) ===\n")
        tl, _, _ = create_dataloaders(config)
        batch = next(iter(tl))
        data_b, labels_b, idx_b = batch
        print(f"batch labels shape: {tuple(labels_b.shape)} dtype={labels_b.dtype}")
        print(f"batch idx shape:    {tuple(idx_b.shape)}")
        print("OK: train DataLoader produced one batch.")


if __name__ == "__main__":
    main()
