"""
Inspect on-disk .pt sample layout and MultiModalDataset outputs for YAML dataloader modes.

Run:
  python data_analysis/inspect_dataloader_samples.py --experiment_name only_audio_resnet18
(from the src2 directory), or from anywhere with absolute paths after this file adds src2 to sys.path.

See also: dataset_utils/DATALOADER_CONFIG.md
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

from data_analysis.analyze_gq_multiclass_labels import (
    print_sample_dict,
    sample_has_audio_flag,
)

_SPLIT_KEYS = {
    "train": "train_index_file",
    "val": "val_index_file",
    "test": "test_index_file",
}


def read_index_paths(index_file: Path) -> list[str]:
    with open(index_file) as f:
        return [line.strip() for line in f if line.strip()]


def print_static_cheat_sheet() -> None:
    print("=== Static: dataloader modes vs .pt / model batch ===\n")
    print(
        "Batches from assemble_supervised_dataloaders are always (data, labels, idx).\n"
    )
    print("legacy_multiclass:")
    print("  YAML: dataloader_configs entry with only type: legacy_multiclass")
    print("  torch.load: weights_only=True (tensor-only checkpoints; numpy in .pt may Unpickle)")
    print("  label: returned as stored (e.g. int / tensor); task num_classes metadata only")
    print("  .pt: requires top-level 'data' and 'label' (or label dict with nested 'label')")
    print()
    print("single_label_only:")
    print("  YAML: type: single_label_only only; task must set class_names, num_classes")
    print("  Init: drops index paths whose raw label has cardinality != 1")
    print("  __getitem__: label -> torch.long scalar class index via class_names")
    print("  .pt: same keys; label must be single-entry string/name or index in class_names")
    print()
    print("multilabel_distance:")
    print("  YAML: type, distance_threshold_m, distance_key; task class_names + num_classes")
    print("  __getitem__: per-class float32 binary vector [num_classes]")
    print("  Target 1.0 iff class in present labels AND distance[class] < threshold")
    print("  .pt: 'data', label (object ndarray of class name strings typical), sample[distance_key]")
    print("  distance dict: flat {class: m} or nested {loc: {class: m}, ...}")
    print("  Optional on experiment: balance_background (see multilabel_distance_loader)")
    print()


def resolve_task_and_dataloader(
    config: dict, experiment_name: str | None, task_name_override: str | None
) -> tuple[str, str | None, dict | None]:
    if experiment_name:
        config["experiment_name"] = experiment_name
        exp_cfg = config["experiments"][experiment_name]
        task_name = exp_cfg["task_name"]
        dl_key, dl_cfg = _resolve_dataloader_entry(config)
        return task_name, dl_key, dl_cfg
    if task_name_override:
        return task_name_override, None, None
    raise ValueError("Provide --experiment_name or --task_name")


def index_path_for_split(task_config: dict, split: str) -> Path:
    key = _SPLIT_KEYS[split]
    if key not in task_config:
        raise KeyError(f"task config missing '{key}'")
    return Path(task_config[key])


def dataset_kwargs_for_mode(
    mode: str,
    task_config: dict,
    multilabel_dl_cfg: dict | None,
) -> dict:
    nc = task_config["num_classes"]
    cn = task_config["class_names"] if "class_names" in task_config else None

    if mode == "legacy_multiclass":
        return {
            "num_classes": nc,
            "multilabel_distance_targets": False,
            "single_label_only": False,
        }
    if mode == "single_label_only":
        if cn is None:
            raise ValueError("single_label_only requires task class_names")
        return {
            "num_classes": nc,
            "multilabel_distance_targets": False,
            "single_label_only": True,
            "class_names": list(cn),
        }
    if mode == "multilabel_distance":
        if multilabel_dl_cfg is None:
            raise ValueError("multilabel_distance requires a dataloader_configs block")
        if cn is None:
            raise ValueError("multilabel_distance requires task class_names")
        return {
            "num_classes": nc,
            "multilabel_distance_targets": True,
            "single_label_only": False,
            "class_names": list(cn),
            "distance_threshold_m": float(multilabel_dl_cfg["distance_threshold_m"]),
            "distance_key": str(multilabel_dl_cfg["distance_key"]),
        }
    raise ValueError(f"unknown mode {mode!r}")


def first_multilabel_config(config: dict) -> dict | None:
    if "dataloader_configs" not in config:
        return None
    dcs = config["dataloader_configs"]
    for _key, block in dcs.items():
        if block["type"] == "multilabel_distance":
            return block
    return None


def describe_tensor(t: object) -> str:
    if isinstance(t, torch.Tensor):
        return f"Tensor shape={tuple(t.shape)} dtype={t.dtype}"
    return f"{type(t).__name__} {t!r}"


def inspect_raw_samples(
    paths: list[str],
    max_samples: int,
    filter_audio: bool,
) -> None:
    print("=== On-disk .pt samples (raw torch.load, weights_only=False) ===\n")
    shown = 0
    for p in paths:
        if shown >= max_samples:
            break
        pt = Path(p)
        if not pt.is_file():
            print(f"  skip missing file: {p}")
            continue
        try:
            sample = torch.load(p, weights_only=False)
        except Exception as e:
            print(f"  ERROR load {p}: {e}")
            continue
        if filter_audio:
            try:
                if not sample_has_audio_flag(sample):
                    continue
            except Exception as e:
                print(f"  skip (flag check failed) {p}: {e}")
                continue
        print(f"--- {p}")
        try:
            print_sample_dict(sample)
        except Exception as e:
            print(f"  (structure print failed: {e})")
            print(f"  keys: {sorted(sample.keys())}")
        print()
        shown += 1
    if shown == 0:
        print("  (no samples printed; check paths, --filter_audio, or --max_samples)\n")


def inspect_dataset_outputs(
    dataset: MultiModalDataset,
    max_samples: int,
    label_mode: str,
) -> None:
    print(
        f"=== MultiModalDataset.__getitem__ ({label_mode}) — first samples ===\n"
    )
    n = min(max_samples, len(dataset))
    for i in range(n):
        try:
            data, label, idx = dataset[i]
        except Exception as e:
            print(f"  dataset[{i}] ERROR: {e}")
            continue
        print(f"--- index {i} (returned idx={idx})")
        if isinstance(data, dict):
            for loc, mods in data.items():
                print(f"  data[{loc!r}]:")
                if isinstance(mods, dict):
                    for mname, t in mods.items():
                        print(f"    {mname}: {describe_tensor(t)}")
                else:
                    print(f"    {describe_tensor(mods)}")
        else:
            print(f"  data: {describe_tensor(data)}")
        print(f"  label: {describe_tensor(label)}")
        print()


def print_all_modes_table(
    index_file: Path,
    task_config: dict,
    config: dict,
) -> None:
    ml_cfg = first_multilabel_config(config)
    if ml_cfg is None:
        print(
            "\n=== --all_dataloader_modes: no multilabel_distance block in dataloader_configs ===\n"
        )
        return

    raw_n = len(read_index_paths(index_file))
    print("\n=== All dataloader modes (same task index file) ===\n")
    print(
        f"{'mode':<22} {'index_lines':>12} {'len(dataset)':>14} {'example_label (first sample)':<50}"
    )
    print("-" * 102)

    for mode in ("legacy_multiclass", "single_label_only", "multilabel_distance"):
        try:
            kw = dataset_kwargs_for_mode(mode, task_config, ml_cfg)
        except ValueError as e:
            print(f"{mode:<22} {'n/a':>12} {'n/a':>14} {str(e)[:50]}")
            continue
        try:
            ds = MultiModalDataset(str(index_file), **kw)
        except Exception as e:
            print(f"{mode:<22} {raw_n:>12} {'ERROR':>14} {str(e)[:50]}")
            continue
        ex = ""
        if len(ds) > 0:
            try:
                _, lab, _ = ds[0]
                ex = describe_tensor(lab)
            except Exception as e:
                ex = f"__getitem__ failed: {e}"
        print(f"{mode:<22} {raw_n:>12} {len(ds):>14} {ex[:50]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect .pt layout and MultiModalDataset outputs for dataloader modes"
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=str(_SRC2_ROOT / "data/Parkland.yaml"),
        help="Path to YAML (default: src2/data/Parkland.yaml next to this script tree)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name under config['experiments'] (with --task_name, optional)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Task block key (required if --experiment_name omitted)",
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
        default=3,
        help="Max raw samples and max dataset rows to print",
    )
    parser.add_argument(
        "--filter_audio",
        action="store_true",
        help="Only print raw samples where sample_has_audio_flag is True",
    )
    parser.add_argument(
        "--all_dataloader_modes",
        action="store_true",
        help="Print comparison table for all three modes on this task index",
    )
    parser.add_argument(
        "--skip_static",
        action="store_true",
        help="Omit static cheat sheet",
    )
    args = parser.parse_args()

    if not args.experiment_name and not args.task_name:
        parser.error("Provide --experiment_name and/or --task_name (at least one required)")

    logging.basicConfig(level=logging.WARNING)

    yaml_path = Path(args.yaml_path)
    if not yaml_path.is_file():
        print(
            f"Error: YAML not found: {yaml_path.resolve()}\n"
            "Hint: pass --yaml_path to your YAML, or use the default under src2/data/.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = load_yaml_config(str(yaml_path))

    try:
        task_name, dl_key, dl_cfg = resolve_task_and_dataloader(
            config, args.experiment_name, args.task_name
        )
    except (KeyError, ValueError) as e:
        print(f"Error resolving experiment/task: {e}", file=sys.stderr)
        sys.exit(1)

    if task_name not in config:
        print(
            f"Error: task_name '{task_name}' not found as top-level key in YAML",
            file=sys.stderr,
        )
        sys.exit(1)

    task_config = config[task_name]

    print("=== Resolution ===\n")
    if args.experiment_name:
        print(f"experiment_name:     {args.experiment_name}")
        print(f"dataloader_configs: {dl_key} -> type={dl_cfg['type']}")
        if dl_cfg["type"] == "multilabel_distance":
            print(
                f"  distance_threshold_m={dl_cfg['distance_threshold_m']}, "
                f"distance_key={dl_cfg['distance_key']!r}"
            )
    else:
        print("experiment_name:   (not set; using --task_name only)")
    print(f"task_name:           {task_name}")
    print(f"num_classes:         {task_config['num_classes']}")
    if "class_names" in task_config:
        print(f"class_names:         {task_config['class_names']}")
    print()

    if not args.skip_static:
        print_static_cheat_sheet()

    try:
        index_file = index_path_for_split(task_config, args.split)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not index_file.is_file():
        print(
            f"Error: index file not found: {index_file}\n"
            "Check train_index_file / val_index_file / test_index_file in the task block.",
            file=sys.stderr,
        )
        sys.exit(1)

    paths = read_index_paths(index_file)
    print(f"=== Index ({args.split}) ===\n{index_file}\nlines: {len(paths)}\n")

    inspect_raw_samples(paths, args.max_samples, args.filter_audio)

    if args.experiment_name and dl_cfg is not None:
        mode = dl_cfg["type"]
        try:
            kw = dataset_kwargs_for_mode(mode, task_config, dl_cfg)
        except ValueError as e:
            print(f"Cannot build dataset kwargs: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"=== Dataset for experiment dataloader ({mode}) ===\n")
        try:
            ds = MultiModalDataset(str(index_file), **kw)
        except Exception as e:
            print(f"Error building MultiModalDataset: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"len(dataset) after init: {len(ds)} (index had {len(paths)} paths)\n")
        inspect_dataset_outputs(ds, args.max_samples, mode)

    elif dl_cfg is None:
        print(
            "=== Skipping single-mode MultiModalDataset (--experiment_name not set) ===\n"
            "Pass --experiment_name to build the dataset matching that experiment's dataloader.\n"
        )

    if args.all_dataloader_modes:
        print_all_modes_table(index_file, task_config, config)


if __name__ == "__main__":
    main()
