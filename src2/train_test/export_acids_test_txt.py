"""
Export the first N ACIDS test samples (preprocessed, post-augmenter) to per-sample
txt files, in BOTH layouts:

    <output_root>/python_nchw/<sample_name>.txt   - flat N C H W (channel-first)
    <output_root>/c_nhwc/<sample_name>.txt        - flat N H W C (channel-last)

Each txt file layout (one file per sample, same for both folders):
    line 1        : sample_name (filename stem of the source .pt, with spaces -> _)
    line 2        : integer class label id
    lines 3..N+2  : flattened float values, one per line

Both folders cover the exact same samples (same order, same labels); only the
flatten order of the float values differs. Reconstruction:

    # Python folder (NCHW):
    vals = np.loadtxt(path, skiprows=2, dtype=np.float32)
    chw = vals.reshape(C, H, W)                    # matches PyTorch [C, H, W]

    # C folder (NHWC):
    vals = np.loadtxt(path, skiprows=2, dtype=np.float32)
    hwc = vals.reshape(H, W, C)                    # NHWC tile used on device
    chw = np.transpose(hwc, (2, 0, 1))             # back to PyTorch layout

The default experiment is `finetune_audio_deepsense_dw_large_mel`, which uses
the registered `single_label_only` ACIDS dataloader and mel preprocessing
(3 channels x 7 segments x 80 mel bins = 1680 values per sample). Override with
--experiment_name if you want a different shape, but make sure the dataloader
type in ACIDS.yaml is one of the registered types in
`src2/dataset_utils/dataloader_factory.py`.

Usage (from src2/train_test):

    python export_acids_test_txt.py \\
        --yaml_path ../data/ACIDS.yaml \\
        --experiment_name finetune_audio_deepsense_dw_large_mel \\
        --num_samples 500 \\
        --gpu -1
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

src2_path = Path(__file__).resolve().parent.parent
if str(src2_path) not in sys.path:
    sys.path.insert(0, str(src2_path))

from data_augmenter import apply_augmentation, create_augmenter
from dataset_utils.MultiModalDataLoader import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export first N ACIDS test samples to per-sample txt files",
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=str(src2_path / "data" / "ACIDS.yaml"),
        help="Path to ACIDS.yaml",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="finetune_audio_deepsense_dw_large_mel",
        help=(
            "Experiment from experiments: block. Must use a registered dataloader "
            "type (single_label_only / legacy_multiclass / multilabel_distance / "
            "single_label_seismic_only)."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="How many test samples to export (default 500).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU id for preprocessing; -1 runs on CPU.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(src2_path / "experiments" / "acids_txt_export"),
        help=(
            "Root directory. Two subfolders are created underneath: "
            "python_nchw/ and c_nhwc/."
        ),
    )
    parser.add_argument(
        "--python_subdir",
        type=str,
        default="python_nchw",
        help="Subfolder name for N C H W (channel-first) txt files.",
    )
    parser.add_argument(
        "--c_subdir",
        type=str,
        default="c_nhwc",
        help="Subfolder name for N H W C (channel-last) txt files.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="",
        help=(
            "CSV manifest path. If empty, writes "
            "<output_root>/test_manifest.csv."
        ),
    )
    return parser.parse_args()


def load_config(yaml_path: Path, experiment_name: str, gpu: int) -> dict:
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["experiment_name"] = experiment_name
    config["yaml_path"] = str(yaml_path)
    config["device"] = "cpu" if gpu < 0 else f"cuda:{gpu}"
    config["include_classes"] = None
    return config


def resolve_experiment_block(config: dict, experiment_name: str) -> dict:
    experiments = config["experiments"]
    if experiment_name not in experiments:
        available = sorted(k for k in experiments if k != "enabled")
        raise ValueError(
            f"Experiment '{experiment_name}' not found. Available: {available}"
        )
    return experiments[experiment_name]


def resolve_label_id(labels: torch.Tensor, sample_idx: int) -> int:
    sample_label = labels[sample_idx]
    if sample_label.ndim > 0 and sample_label.numel() > 1:
        return int(torch.argmax(sample_label).item())
    return int(sample_label.item())


def to_nhwc_flat(sample_chw: torch.Tensor) -> np.ndarray:
    """Convert a [C, H, W] tensor to a float32 1-D array in HWC (channel-last) order."""
    if sample_chw.ndim != 3:
        raise ValueError(
            f"Expected per-sample tensor with 3 dims [C, H, W], got shape {tuple(sample_chw.shape)}"
        )
    hwc = sample_chw.permute(1, 2, 0).contiguous()
    return hwc.detach().cpu().numpy().astype(np.float32).reshape(-1)


def to_nchw_flat(sample_chw: torch.Tensor) -> np.ndarray:
    """Convert a [C, H, W] tensor to a float32 1-D array in CHW (channel-first) order."""
    if sample_chw.ndim != 3:
        raise ValueError(
            f"Expected per-sample tensor with 3 dims [C, H, W], got shape {tuple(sample_chw.shape)}"
        )
    chw = sample_chw.contiguous()
    return chw.detach().cpu().numpy().astype(np.float32).reshape(-1)


def write_sample_txt(path: Path, sample_name: str, label_id: int, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{sample_name}\n")
        handle.write(f"{label_id}\n")
        for value in values:
            handle.write(f"{float(value):.9g}\n")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()

    yaml_path = Path(args.yaml_path).resolve()
    output_root = Path(args.output_root).resolve()
    python_dir = output_root / args.python_subdir
    c_dir = output_root / args.c_subdir
    if args.manifest_path:
        manifest_path = Path(args.manifest_path).resolve()
    else:
        manifest_path = output_root / "test_manifest.csv"

    config = load_config(yaml_path, args.experiment_name, args.gpu)
    experiment_config = resolve_experiment_block(config, args.experiment_name)
    if "task_name" not in experiment_config:
        raise ValueError(
            f"Experiment '{args.experiment_name}' must define 'task_name'"
        )
    config["task_name"] = experiment_config["task_name"]

    model_name = experiment_config["model"]
    model_cfg = config["models"][model_name]
    active_modality = model_cfg["active_modality"]
    location = config["location_names"][0]
    class_names = config[experiment_config["task_name"]]["class_names"]

    logging.info("Experiment: %s", args.experiment_name)
    logging.info("Model: %s (modality=%s)", model_name, active_modality)
    logging.info(
        "Preprocess mode: %s",
        experiment_config["preprocess_mode"]
        if "preprocess_mode" in experiment_config
        else config["preprocess_mode"]
        if "preprocess_mode" in config
        else "fft",
    )

    _, _, test_loader = create_dataloaders(config=config)
    augmenter = create_augmenter(
        config=config, augmentation_mode="no", experiment_config=experiment_config
    )

    dataset = test_loader.dataset
    total_test = len(dataset)
    target = min(args.num_samples, total_test)
    logging.info(
        "Test split has %d samples; exporting the first %d.", total_test, target
    )

    python_dir.mkdir(parents=True, exist_ok=True)
    c_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    duplicate_counter: dict = {}
    written = 0

    with torch.no_grad():
        for batch in test_loader:
            if written >= target:
                break
            data, labels, indices = batch

            freq_data, out_labels = apply_augmentation(augmenter, data, labels)
            feats = freq_data[location][active_modality]  # [B, C, H, W]
            batch_size = feats.shape[0]

            for i in range(batch_size):
                if written >= target:
                    break
                ds_idx = int(indices[i].item())
                source_path = Path(dataset.sample_files[ds_idx])
                base_name = source_path.stem.replace(" ", "_")

                if base_name in duplicate_counter:
                    duplicate_counter[base_name] += 1
                    suffix = duplicate_counter[base_name]
                    sample_name = f"{base_name}__dup{suffix}"
                else:
                    duplicate_counter[base_name] = 0
                    sample_name = base_name

                label_id = resolve_label_id(out_labels, i)
                nchw_flat = to_nchw_flat(feats[i])
                nhwc_flat = to_nhwc_flat(feats[i])

                python_path = python_dir / f"{sample_name}.txt"
                c_path = c_dir / f"{sample_name}.txt"
                write_sample_txt(python_path, sample_name, label_id, nchw_flat)
                write_sample_txt(c_path, sample_name, label_id, nhwc_flat)

                c, h, w = feats[i].shape
                label_name = (
                    class_names[label_id] if 0 <= label_id < len(class_names) else ""
                )
                rows.append(
                    {
                        "sample_name": sample_name,
                        "split": "test",
                        "label_id": label_id,
                        "label_name": label_name,
                        "num_values": int(nchw_flat.size),
                        "C": int(c),
                        "H": int(h),
                        "W": int(w),
                        "python_nchw_txt": str(python_path),
                        "c_nhwc_txt": str(c_path),
                        "source_sample_path": str(source_path),
                    }
                )
                written += 1
                if written % 100 == 0:
                    logging.info("  wrote %d / %d samples", written, target)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_name",
        "split",
        "label_id",
        "label_name",
        "num_values",
        "C",
        "H",
        "W",
        "python_nchw_txt",
        "c_nhwc_txt",
        "source_sample_path",
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Exported %d samples.", len(rows))
    logging.info("  Python (NCHW): %s", python_dir)
    logging.info("  C      (NHWC): %s", c_dir)
    logging.info("  Manifest     : %s", manifest_path)


if __name__ == "__main__":
    main()
