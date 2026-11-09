#!/usr/bin/env python3
"""Train audio, seismic, and audio+seismic spectral MLPs, then plot per-class recall.

Uses the Welch frontend and MLP from train_from_1600.py (the pushed deploy trainer):
  channel 0, flatten (7, 256) -> 1600 samples, Welch n_fft=160 hop=80,
  StandardScaler, MLP(in -> 128 -> 128 -> C).

Each model is trained once in float32. Weights are then quantized directly
(per-output-channel symmetric absmax, activations and biases left float) at
bit widths 2..8 with no retraining. The heatmap is per-class recall on the
held-out test split. The checkpoint is the epoch with the best validation accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score

PKG = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(PKG))

from train_from_1600 import (  # noqa: E402
    SpectralMLP,
    resolve_class_names,
    resolve_num_classes,
    train_mlp,
)
from welch_features import (  # noqa: E402
    EPS,
    FEATURE_DIM,
    HOP,
    N_FFT,
    SAMPLE_RATE,
    streams_to_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

USED_SAMPLES = 1600
NUM_SEGMENTS = 7
PADDED_SEG_LEN = 256
BITS = (2, 3, 4, 5, 6, 7, 8)
MODELS = ("audio", "seismic", "audio+seismic")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths_yaml",
        type=Path,
        default=PKG / "paths.yaml",
    )
    parser.add_argument("--output_dir", type=Path, default=PKG / "artifacts" / "class_recall")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Force MLP head size. Default: max(train/val label)+1.",
    )
    parser.add_argument(
        "--class_names",
        type=Path,
        default=None,
        help="Text file with one class name per line (length must equal num_classes).",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="Cap each split (0 = all)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reuse_cache", action="store_true")
    parser.add_argument("--reuse_checkpoints", action="store_true")
    return parser.parse_args()


def read_index(path: Path, max_samples: int) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if max_samples > 0:
        lines = lines[:max_samples]
    return lines


def _load_chunk(payload: tuple[list[str], str]) -> tuple[np.ndarray, np.ndarray]:
    paths, modality = payload
    streams = np.empty((len(paths), USED_SAMPLES), dtype=np.float32)
    labels = np.empty(len(paths), dtype=np.int64)
    for i, pt_path in enumerate(paths):
        sample = torch.load(pt_path, map_location="cpu", weights_only=False)
        tensor = sample["data"]["shake"][modality]
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        tensor = tensor.float()
        if tuple(tensor.shape[1:]) != (NUM_SEGMENTS, PADDED_SEG_LEN):
            raise ValueError(f"{pt_path} {modality} shape {tuple(tensor.shape)}")
        streams[i] = tensor[0].reshape(-1)[:USED_SAMPLES].cpu().numpy()
        label = sample["label"]["vehicle_type"]
        labels[i] = int(label.item()) if torch.is_tensor(label) else int(label)
    return streams, labels


def load_split(paths: list[str], modality: str, workers: int) -> tuple[np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("empty index")
    chunk = max(1, (len(paths) + workers * 4 - 1) // (workers * 4))
    payloads = []
    for start in range(0, len(paths), chunk):
        payloads.append((paths[start : start + chunk], modality))
    streams = []
    labels = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part_x, part_y in pool.map(_load_chunk, payloads):
            streams.append(part_x)
            labels.append(part_y)
    return np.concatenate(streams, axis=0), np.concatenate(labels, axis=0)


def welch_matrix(streams: np.ndarray) -> np.ndarray:
    """Vectorized Welch features. Checked against streams_to_features."""
    if streams.ndim != 2 or streams.shape[1] != USED_SAMPLES:
        raise ValueError(f"expected (N,{USED_SAMPLES}), got {streams.shape}")
    last_start = USED_SAMPLES - N_FFT
    starts = list(range(0, last_start + 1, HOP))
    if starts[-1] != last_start:
        starts.append(last_start)
    frames = np.stack([streams[:, s : s + N_FFT] for s in starts], axis=1)
    fft = np.fft.rfft(frames.astype(np.float32), n=N_FFT, axis=-1)
    power = (fft.real.astype(np.float64) ** 2 + fft.imag.astype(np.float64) ** 2).mean(axis=1)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE).astype(np.float64)
    psd = power / (power.sum(axis=-1, keepdims=True) + EPS)
    centroid = (power * freqs).sum(axis=-1) / (power.sum(axis=-1) + EPS)
    mean_energy = power.mean(axis=-1)
    features = np.concatenate(
        [centroid.reshape(-1, 1), mean_energy.reshape(-1, 1), psd],
        axis=1,
    ).astype(np.float32)
    if features.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"expected {FEATURE_DIM} features, got {features.shape[1]}")
    return features


def assert_welch_matches(streams: np.ndarray) -> None:
    probe = streams[: min(8, len(streams))]
    got = welch_matrix(probe)
    ref = streams_to_features(probe)
    if not np.allclose(got, ref, rtol=1e-5, atol=1e-5):
        err = float(np.max(np.abs(got - ref)))
        raise RuntimeError(f"vectorized Welch diverges from deploy frontend by {err}")
    err = float(np.max(np.abs(got - ref)))
    logger.info("Welch matches deploy frontend (max abs err %.3g)", err)


def quantize_linear_weights_(model: nn.Module, bits: int) -> None:
    """Per-output-channel symmetric absmax. Biases and activations stay float."""
    if bits < 2:
        raise ValueError(bits)
    qmax = float((1 << (bits - 1)) - 1)
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, nn.Linear):
                continue
            weight = module.weight.data
            amax = weight.abs().amax(dim=1, keepdim=True)
            scale = torch.where(amax > 0, amax / qmax, torch.full_like(amax, 1.0 / qmax))
            quantized = torch.round(weight / scale).clamp(-qmax, qmax)
            module.weight.data.copy_(quantized * scale)


def predict(model: nn.Module, scaler_mean: np.ndarray, scaler_scale: np.ndarray, features: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    scaled = ((features - scaler_mean) / scaler_scale).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(scaled).to(device))
        return logits.argmax(dim=-1).cpu().numpy()


def score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    recall = recall_score(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": recall.astype(np.float64),
    }


def features_for(name: str, audio: np.ndarray, seismic: np.ndarray) -> np.ndarray:
    if name == "audio":
        return audio
    if name == "seismic":
        return seismic
    if name == "audio+seismic":
        return np.concatenate([audio, seismic], axis=1)
    raise KeyError(name)


def plot_heatmaps(
    rows: list[dict],
    class_counts: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    heatmap_split: str,
    split_note: str,
) -> None:
    bit_labels = [str(b) for b in BITS] + ["fp32"]
    recall = {
        name: np.zeros((len(class_names), len(bit_labels)), dtype=np.float64) for name in MODELS
    }
    accuracy = {name: np.zeros(len(bit_labels), dtype=np.float64) for name in MODELS}
    for row in rows:
        if row["split"] != heatmap_split:
            continue
        col = bit_labels.index(row["bits"])
        recall[row["model"]][:, col] = row["recall"]
        accuracy[row["model"]][col] = row["accuracy"]

    row_labels = [f"{name}  n={int(class_counts[i])}" for i, name in enumerate(class_names)]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2), sharey=True)
    image = None
    for ax, name in zip(axes, MODELS):
        grid = recall[name]
        image = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(bit_labels)))
        ax.set_xticklabels(
            [f"{bit}\n{100.0 * accuracy[name][i]:.1f}%" for i, bit in enumerate(bit_labels)],
            fontsize=8,
        )
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xlabel("Weight bits\noverall accuracy")
        ax.set_title(name)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                value = 100.0 * grid[r, c]
                color = "white" if grid[r, c] < 0.55 else "black"
                ax.text(c, r, f"{value:.0f}", ha="center", va="center", color=color, fontsize=7)
    axes[0].set_ylabel(f"Class ({heatmap_split} count)")
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label="Per-class recall")
    fig.suptitle(
        "ACIDS spectral MLP: per-class recall vs weight bit width\n"
        "Direct per-channel symmetric weight quantization, activations float, no retraining. "
        + split_note,
        fontsize=11,
    )
    fig.savefig(output_dir / "recall_heatmaps.png", dpi=160, bbox_inches="tight")
    fig.savefig(output_dir / "recall_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)

    for name in MODELS:
        fig_one, ax = plt.subplots(figsize=(8.2, 6.0))
        grid = recall[name]
        image = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(bit_labels)))
        ax.set_xticklabels(
            [f"{bit}\n{100.0 * accuracy[name][i]:.1f}%" for i, bit in enumerate(bit_labels)],
            fontsize=9,
        )
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Weight bits\noverall accuracy")
        ax.set_ylabel(f"Class ({heatmap_split} count)")
        ax.set_title(f"{name}: per-class recall")
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                value = 100.0 * grid[r, c]
                color = "white" if grid[r, c] < 0.55 else "black"
                ax.text(c, r, f"{value:.0f}", ha="center", va="center", color=color, fontsize=8)
        fig_one.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Per-class recall")
        slug = name.replace("+", "_")
        fig_one.savefig(output_dir / f"recall_heatmap_{slug}.png", dpi=160, bbox_inches="tight")
        plt.close(fig_one)


def main():
    args = parse_args()
    import yaml

    cfg = yaml.safe_load(args.paths_yaml.read_text(encoding="utf-8"))
    output_dir = args.output_dir
    cache_dir = output_dir / "cache"
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
    splits = {
        "train": read_index(Path(cfg["train_index_file"]), args.max_samples),
        "val": read_index(Path(cfg["val_index_file"]), args.max_samples),
        "test": read_index(Path(cfg["test_index_file"]), args.max_samples),
    }
    val_equals_test = splits["val"] == splits["test"]
    if val_equals_test:
        logger.warning(
            "val index equals test index (%d paths). Checkpoint selection and the "
            "heatmap use the same clips, so absolute numbers are optimistic.",
            len(splits["val"]),
        )
        splits_to_load = ("train", "val")
        heatmap_split = "val=test"
        split_note = (
            f"N={len(splits['val'])}; val index == test index, so absolute numbers are optimistic."
        )
    else:
        overlap = set(splits["val"]) & set(splits["test"])
        if overlap:
            raise RuntimeError(f"val and test share {len(overlap)} paths")
        splits_to_load = ("train", "val", "test")
        heatmap_split = "test"
        split_note = f"Held-out test split, N={len(splits['test'])}."

    raw = {}
    for split in splits_to_load:
        paths = splits[split]
        for modality in ("audio", "seismic"):
            x_path = cache_dir / f"{split}_{modality}_X.npy"
            y_path = cache_dir / f"{split}_{modality}_y.npy"
            if args.reuse_cache and x_path.exists() and y_path.exists():
                raw[(split, modality)] = (np.load(x_path), np.load(y_path))
                logger.info("cache %s %s %s", split, modality, raw[(split, modality)][0].shape)
                continue
            logger.info("loading %s %s (%d files)", split, modality, len(paths))
            streams, labels = load_split(paths, modality, args.workers)
            np.save(x_path, streams)
            np.save(y_path, labels)
            raw[(split, modality)] = (streams, labels)

    if val_equals_test:
        for modality in ("audio", "seismic"):
            raw[("test", modality)] = raw[("val", modality)]

    for split in splits_to_load:
        y_audio = raw[(split, "audio")][1]
        y_seismic = raw[(split, "seismic")][1]
        if not np.array_equal(y_audio, y_seismic):
            raise RuntimeError(f"{split} audio/seismic labels disagree")

    features = {}
    checked = False
    for split in splits_to_load:
        for modality in ("audio", "seismic"):
            feat_path = cache_dir / f"{split}_{modality}_feat.npy"
            if args.reuse_cache and feat_path.exists():
                features[(split, modality)] = np.load(feat_path)
                continue
            streams = raw[(split, modality)][0]
            if not checked:
                assert_welch_matches(streams)
                checked = True
            logger.info("Welch %s %s", split, modality)
            feat = welch_matrix(streams)
            np.save(feat_path, feat)
            features[(split, modality)] = feat
    if val_equals_test:
        for modality in ("audio", "seismic"):
            features[("test", modality)] = features[("val", modality)]

    labels = {split: raw[(split, "audio")][1] for split in ("train", "val", "test")}
    num_classes = resolve_num_classes(labels["train"], labels["val"], args.num_classes)
    class_names = resolve_class_names(num_classes, args.class_names)
    logger.info("num_classes=%d class_names=%s", num_classes, class_names)
    rows: list[dict] = []

    for name in MODELS:
        slug = name.replace("+", "_")
        ckpt_path = model_dir / f"{slug}.pth"
        x_train = features_for(name, features[("train", "audio")], features[("train", "seismic")])
        x_val = features_for(name, features[("val", "audio")], features[("val", "seismic")])
        x_test = features_for(name, features[("test", "audio")], features[("test", "seismic")])
        y_train, y_val, y_test = labels["train"], labels["val"], labels["test"]

        if args.reuse_checkpoints and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = SpectralMLP(int(ckpt["in_dim"]), int(ckpt["num_classes"])).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            mean = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
            scale = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
            logger.info("reused %s", ckpt_path)
        else:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            logger.info("training %s in_dim=%d num_classes=%d", name, x_train.shape[1], num_classes)
            model, scaler = train_mlp(
                x_train,
                y_train,
                x_val,
                y_val,
                num_classes,
                device,
                args.epochs,
                args.batch_size,
                args.lr,
            )
            mean = np.asarray(scaler.mean_, dtype=np.float64)
            scale = np.asarray(scaler.scale_, dtype=np.float64)
            torch.save(
                {
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "scaler_mean": mean,
                    "scaler_scale": scale,
                    "in_dim": int(x_train.shape[1]),
                    "num_classes": int(num_classes),
                    "model_name": name,
                    "frontend": "welch",
                    "class_names": class_names,
                    "seed": args.seed,
                    "epochs": args.epochs,
                },
                ckpt_path,
            )
            logger.info("wrote %s", ckpt_path)

        model.to(device)
        float_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        settings = [(str(bits), bits) for bits in BITS] + [("fp32", None)]
        eval_splits = [("val=test", x_val, y_val)] if val_equals_test else [
            ("val", x_val, y_val),
            ("test", x_test, y_test),
        ]
        for split_name, x_split, y_split in eval_splits:
            for bit_label, bits in settings:
                model.load_state_dict(float_state)
                model.to(device)
                if bits is not None:
                    quantize_linear_weights_(model, bits)
                pred = predict(model, mean, scale, x_split, device)
                metrics = score(y_split, pred, num_classes)
                logger.info(
                    "%s %s bits=%s acc=%.4f macro_f1=%.4f",
                    name,
                    split_name,
                    bit_label,
                    metrics["accuracy"],
                    metrics["macro_f1"],
                )
                rows.append(
                    {
                        "model": name,
                        "split": split_name,
                        "bits": bit_label,
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "recall": metrics["recall"],
                    }
                )

    csv_path = output_dir / "per_class_recall.csv"
    fieldnames = ["model", "split", "bits", "accuracy", "macro_f1"] + [
        f"recall_{class_name}" for class_name in class_names
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {
                "model": row["model"],
                "split": row["split"],
                "bits": row["bits"],
                "accuracy": f"{row['accuracy']:.6f}",
                "macro_f1": f"{row['macro_f1']:.6f}",
            }
            for class_name, value in zip(class_names, row["recall"]):
                flat[f"recall_{class_name}"] = f"{float(value):.6f}"
            writer.writerow(flat)

    class_counts = np.bincount(
        labels["val" if val_equals_test else "test"], minlength=len(class_names)
    )
    plot_heatmaps(rows, class_counts, class_names, output_dir, heatmap_split, split_note)
    summary = {
        "test_samples": int(len(labels["test"])),
        "val_samples": int(len(labels["val"])),
        "train_samples": int(len(labels["train"])),
        "num_classes": int(num_classes),
        "class_names": class_names,
        "heatmap_class_counts": class_counts.tolist(),
        "bits": list(BITS) + ["fp32"],
        "quantization": "per-output-channel symmetric absmax, qmax=2^(bits-1)-1, activations and biases float",
        "checkpoint_selection": "best validation accuracy",
        "heatmap_split": heatmap_split,
        "val_equals_test": val_equals_test,
        "split_note": split_note,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", output_dir / "recall_heatmaps.png")


if __name__ == "__main__":
    main()
