"""
Sliding-window binary classifier evaluation.

Runs a trained 2-class model (default: binary kona vs wagoneer) on a split,
aggregates per-sample softmax probabilities inside a sliding window of size W,
and classifies each window as class 0 / class 1 / uncertain based on per-class
confidence thresholds.

For threshold selection we sweep a (T_a, T_b) grid and return:
  - the Pareto-optimal frontier on (coverage, accuracy_on_covered)
  - a recommended (T_a, T_b) closest (L2) to the ideal point (max coverage, max
    accuracy_on_covered over the sweep)

Usage (from src2/train_test):

    python3 sliding_window_eval.py \\
        --experiment_dir ../experiments/<run_dir> \\
        --split test \\
        --window_size 10 \\
        --stride 1 \\
        --gpu 0

    # Apply a fixed threshold instead of sweeping:
    python3 sliding_window_eval.py \\
        --experiment_dir ../experiments/<run_dir> \\
        --fixed_threshold_a 0.7 \\
        --fixed_threshold_b 0.7

Outputs a new subdirectory under experiment_dir:
    sliding_window_eval_YYYYMMDD_HHMMSS/
      ├── logs/eval.log
      ├── results.json      # per-window predictions + Pareto sweep summary
      └── pareto_plot.png   # coverage vs accuracy (only if sweep is run)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

src2_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src2_path))

from data_augmenter import apply_augmentation, create_augmenter
from dataset_utils.MultiModalDataLoader import create_dataloaders
from models.create_models import create_single_modal_model
from train_test.normalize import setup_normalization
from train_test.train_test_utils import load_checkpoint

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


UNCERTAIN_LABEL = -1


def parse_args():
    p = argparse.ArgumentParser(
        description="Sliding-window binary evaluator with Pareto thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument(
        "--split",
        type=str,
        nargs="+",
        choices=("train", "val", "test"),
        default=["test"],
        help=(
            "One or more splits to evaluate. Multiple splits are concatenated; "
            "sliding windows stay within each split (no cross-split windows)."
        ),
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Number of consecutive samples aggregated per window",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Step size for the sliding window (1 = dense overlap)",
    )
    p.add_argument(
        "--aggregation",
        choices=("mean_softmax",),
        default="mean_softmax",
        help="How to aggregate per-sample probabilities within a window",
    )
    p.add_argument(
        "--window_label_rule",
        choices=("majority", "center"),
        default="majority",
        help="Ground-truth label assigned to a window for accuracy scoring",
    )
    p.add_argument(
        "--threshold_grid_steps",
        type=int,
        default=21,
        help="Grid resolution per class for the (T_a, T_b) sweep (default 21)",
    )
    p.add_argument(
        "--threshold_min",
        type=float,
        default=0.50,
    )
    p.add_argument(
        "--threshold_max",
        type=float,
        default=0.99,
    )
    p.add_argument(
        "--fixed_threshold_a",
        type=float,
        default=None,
        help="Class-0 threshold. If given with --fixed_threshold_b, skip sweep.",
    )
    p.add_argument(
        "--fixed_threshold_b",
        type=float,
        default=None,
        help="Class-1 threshold. If given with --fixed_threshold_a, skip sweep.",
    )
    return p.parse_args()


def _resolve_experiment_block(config):
    experiment_name = config["experiment_name"]
    if "distillation" in config and experiment_name in config["distillation"]:
        experiment_config = config["distillation"][experiment_name]
        model_name = experiment_config["models"][0]
        loss_source_config = experiment_config["stages"][0]
    else:
        experiment_config = config["experiments"][experiment_name]
        model_name = experiment_config["model"]
        training_config_name = experiment_config["training"]
        loss_source_config = config["training_configs"][training_config_name]
    return experiment_config, model_name, loss_source_config


def _build_model_and_loaders(config, experiment_dir, checkpoint_path_cli, device):
    """Mirror confidence_analysis.py setup: loaders + normalization + model + ckpt."""
    experiment_config, model_name, loss_source_config = _resolve_experiment_block(
        config
    )
    model_config = config["models"][model_name]

    logging.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config=config)

    skip_normalization = False
    if (
        isinstance(loss_source_config, dict)
        and loss_source_config.get("type") == "finetune"
    ):
        skip_normalization = True

    if skip_normalization:
        logging.info("Skipping normalization setup to match finetune.py behavior")
    else:
        logging.info("Setting up normalization...")
        train_loader, val_loader, test_loader = setup_normalization(
            train_loader, val_loader, test_loader, config
        )

    logging.info("Creating augmenter (eval mode: disabled)...")
    augmenter = create_augmenter(
        config, augmentation_mode="no", experiment_config=experiment_config
    )

    logging.info("Creating model...")
    config["models"][model_name]["pretrain_mode"] = False
    model = create_single_modal_model(config, model_name)

    if checkpoint_path_cli:
        checkpoint_path = Path(checkpoint_path_cli)
    else:
        checkpoint_path = experiment_dir / "models" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    model = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()

    return (
        model,
        augmenter,
        train_loader,
        val_loader,
        test_loader,
        model_name,
        checkpoint_path,
    )


def _make_deterministic_loader(src_loader):
    """
    Rebuild a DataLoader over the same dataset with shuffle=False and no sampler.

    Needed for the train split, which is normally served with shuffle=True or
    a WeightedRandomSampler; neither yields the deterministic index-file order
    required for sliding-window aggregation.
    """
    dataset = src_loader.dataset
    batch_size = src_loader.batch_size
    num_workers = src_loader.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def collect_logits_labels(model, loader, device, augmenter):
    """
    Forward only; returns float numpy arrays logits [N, C], labels [N],
    and a numpy int64 array indices [N] of dataset-local sample indices.

    Indices come from the dataset's __getitem__ third element and let us
    map each row back to its source .pt file via dataset.sample_files[idx].
    """
    model.eval()
    logits_list = []
    labels_list = []
    idx_list = []
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                data, labels, ds_idx = batch_data
            else:
                data, labels = batch_data[0], batch_data[1]
                ds_idx = None

            if augmenter is not None:
                data, labels = apply_augmentation(augmenter, data, labels)

            labels = labels.to(device)
            if isinstance(data, dict):
                for loc in data:
                    for mod in data[loc]:
                        data[loc][mod] = data[loc][mod].to(device)
            else:
                data = data.to(device)

            if len(labels.shape) == 2 and labels.shape[1] > 1:
                loss_labels = torch.argmax(labels, dim=1)
            else:
                loss_labels = labels.long()

            outputs = model(data)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            logits_list.append(logits.float().cpu().numpy())
            labels_list.append(loss_labels.cpu().numpy())
            if ds_idx is not None:
                idx_list.append(np.asarray(ds_idx).astype(np.int64).reshape(-1))

    logits_np = np.concatenate(logits_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0).astype(np.int64)
    if idx_list:
        idx_np = np.concatenate(idx_list, axis=0)
    else:
        idx_np = np.arange(len(labels_np), dtype=np.int64)
    return logits_np, labels_np, idx_np


def _parse_utc_to_epoch(utc_str):
    """
    Parse the .pt's `window_center_utc` (e.g. "2026-04-14T17:38:32+0000") into
    Unix seconds. Returns None if missing/unparseable.

    `datetime.fromisoformat` doesn't accept the "+0000" form before Python 3.11,
    so we normalize it to "+00:00" before parsing.
    """
    if utc_str is None:
        return None
    try:
        s = str(utc_str)
        if len(s) >= 5 and (s[-5] in "+-") and s[-3] != ":":
            s = s[:-2] + ":" + s[-2:]
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except Exception:
        return None


def _load_sample_metadata(sample_paths):
    """
    Load gps, node, distance, and identifier fields from each .pt file.

    Returns a list of dicts (one per path) with keys:
      path, run_id, node_id, window_center_utc, time_epoch, vehicle,
      distance_m, gps_lat, gps_lon, gps_elev, node_lat, node_lon
    """
    meta = []
    for p in sample_paths:
        try:
            s = torch.load(p, weights_only=False)
        except Exception as e:
            logging.warning(f"Failed to load metadata from {p}: {e}")
            meta.append({"path": str(p)})
            continue
        gps = s["gps"] if "gps" in s else {}
        node = s["node"] if "node" in s else {}
        utc_str = (
            str(s["window_center_utc"]) if "window_center_utc" in s else None
        )
        meta.append(
            {
                "path": str(p),
                "run_id": int(s["run_id"]) if "run_id" in s else None,
                "node_id": int(s["node_id"]) if "node_id" in s else None,
                "window_center_utc": utc_str,
                "time_epoch": _parse_utc_to_epoch(utc_str),
                "vehicle": str(s["vehicle"]) if "vehicle" in s else None,
                "distance_m": (
                    float(s["distance_m"]) if "distance_m" in s else None
                ),
                "gps_lat": float(gps["lat"]) if "lat" in gps else None,
                "gps_lon": float(gps["lon"]) if "lon" in gps else None,
                "gps_elev": float(gps["elev"]) if "elev" in gps else None,
                "node_lat": float(node["lat"]) if "lat" in node else None,
                "node_lon": float(node["lon"]) if "lon" in node else None,
            }
        )
    return meta


def _status_from(true_int: int, pred_int: int) -> str:
    if pred_int == UNCERTAIN_LABEL:
        return "unclassified"
    return "correct" if pred_int == true_int else "wrong"


def _write_per_sample_csv(rows: list, path: Path, class_names: list):
    headers = [
        "global_index",
        "split",
        "split_local_index",
        "path",
        "run_id",
        "node_id",
        "window_center_utc",
        "time_epoch",
        "time_s_from_run_start",
        "latitude",
        "longitude",
        "node_lat",
        "node_lon",
        "distance_m",
        "true_label_name",
        f"p_{class_names[0]}",
        f"p_{class_names[1]}",
        "predicted_label_name",
        "status",
    ]
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            parts = []
            for h in headers:
                v = r[h] if h in r else None
                if v is None:
                    parts.append("")
                elif isinstance(v, float):
                    parts.append(f"{v:.8f}")
                else:
                    parts.append(str(v))
            f.write(",".join(parts) + "\n")


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / ex.sum(axis=1, keepdims=True)


def slide_windows(
    probs: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
    label_rule: str,
):
    """
    Build sliding-window aggregates.

    probs: [N, C] per-sample softmax probabilities
    labels: [N] per-sample integer class labels

    Returns:
      win_probs: [W, C] mean softmax per window
      win_labels: [W] ground-truth label assigned to each window
      win_starts: [W] starting index of each window in the original sequence
    """
    n, c = probs.shape
    if window_size <= 0:
        raise ValueError("window_size must be >= 1")
    if stride <= 0:
        raise ValueError("stride must be >= 1")
    if window_size > n:
        raise ValueError(
            f"window_size ({window_size}) exceeds number of samples ({n})"
        )

    starts = list(range(0, n - window_size + 1, stride))
    win_probs = np.zeros((len(starts), c), dtype=np.float64)
    win_labels = np.zeros(len(starts), dtype=np.int64)
    for w_i, s in enumerate(starts):
        chunk_probs = probs[s : s + window_size]
        chunk_labels = labels[s : s + window_size]
        win_probs[w_i] = chunk_probs.mean(axis=0)
        if label_rule == "center":
            win_labels[w_i] = int(chunk_labels[window_size // 2])
        elif label_rule == "majority":
            counts = np.bincount(chunk_labels, minlength=c)
            win_labels[w_i] = int(counts.argmax())
        else:
            raise ValueError(f"unknown label_rule: {label_rule}")
    return win_probs, win_labels, np.array(starts, dtype=np.int64)


def classify_windows(win_probs: np.ndarray, t_a: float, t_b: float) -> np.ndarray:
    """
    Per-window prediction under per-class thresholds:
      - class 0 if p[0] >= t_a (and ties broken in favor of 0 if p[1] also >= t_b)
      - class 1 if p[1] >= t_b
      - UNCERTAIN_LABEL (-1) otherwise

    If BOTH thresholds are met we pick the higher-probability class (can't normally
    happen with softmax in 2-class when both > 0.5, but works for any thresholds).
    """
    w = win_probs.shape[0]
    preds = np.full(w, UNCERTAIN_LABEL, dtype=np.int64)
    p0 = win_probs[:, 0]
    p1 = win_probs[:, 1]
    sel0 = p0 >= t_a
    sel1 = p1 >= t_b
    both = sel0 & sel1
    only0 = sel0 & ~both
    only1 = sel1 & ~both
    preds[only0] = 0
    preds[only1] = 1
    preds[both] = np.where(p0[both] >= p1[both], 0, 1)
    return preds


def _confusion_binary(labels: np.ndarray, preds: np.ndarray):
    """3x2 confusion: rows=true (0,1), cols=pred (0,1,uncertain)."""
    cm = np.zeros((2, 3), dtype=np.int64)
    for t, p in zip(labels, preds):
        if p == UNCERTAIN_LABEL:
            cm[int(t), 2] += 1
        else:
            cm[int(t), int(p)] += 1
    return cm


def _metrics_for_threshold(
    win_probs: np.ndarray,
    win_labels: np.ndarray,
    t_a: float,
    t_b: float,
):
    preds = classify_windows(win_probs, t_a, t_b)
    n_total = int(win_probs.shape[0])
    covered_mask = preds != UNCERTAIN_LABEL
    n_covered = int(covered_mask.sum())
    coverage = float(n_covered / n_total) if n_total > 0 else 0.0
    if n_covered == 0:
        accuracy = 0.0
        precision_0 = 0.0
        precision_1 = 0.0
        recall_0 = 0.0
        recall_1 = 0.0
    else:
        correct = int((preds[covered_mask] == win_labels[covered_mask]).sum())
        accuracy = float(correct / n_covered)

        pred_is_0 = preds == 0
        pred_is_1 = preds == 1
        precision_0 = (
            float(((preds == 0) & (win_labels == 0)).sum() / max(1, pred_is_0.sum()))
            if pred_is_0.sum() > 0
            else 0.0
        )
        precision_1 = (
            float(((preds == 1) & (win_labels == 1)).sum() / max(1, pred_is_1.sum()))
            if pred_is_1.sum() > 0
            else 0.0
        )
        true_is_0 = win_labels == 0
        true_is_1 = win_labels == 1
        recall_0 = (
            float(((preds == 0) & true_is_0).sum() / max(1, true_is_0.sum()))
            if true_is_0.sum() > 0
            else 0.0
        )
        recall_1 = (
            float(((preds == 1) & true_is_1).sum() / max(1, true_is_1.sum()))
            if true_is_1.sum() > 0
            else 0.0
        )
    return {
        "t_a": float(t_a),
        "t_b": float(t_b),
        "coverage": coverage,
        "accuracy_on_covered": accuracy,
        "n_total": n_total,
        "n_covered": n_covered,
        "precision_class0": precision_0,
        "precision_class1": precision_1,
        "recall_class0": recall_0,
        "recall_class1": recall_1,
    }


def pareto_maximize_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True at i if (a[i], b[i]) is non-dominated w.r.t. simultaneous max of a,b."""
    n = int(a.shape[0])
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if a[j] >= a[i] and b[j] >= b[i] and (a[j] > a[i] or b[j] > b[i]):
                keep[i] = False
                break
    return keep


def sweep_thresholds(
    win_probs: np.ndarray,
    win_labels: np.ndarray,
    t_min: float,
    t_max: float,
    steps: int,
):
    grid = np.linspace(t_min, t_max, steps)
    rows = []
    for t_a in grid:
        for t_b in grid:
            rows.append(
                _metrics_for_threshold(
                    win_probs, win_labels, float(t_a), float(t_b)
                )
            )

    cov = np.array([r["coverage"] for r in rows], dtype=np.float64)
    acc = np.array([r["accuracy_on_covered"] for r in rows], dtype=np.float64)
    mask = pareto_maximize_mask(cov, acc)
    pareto_rows = [rows[i] for i in range(len(rows)) if mask[i]]

    ideal_cov = float(cov.max()) if len(cov) else 0.0
    ideal_acc = float(acc.max()) if len(acc) else 0.0
    best = None
    best_d = 1e18
    for i, r in enumerate(rows):
        if not mask[i]:
            continue
        d = (ideal_cov - cov[i]) ** 2 + (ideal_acc - acc[i]) ** 2
        if d < best_d:
            best_d = d
            best = dict(r)
    if best is not None:
        best["l2_gap_to_ideal"] = float(np.sqrt(best_d))
        best["ideal_point"] = {
            "coverage": ideal_cov,
            "accuracy_on_covered": ideal_acc,
        }
        best["selection_rule"] = (
            "pareto_max(coverage, accuracy_on_covered)_then_l2_closest_to_ideal"
        )
    return rows, pareto_rows, best


def _build_per_window_rows(
    win_probs: np.ndarray,
    win_labels: np.ndarray,
    win_starts: np.ndarray,
    win_split: np.ndarray,
    window_size: int,
    class_names: list,
    t_a: float,
    t_b: float,
):
    """Per-window breakdown suitable for JSON/CSV output."""
    preds = classify_windows(win_probs, t_a, t_b)
    rows = []
    for i in range(len(win_labels)):
        pred = int(preds[i])
        true = int(win_labels[i])
        if pred == UNCERTAIN_LABEL:
            pred_name = "uncertain"
            correct = None
        else:
            pred_name = class_names[pred]
            correct = bool(pred == true)
        rows.append(
            {
                "window_index": int(i),
                "split": str(win_split[i]),
                "start_index_in_split": int(win_starts[i]),
                "end_index_in_split": int(win_starts[i]) + int(window_size) - 1,
                "true_label": true,
                "true_label_name": class_names[true],
                f"mean_p_{class_names[0]}": float(win_probs[i, 0]),
                f"mean_p_{class_names[1]}": float(win_probs[i, 1]),
                "predicted_label": pred,
                "predicted_label_name": pred_name,
                "correct": correct,
            }
        )
    return rows


def _write_per_window_csv(rows: list, path: Path, class_names: list):
    """Small CSV so you can eyeball per-window predictions."""
    if not rows:
        return
    headers = [
        "window_index",
        "split",
        "start_index_in_split",
        "end_index_in_split",
        "true_label",
        "true_label_name",
        f"mean_p_{class_names[0]}",
        f"mean_p_{class_names[1]}",
        "predicted_label",
        "predicted_label_name",
        "correct",
    ]
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(
                ",".join(
                    ""
                    if r[h] is None
                    else (
                        f"{r[h]:.6f}"
                        if isinstance(r[h], float)
                        else str(r[h])
                    )
                    for h in headers
                )
                + "\n"
            )


def _plot_pareto(all_rows, pareto_rows, best, out_path):
    if not _HAS_MPL:
        return
    cov = np.array([r["coverage"] for r in all_rows])
    acc = np.array([r["accuracy_on_covered"] for r in all_rows])
    p_cov = np.array([r["coverage"] for r in pareto_rows])
    p_acc = np.array([r["accuracy_on_covered"] for r in pareto_rows])
    order = np.argsort(p_cov)
    p_cov_s, p_acc_s = p_cov[order], p_acc[order]

    plt.figure(figsize=(7, 5))
    plt.scatter(cov, acc, s=10, alpha=0.3, label="sweep points")
    plt.plot(p_cov_s, p_acc_s, "r-o", label="Pareto frontier", markersize=5)
    if best is not None:
        plt.scatter(
            [best["coverage"]],
            [best["accuracy_on_covered"]],
            color="gold",
            edgecolor="black",
            s=120,
            zorder=10,
            label="recommended",
        )
    plt.xlabel("Coverage  (1 - uncertain_rate)")
    plt.ylabel("Accuracy on covered windows")
    plt.title("Sliding-window Pareto: coverage vs accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    args = parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    config["device"] = str(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = experiment_dir / f"sliding_window_eval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(logs_dir / "eval.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logging.info("=" * 80)
    logging.info("SLIDING WINDOW EVALUATION")
    logging.info("=" * 80)
    logging.info(f"experiment_dir: {experiment_dir}")
    logging.info(f"splits: {args.split}")
    logging.info(f"window_size={args.window_size}, stride={args.stride}")
    logging.info(f"aggregation={args.aggregation}, label_rule={args.window_label_rule}")
    logging.info(f"device: {device}")

    task_name = config["task_name"]
    class_names = config[task_name]["class_names"]
    num_classes = config[task_name]["num_classes"]
    if num_classes != 2:
        raise ValueError(
            "sliding_window_eval is scoped to binary (2-class) tasks; "
            f"got num_classes={num_classes} for task '{task_name}'."
        )
    logging.info(f"task: {task_name}  classes: {class_names}")

    (
        model,
        augmenter,
        train_loader_orig,
        val_loader,
        test_loader,
        model_name,
        checkpoint_path,
    ) = _build_model_and_loaders(config, experiment_dir, args.checkpoint_path, device)

    split_to_loader = {
        "train": _make_deterministic_loader(train_loader_orig),
        "val": val_loader,
        "test": test_loader,
    }
    splits = list(args.split)

    per_sample_rows: list = []
    win_probs_chunks: list = []
    win_labels_chunks: list = []
    win_starts_chunks: list = []
    win_split_chunks: list = []
    per_split_summary: dict = {}
    sample_offset = 0
    per_split_probs: dict = {}
    per_split_labels: dict = {}

    for split_name in splits:
        loader = split_to_loader[split_name]
        logging.info(f"Running forward pass on split={split_name}...")
        logits_s, labels_s, idx_s = collect_logits_labels(
            model, loader, device, augmenter
        )
        probs_s = softmax_np(logits_s)
        logging.info(
            f"  [{split_name}] collected {len(labels_s)} samples, "
            f"shape={probs_s.shape}"
        )

        dataset = loader.dataset
        sample_paths_s = [dataset.sample_files[int(j)] for j in idx_s]
        logging.info(f"  [{split_name}] loading per-sample metadata (gps, node)...")
        meta_s = _load_sample_metadata(sample_paths_s)

        per_split_probs[split_name] = probs_s
        per_split_labels[split_name] = labels_s

        # Per (run_id, node_id) time origin so each run gets its own 0s anchor.
        # Falls back to None if any sample in the group lacks a parseable timestamp.
        run_node_t0: dict = {}
        for m in meta_s:
            r = m["run_id"] if "run_id" in m else None
            n = m["node_id"] if "node_id" in m else None
            t = m["time_epoch"] if "time_epoch" in m else None
            if t is None:
                continue
            key = (r, n)
            if key not in run_node_t0 or t < run_node_t0[key]:
                run_node_t0[key] = t

        for i in range(len(labels_s)):
            m = meta_s[i]
            t_epoch = m["time_epoch"] if "time_epoch" in m else None
            r = m["run_id"] if "run_id" in m else None
            n = m["node_id"] if "node_id" in m else None
            t0 = run_node_t0[(r, n)] if (r, n) in run_node_t0 else None
            t_rel = (
                float(t_epoch - t0) if (t_epoch is not None and t0 is not None) else None
            )
            per_sample_rows.append(
                {
                    "global_index": int(sample_offset + i),
                    "split": split_name,
                    "split_local_index": int(i),
                    "true_label": int(labels_s[i]),
                    "true_label_name": class_names[int(labels_s[i])],
                    f"p_{class_names[0]}": float(probs_s[i, 0]),
                    f"p_{class_names[1]}": float(probs_s[i, 1]),
                    "argmax_label": int(np.argmax(probs_s[i])),
                    "argmax_label_name": class_names[int(np.argmax(probs_s[i]))],
                    "path": m["path"] if "path" in m else None,
                    "run_id": r,
                    "node_id": n,
                    "window_center_utc": (
                        m["window_center_utc"] if "window_center_utc" in m else None
                    ),
                    "time_epoch": t_epoch,
                    "time_s_from_run_start": t_rel,
                    "latitude": m["gps_lat"] if "gps_lat" in m else None,
                    "longitude": m["gps_lon"] if "gps_lon" in m else None,
                    "node_lat": m["node_lat"] if "node_lat" in m else None,
                    "node_lon": m["node_lon"] if "node_lon" in m else None,
                    "distance_m": m["distance_m"] if "distance_m" in m else None,
                }
            )
        sample_offset += len(labels_s)

        if len(labels_s) < args.window_size:
            logging.warning(
                f"  [{split_name}] has {len(labels_s)} samples < window_size="
                f"{args.window_size}; skipping window aggregation for this split."
            )
            per_split_summary[split_name] = {
                "num_samples": int(len(labels_s)),
                "num_windows": 0,
            }
            continue

        win_probs_s, win_labels_s, win_starts_s = slide_windows(
            probs_s,
            labels_s,
            window_size=args.window_size,
            stride=args.stride,
            label_rule=args.window_label_rule,
        )
        win_probs_chunks.append(win_probs_s)
        win_labels_chunks.append(win_labels_s)
        win_starts_chunks.append(win_starts_s)
        win_split_chunks.append(
            np.array([split_name] * len(win_labels_s), dtype=object)
        )
        per_split_summary[split_name] = {
            "num_samples": int(len(labels_s)),
            "num_windows": int(len(win_labels_s)),
            "window_label_counts": {
                class_names[0]: int((win_labels_s == 0).sum()),
                class_names[1]: int((win_labels_s == 1).sum()),
            },
        }
        logging.info(
            f"  [{split_name}] windows={len(win_labels_s)}  "
            f"(kona={(win_labels_s == 0).sum()}, "
            f"wagoneer={(win_labels_s == 1).sum()})"
        )

    if not win_probs_chunks:
        raise RuntimeError(
            "No split produced any sliding windows (all splits smaller than "
            f"window_size={args.window_size}). Nothing to evaluate."
        )

    win_probs = np.concatenate(win_probs_chunks, axis=0)
    win_labels = np.concatenate(win_labels_chunks, axis=0)
    win_starts = np.concatenate(win_starts_chunks, axis=0)
    win_split = np.concatenate(win_split_chunks, axis=0)
    logging.info(
        f"Total windows across splits ({'+'.join(splits)}): {len(win_labels)} "
        f"(kona={(win_labels == 0).sum()}, wagoneer={(win_labels == 1).sum()})"
    )

    results: dict = {
        "experiment_dir": str(experiment_dir),
        "checkpoint_path": str(checkpoint_path),
        "splits": splits,
        "task_name": task_name,
        "class_names": class_names,
        "num_samples": int(sample_offset),
        "window_size": args.window_size,
        "stride": args.stride,
        "aggregation": args.aggregation,
        "window_label_rule": args.window_label_rule,
        "num_windows": int(len(win_labels)),
        "window_label_counts": {
            class_names[0]: int((win_labels == 0).sum()),
            class_names[1]: int((win_labels == 1).sum()),
        },
        "per_split_summary": per_split_summary,
    }

    if args.fixed_threshold_a is not None and args.fixed_threshold_b is not None:
        logging.info(
            f"Applying fixed thresholds: T_a={args.fixed_threshold_a}, "
            f"T_b={args.fixed_threshold_b}"
        )
        metrics = _metrics_for_threshold(
            win_probs,
            win_labels,
            args.fixed_threshold_a,
            args.fixed_threshold_b,
        )
        preds = classify_windows(
            win_probs, args.fixed_threshold_a, args.fixed_threshold_b
        )
        cm = _confusion_binary(win_labels, preds)
        logging.info(
            f"coverage={metrics['coverage']:.4f}  "
            f"accuracy_on_covered={metrics['accuracy_on_covered']:.4f}  "
            f"covered={metrics['n_covered']}/{metrics['n_total']}"
        )
        logging.info(
            f"Confusion matrix (rows=true [{class_names[0]},{class_names[1]}], "
            f"cols=pred [{class_names[0]},{class_names[1]},uncertain]):\n{cm}"
        )
        results["mode"] = "fixed_threshold"
        results["metrics"] = metrics
        results["confusion_matrix"] = cm.tolist()
        results["confusion_matrix_rows"] = [f"true_{c}" for c in class_names]
        results["confusion_matrix_cols"] = [
            f"pred_{class_names[0]}",
            f"pred_{class_names[1]}",
            "pred_uncertain",
        ]
        per_window_rows = _build_per_window_rows(
            win_probs,
            win_labels,
            win_starts,
            win_split,
            args.window_size,
            class_names,
            args.fixed_threshold_a,
            args.fixed_threshold_b,
        )
        results["per_window_predictions"] = per_window_rows
        results["per_window_thresholds_used"] = {
            f"t_{class_names[0]}": args.fixed_threshold_a,
            f"t_{class_names[1]}": args.fixed_threshold_b,
        }
        _write_per_window_csv(
            per_window_rows, out_dir / "per_window.csv", class_names
        )
        logging.info(f"Per-window CSV saved: {out_dir / 'per_window.csv'}")
    else:
        logging.info(
            f"Sweeping thresholds on [{args.threshold_min}, {args.threshold_max}] "
            f"with {args.threshold_grid_steps} steps per class..."
        )
        all_rows, pareto_rows, best = sweep_thresholds(
            win_probs,
            win_labels,
            args.threshold_min,
            args.threshold_max,
            args.threshold_grid_steps,
        )
        if best is not None:
            logging.info(
                f"Recommended: T_{class_names[0]}={best['t_a']:.3f}, "
                f"T_{class_names[1]}={best['t_b']:.3f}  "
                f"coverage={best['coverage']:.4f}  "
                f"accuracy_on_covered={best['accuracy_on_covered']:.4f}"
            )
            preds = classify_windows(win_probs, best["t_a"], best["t_b"])
            cm = _confusion_binary(win_labels, preds)
            logging.info(
                f"Confusion @ recommended (rows=true, cols=pred + uncertain):\n{cm}"
            )
            results["recommended_confusion_matrix"] = cm.tolist()
            per_window_rows = _build_per_window_rows(
                win_probs,
                win_labels,
                win_starts,
                win_split,
                args.window_size,
                class_names,
                best["t_a"],
                best["t_b"],
            )
            results["per_window_predictions"] = per_window_rows
            results["per_window_thresholds_used"] = {
                f"t_{class_names[0]}": best["t_a"],
                f"t_{class_names[1]}": best["t_b"],
                "source": "pareto_recommended",
            }
            _write_per_window_csv(
                per_window_rows, out_dir / "per_window.csv", class_names
            )
            logging.info(f"Per-window CSV saved: {out_dir / 'per_window.csv'}")
        else:
            logging.warning("No Pareto-optimal point found (empty sweep?).")
        results["mode"] = "sweep"
        results["pareto_recommended"] = best
        results["pareto_optimal_points"] = pareto_rows
        results["sweep_all_points"] = all_rows
        results["confusion_matrix_rows"] = [f"true_{c}" for c in class_names]
        results["confusion_matrix_cols"] = [
            f"pred_{class_names[0]}",
            f"pred_{class_names[1]}",
            "pred_uncertain",
        ]
        if _HAS_MPL:
            _plot_pareto(all_rows, pareto_rows, best, out_dir / "pareto_plot.png")
            logging.info(f"Pareto plot saved: {out_dir / 'pareto_plot.png'}")

    if args.fixed_threshold_a is not None and args.fixed_threshold_b is not None:
        t_a_sample = float(args.fixed_threshold_a)
        t_b_sample = float(args.fixed_threshold_b)
        threshold_source = "fixed_cli"
    elif (
        results.get("mode") == "sweep"
        and results.get("pareto_recommended") is not None
    ):
        t_a_sample = float(results["pareto_recommended"]["t_a"])
        t_b_sample = float(results["pareto_recommended"]["t_b"])
        threshold_source = "pareto_recommended"
    else:
        t_a_sample = 0.5
        t_b_sample = 0.5
        threshold_source = "fallback_0.5"

    logging.info(
        f"Per-sample thresholds applied: "
        f"T_{class_names[0]}={t_a_sample:.4f}, "
        f"T_{class_names[1]}={t_b_sample:.4f}  (source={threshold_source})"
    )
    n_correct = 0
    n_wrong = 0
    n_uncertain = 0
    for r in per_sample_rows:
        p0 = r[f"p_{class_names[0]}"]
        p1 = r[f"p_{class_names[1]}"]
        meets0 = p0 >= t_a_sample
        meets1 = p1 >= t_b_sample
        if meets0 and meets1:
            pred_int = 0 if p0 >= p1 else 1
        elif meets0:
            pred_int = 0
        elif meets1:
            pred_int = 1
        else:
            pred_int = UNCERTAIN_LABEL
        r["predicted_label"] = pred_int
        r["predicted_label_name"] = (
            "uncertain" if pred_int == UNCERTAIN_LABEL else class_names[pred_int]
        )
        r["status"] = _status_from(int(r["true_label"]), pred_int)
        if r["status"] == "correct":
            n_correct += 1
        elif r["status"] == "wrong":
            n_wrong += 1
        else:
            n_uncertain += 1

    n_total_samples = len(per_sample_rows)
    logging.info(
        f"Per-sample summary ({n_total_samples} inferences): "
        f"correct={n_correct}, wrong={n_wrong}, unclassified={n_uncertain}  "
        f"(accuracy_on_classified="
        f"{n_correct / max(1, n_correct + n_wrong):.4f}, "
        f"coverage={(n_correct + n_wrong) / max(1, n_total_samples):.4f})"
    )

    per_sample_csv = out_dir / "per_sample.csv"
    _write_per_sample_csv(per_sample_rows, per_sample_csv, class_names)
    logging.info(f"Per-sample CSV saved: {per_sample_csv}")

    results["per_sample_predictions"] = per_sample_rows
    results["per_sample_thresholds"] = {
        f"t_{class_names[0]}": t_a_sample,
        f"t_{class_names[1]}": t_b_sample,
        "source": threshold_source,
    }
    results["per_sample_summary"] = {
        "n_total": n_total_samples,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_unclassified": n_uncertain,
        "accuracy_on_classified": float(
            n_correct / max(1, n_correct + n_wrong)
        ),
        "coverage": float(
            (n_correct + n_wrong) / max(1, n_total_samples)
        ),
    }

    results_file = out_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results written: {results_file}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
