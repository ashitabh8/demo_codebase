"""
Post-hoc analysis for single-label multiclass (softmax) models.

Primary deliverables:
  - **Pareto-optimal abstention thresholds** (max prob < tau -> background): non-dominated
    points in (accuracy, macro_f1), plus a **recommended tau** (closest L2 to the ideal
    point = max accuracy and max macro_f1 over the sweep). Each sweep row includes
    **correct_count** / **n_total** / **accuracy** like test.py.
  - **Bootstrap 95% intervals** for metrics and for the Pareto-recommended tau on each
    resample (same rule).

Also reports:
  - Temperature T* (grid NLL), ECE / NLL point estimates, reliability PNGs
  - Optional one-vs-rest threshold sweeps (--skip_ovr to omit large JSON)

Usage (from src2/train_test):
  python3 confidence_analysis.py --experiment_dir ../experiments/<run_dir> --gpu 0
  python3 confidence_analysis.py ... --bootstrap_replicates 1000 --split test

Notes:
  - CIs are from **i.i.d. resampling rows** (logits/labels); for grouped data consider
    cluster bootstrap separately.
  - Fitting T* on the full split then bootstrapping metrics uses a **fixed T*** each
    replicate (common for reporting calibrated metrics).
  - Requires sklearn.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, log_loss

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


def parse_args():
    p = argparse.ArgumentParser(
        description="Multiclass softmax confidence and calibration analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Experiment directory (config.yaml + models/)",
    )
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint .pth (default: models/best_model.pth under experiment_dir)",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=("val", "test"),
        default="val",
        help="Which dataloader split to run (default: val)",
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id (-1 for CPU)",
    )
    p.add_argument(
        "--n_bins_ece",
        type=int,
        default=15,
        help="Number of bins for multiclass ECE (max-confidence)",
    )
    p.add_argument(
        "--temperature_grid",
        type=int,
        default=120,
        help="Number of points in log-spaced grid for temperature search",
    )
    p.add_argument(
        "--bootstrap_replicates",
        type=int,
        default=500,
        help="Number of bootstrap resamples for 95%% percentile CIs (0 to skip)",
    )
    p.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
        help="RNG seed for bootstrap resampling",
    )
    p.add_argument(
        "--skip_ovr",
        action="store_true",
        help="Omit one-vs-rest threshold curves from JSON (smaller file)",
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


def collect_logits_labels(
    model,
    loader,
    device,
    augmenter,
):
    """Forward only; returns float numpy arrays logits [N, C], labels [N]."""
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                data, labels, _ = batch_data
            else:
                data, labels = batch_data[0], batch_data[1]

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

    logits_np = np.concatenate(logits_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0).astype(np.int64)
    return logits_np, labels_np


def softmax_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / ex.sum(axis=1, keepdims=True)


def multiclass_nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = softmax_np(logits, temperature)
    n = labels.shape[0]
    p_true = probs[np.arange(n), labels]
    p_true = np.clip(p_true, 1e-12, 1.0)
    return float(-np.mean(np.log(p_true)))


def multiclass_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """ECE using confidence = max predicted probability (standard multiclass)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences > lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        prop = float(in_bin.mean())
        if prop > 0.0:
            acc_in_bin = float(accuracies[in_bin].mean())
            conf_in_bin = float(confidences[in_bin].mean())
            ece += abs(conf_in_bin - acc_in_bin) * prop
    return float(ece)


def fit_temperature_grid(
    logits: np.ndarray,
    labels: np.ndarray,
    n_grid: int,
) -> tuple[float, list[dict[str, float]]]:
    """Grid search T in log-space; minimize mean NLL of true class."""
    grid = np.logspace(np.log10(0.15), np.log10(8.0), n_grid)
    curve = []
    best_t = 1.0
    best_nll = 1e18
    for t in grid:
        nll = multiclass_nll(logits, labels, float(t))
        curve.append({"temperature": float(t), "nll": float(nll)})
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t, curve


def pareto_maximize_mask(acc: np.ndarray, f1: np.ndarray) -> np.ndarray:
    """
    True at index i if point i is Pareto-optimal for simultaneous maximization
    of accuracy and macro-F1 (no other point is >= on both and strictly > on one).
    """
    n = int(acc.shape[0])
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if acc[j] >= acc[i] and f1[j] >= f1[i] and (
                acc[j] > acc[i] or f1[j] > f1[i]
            ):
                keep[i] = False
                break
    return keep


def pareto_recommended_from_sweep(rows: list[dict]) -> dict:
    """
    Among Pareto-optimal (acc, macro_f1) rows, pick tau closest (L2) to the ideal
    point (max acc over sweep, max macro_f1 over sweep).
    """
    if not rows:
        return {}
    acc = np.array([float(r["accuracy"]) for r in rows], dtype=np.float64)
    f1 = np.array([float(r["macro_f1"]) for r in rows], dtype=np.float64)
    mask = pareto_maximize_mask(acc, f1)
    ideal_acc = float(acc.max())
    ideal_f1 = float(f1.max())
    best_i = None
    best_d = 1e18
    for i in range(len(rows)):
        if not mask[i]:
            continue
        d = (ideal_acc - acc[i]) ** 2 + (ideal_f1 - f1[i]) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    if best_i is None:
        return {}
    out = dict(rows[best_i])
    out["selection_rule"] = (
        "pareto_front_then_l2_closest_to_ideal(max_acc,max_macro_f1_over_full_sweep)"
    )
    out["ideal_point"] = {"accuracy": ideal_acc, "macro_f1": ideal_f1}
    out["l2_gap_to_ideal"] = float(np.sqrt(best_d))
    return out


def pareto_abstention_report(rows: list[dict]) -> dict:
    """Pareto frontier on (accuracy, macro_f1) plus recommended tau."""
    if not rows:
        return {
            "objectives": ["maximize_accuracy", "maximize_macro_f1"],
            "pareto_optimal_thresholds": [],
            "recommended": {},
        }
    acc = np.array([float(r["accuracy"]) for r in rows], dtype=np.float64)
    f1 = np.array([float(r["macro_f1"]) for r in rows], dtype=np.float64)
    mask = pareto_maximize_mask(acc, f1)
    pareto_rows = [rows[i] for i in range(len(rows)) if mask[i]]
    rec = pareto_recommended_from_sweep(rows)
    return {
        "objectives": [
            "maximize_accuracy (fraction correct; same as test.py)",
            "maximize_macro_f1",
        ],
        "pareto_optimal_thresholds": pareto_rows,
        "recommended": rec,
    }


def abstention_macro_f1_sweep(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    background_class: int,
    n_tau: int = 60,
) -> list[dict[str, float]]:
    """
    If max prob < tau, predict background_class; else argmax. Sweep tau.

    Each row includes test.py-style counts: correct_count, n_total, accuracy,
    plus abstention_rate / n_abstained for coverage vs quality tradeoffs.
    """
    probs = softmax_np(logits, temperature)
    taus = np.linspace(0.25, 1.0, n_tau)
    rows = []
    n_total = int(labels.shape[0])
    for tau in taus:
        max_p = probs.max(axis=1)
        pred = np.argmax(probs, axis=1)
        low = max_p < tau
        pred_adj = pred.copy()
        pred_adj[low] = background_class
        f1m = float(
            f1_score(labels, pred_adj, average="macro", zero_division=0)
        )
        n_correct = int((pred_adj == labels).sum())
        acc = float(n_correct / n_total) if n_total > 0 else 0.0
        n_abstained = int(low.sum())
        abstention_rate = float(n_abstained / n_total) if n_total > 0 else 0.0
        rows.append(
            {
                "tau": float(tau),
                "macro_f1": f1m,
                "accuracy": acc,
                "correct_count": n_correct,
                "n_total": n_total,
                "n_abstained": n_abstained,
                "abstention_rate": abstention_rate,
            }
        )
    return rows


def pareto_recommended_tau_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    background_class: int,
    n_tau: int = 60,
) -> dict:
    """Full sweep + Pareto report + recommended row (used by bootstrap)."""
    rows = abstention_macro_f1_sweep(
        logits, labels, temperature, background_class, n_tau=n_tau
    )
    report = pareto_abstention_report(rows)
    rec = report["recommended"]
    return {"sweep_rows": rows, "pareto": report, "recommended": rec}


def _ci_summary(bootstrap_values: np.ndarray, point_estimate: float) -> dict:
    """95% percentile CI plus bootstrap mean/std."""
    a = np.asarray(bootstrap_values, dtype=np.float64)
    if a.size == 0:
        return {
            "point_estimate": float(point_estimate),
            "ci_low_2.5pct": None,
            "ci_high_97.5pct": None,
            "bootstrap_mean": None,
            "bootstrap_std": None,
        }
    return {
        "point_estimate": float(point_estimate),
        "ci_low_2.5pct": float(np.percentile(a, 2.5)),
        "ci_high_97.5pct": float(np.percentile(a, 97.5)),
        "bootstrap_mean": float(a.mean()),
        "bootstrap_std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
    }


def bootstrap_confidence_intervals(
    logits: np.ndarray,
    labels: np.ndarray,
    T_star: float,
    n_bins_ece: int,
    background_idx: int | None,
    n_replicates: int,
    seed: int,
    point: dict[str, float],
    n_tau_abstain: int = 60,
) -> dict:
    """
    Nonparametric bootstrap: resample (logits, labels) rows with replacement.

    Uses fixed T_star on each replicate. When background_idx is set, the abstention
    tau on each replicate is the **Pareto-recommended** tau (same rule as full-data
    ``pareto_abstention.recommended``).

    ``point`` must hold full-data point estimates keyed like the returned CI entries.
    """
    rng = np.random.RandomState(seed)
    n = int(labels.shape[0])
    acc_u, acc_c = [], []
    mf1_u, mf1_c = [], []
    nll_u, nll_c = [], []
    ece_u, ece_c = [], []
    tau_star_samples: list[float] = []
    abstain_mf1_samples: list[float] = []
    abstain_acc_samples: list[float] = []

    for _ in range(n_replicates):
        idx = rng.randint(0, n, size=n)
        lb = labels[idx]
        zb = logits[idx]
        pu = softmax_np(zb, 1.0)
        pc = softmax_np(zb, T_star)
        pred_u = np.argmax(pu, axis=1)
        pred_c = np.argmax(pc, axis=1)
        acc_u.append(float((pred_u == lb).mean()))
        acc_c.append(float((pred_c == lb).mean()))
        mf1_u.append(
            float(f1_score(lb, pred_u, average="macro", zero_division=0))
        )
        mf1_c.append(
            float(f1_score(lb, pred_c, average="macro", zero_division=0))
        )
        nll_u.append(multiclass_nll(zb, lb, 1.0))
        nll_c.append(multiclass_nll(zb, lb, T_star))
        ece_u.append(multiclass_ece(pu, lb, n_bins_ece))
        ece_c.append(multiclass_ece(pc, lb, n_bins_ece))
        if background_idx is not None:
            pr = pareto_recommended_tau_metrics(
                zb, lb, T_star, background_idx, n_tau=n_tau_abstain
            )
            rec = pr["recommended"]
            if rec:
                tau_star_samples.append(float(rec["tau"]))
                abstain_mf1_samples.append(float(rec["macro_f1"]))
                abstain_acc_samples.append(float(rec["accuracy"]))

    out: dict = {
        "n_replicates": int(n_replicates),
        "seed": int(seed),
        "method": "iid_row_resample_percentile_95",
        "accuracy_uncalibrated_argmax": _ci_summary(
            np.asarray(acc_u, dtype=np.float64),
            point["accuracy_uncalibrated_argmax"],
        ),
        "accuracy_calibrated_argmax": _ci_summary(
            np.asarray(acc_c, dtype=np.float64),
            point["accuracy_calibrated_argmax"],
        ),
        "macro_f1_uncalibrated_argmax": _ci_summary(
            np.asarray(mf1_u, dtype=np.float64),
            point["macro_f1_uncalibrated_argmax"],
        ),
        "macro_f1_calibrated_argmax": _ci_summary(
            np.asarray(mf1_c, dtype=np.float64),
            point["macro_f1_calibrated_argmax"],
        ),
        "nll_mean_true_class_uncalibrated": _ci_summary(
            np.asarray(nll_u, dtype=np.float64),
            point["nll_mean_true_class_uncalibrated"],
        ),
        "nll_mean_true_class_calibrated_T_star": _ci_summary(
            np.asarray(nll_c, dtype=np.float64),
            point["nll_mean_true_class_calibrated_T_star"],
        ),
        "ece_max_conf_uncalibrated": _ci_summary(
            np.asarray(ece_u, dtype=np.float64),
            point["ece_max_conf_uncalibrated"],
        ),
        "ece_max_conf_calibrated_T_star": _ci_summary(
            np.asarray(ece_c, dtype=np.float64),
            point["ece_max_conf_calibrated_T_star"],
        ),
    }
    if background_idx is not None and tau_star_samples:
        out["pareto_recommended_abstention_tau"] = _ci_summary(
            np.asarray(tau_star_samples, dtype=np.float64),
            point["abstention_pareto_recommended_tau"],
        )
        out["accuracy_at_pareto_recommended_tau"] = _ci_summary(
            np.asarray(abstain_acc_samples, dtype=np.float64),
            point["abstention_accuracy_at_recommended_tau"],
        )
        out["macro_f1_at_pareto_recommended_tau"] = _ci_summary(
            np.asarray(abstain_mf1_samples, dtype=np.float64),
            point["abstention_macro_f1_at_recommended_tau"],
        )
    return out


def ovr_per_class_threshold_sweep(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    thresholds: np.ndarray,
) -> list[dict]:
    """Binary detection per class: score = probs[:,c], true = (labels==c)."""
    n_classes = probs.shape[1]
    out = []
    for c in range(n_classes):
        y_true = (labels == c).astype(np.int32)
        scores = probs[:, c]
        if int(y_true.sum()) == 0 or int(y_true.sum()) == len(y_true):
            out.append(
                {
                    "class": class_names[c],
                    "average_precision": None,
                    "best_threshold": None,
                    "best_f1": None,
                    "note": "degenerate single-class labels for OVR",
                }
            )
            continue
        ap = float(average_precision_score(y_true, scores))
        best_f1 = -1.0
        best_t = 0.5
        curve = []
        for t in thresholds:
            y_pred = (scores >= t).astype(np.int32)
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            curve.append(
                {
                    "threshold": round(float(t), 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                }
            )
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        out.append(
            {
                "class": class_names[c],
                "average_precision": round(ap, 4),
                "best_threshold": round(best_t, 4),
                "best_f1": round(float(best_f1), 4),
                "curve": curve,
            }
        )
    return out


def plot_reliability(probs, labels, out_path, title_suffix: str):
    if not _HAS_MPL:
        return False
    n_bins = 15
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    xs = []
    ys = []
    counts = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences > lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        n = int(in_bin.sum())
        counts.append(n)
        if n > 0:
            xs.append(float(confidences[in_bin].mean()))
            ys.append(float(accuracies[in_bin].mean()))
        else:
            xs.append(float(0.5 * (lo + hi)))
            ys.append(0.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="ideal")
    ax.plot(xs, ys, "o-", label="bins")
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.set_title(f"Reliability (max prob) {title_suffix}")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"experiment_dir not found: {experiment_dir}")

    config_path = experiment_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.yaml missing: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
    else:
        checkpoint_path = experiment_dir / "models" / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    config["device"] = str(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = experiment_dir / f"confidence_analysis_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment_config, model_name, loss_source_config = _resolve_experiment_block(config)
    model_config = config["models"][model_name]

    task_name = config["task_name"]
    task_cfg = config[task_name]
    class_names = list(task_cfg["class_names"])

    logging.info("Loading dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    if args.split == "val":
        loader = val_loader
    else:
        loader = test_loader

    skip_normalization = False
    if "type" in loss_source_config and loss_source_config["type"] == "finetune":
        skip_normalization = True
    if not skip_normalization:
        train_loader, val_loader, test_loader = setup_normalization(
            train_loader, val_loader, test_loader, config
        )
        if args.split == "val":
            loader = val_loader
        else:
            loader = test_loader

    augmenter = create_augmenter(
        config,
        augmentation_mode="no",
        experiment_config=experiment_config,
    )

    config["models"][model_name]["pretrain_mode"] = False
    model = create_single_modal_model(config, model_name)
    model = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)

    logging.info("Collecting logits on split=%s ...", args.split)
    logits_np, labels_np = collect_logits_labels(model, loader, device, augmenter)
    n, num_classes = logits_np.shape
    if num_classes != len(class_names):
        logging.warning(
            "num_classes from logits (%d) != len(class_names) (%d); using logits C",
            num_classes,
            len(class_names),
        )

    probs_uncal = softmax_np(logits_np, 1.0)
    nll_uncal = multiclass_nll(logits_np, labels_np, 1.0)
    ece_uncal = multiclass_ece(probs_uncal, labels_np, args.n_bins_ece)
    try:
        ll_sklearn_uncal = float(
            log_loss(labels_np, probs_uncal, labels=list(range(num_classes)))
        )
    except Exception:
        ll_sklearn_uncal = None

    T_star, temp_curve = fit_temperature_grid(
        logits_np, labels_np, args.temperature_grid
    )
    probs_cal = softmax_np(logits_np, T_star)
    nll_cal = multiclass_nll(logits_np, labels_np, T_star)
    ece_cal = multiclass_ece(probs_cal, labels_np, args.n_bins_ece)
    try:
        ll_sklearn_cal = float(
            log_loss(labels_np, probs_cal, labels=list(range(num_classes)))
        )
    except Exception:
        ll_sklearn_cal = None

    pred_uncal = np.argmax(probs_uncal, axis=1)
    pred_cal = np.argmax(probs_cal, axis=1)
    acc_uncal = float((pred_uncal == labels_np).mean())
    acc_cal = float((pred_cal == labels_np).mean())
    macro_f1_uncal = float(
        f1_score(labels_np, pred_uncal, average="macro", zero_division=0)
    )
    macro_f1_cal = float(
        f1_score(labels_np, pred_cal, average="macro", zero_division=0)
    )

    thresholds_ovr = np.arange(0.05, 0.96, 0.05)
    if args.skip_ovr:
        ovr_results = []
    else:
        ovr_results = ovr_per_class_threshold_sweep(
            probs_cal, labels_np, class_names, thresholds_ovr
        )

    abstention_rows = None
    pareto_abstention = None
    abstention_recommended = None
    bg_idx = None
    if "background" in class_names:
        bg_idx = class_names.index("background")
        pa = pareto_recommended_tau_metrics(
            logits_np, labels_np, T_star, bg_idx
        )
        abstention_rows = pa["sweep_rows"]
        pareto_abstention = pa["pareto"]
        abstention_recommended = pa["recommended"]

    point_for_bootstrap = {
        "accuracy_uncalibrated_argmax": acc_uncal,
        "accuracy_calibrated_argmax": acc_cal,
        "macro_f1_uncalibrated_argmax": macro_f1_uncal,
        "macro_f1_calibrated_argmax": macro_f1_cal,
        "nll_mean_true_class_uncalibrated": nll_uncal,
        "nll_mean_true_class_calibrated_T_star": nll_cal,
        "ece_max_conf_uncalibrated": ece_uncal,
        "ece_max_conf_calibrated_T_star": ece_cal,
    }
    if abstention_recommended is not None and abstention_recommended:
        point_for_bootstrap["abstention_pareto_recommended_tau"] = float(
            abstention_recommended["tau"]
        )
        point_for_bootstrap["abstention_macro_f1_at_recommended_tau"] = float(
            abstention_recommended["macro_f1"]
        )
        point_for_bootstrap["abstention_accuracy_at_recommended_tau"] = float(
            abstention_recommended["accuracy"]
        )

    bootstrap_ci = None
    if args.bootstrap_replicates > 0:
        logging.info(
            "Bootstrap percentile CIs (%d replicates, seed=%d)...",
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )
        bootstrap_ci = bootstrap_confidence_intervals(
            logits_np,
            labels_np,
            T_star,
            args.n_bins_ece,
            bg_idx,
            args.bootstrap_replicates,
            args.bootstrap_seed,
            point_for_bootstrap,
        )

    results = {
        "experiment_dir": str(experiment_dir),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "n_samples": int(n),
        "num_classes": int(num_classes),
        "class_names": class_names,
        "temperature_star": T_star,
        "uncalibrated": {
            "nll_mean_true_class": nll_uncal,
            "ece_max_conf": ece_uncal,
            "accuracy_argmax": acc_uncal,
            "macro_f1_argmax": macro_f1_uncal,
            "log_loss_sklearn": ll_sklearn_uncal,
        },
        "calibrated_temperature": {
            "nll_mean_true_class": nll_cal,
            "ece_max_conf": ece_cal,
            "accuracy_argmax": acc_cal,
            "macro_f1_argmax": macro_f1_cal,
            "log_loss_sklearn": ll_sklearn_cal,
        },
        "temperature_search_curve": temp_curve,
        "one_vs_rest_on_calibrated_probs": ovr_results,
        "abstention_threshold_sweep": abstention_rows,
        "pareto_abstention": pareto_abstention,
        "bootstrap_95pct_intervals": bootstrap_ci,
    }

    results_path = out_dir / "confidence_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logging.info("Wrote %s", results_path)
    logging.info(
        "Uncalibrated: NLL=%.4f ECE=%.4f Acc=%.4f",
        nll_uncal,
        ece_uncal,
        acc_uncal,
    )
    logging.info(
        "Calibrated T*=%.4f: NLL=%.4f ECE=%.4f Acc=%.4f",
        T_star,
        nll_cal,
        ece_cal,
        acc_cal,
    )
    if abstention_recommended is not None and abstention_recommended:
        logging.info(
            "Pareto recommended abstention: tau=%.4f acc=%.4f (%d/%d correct) "
            "macro_f1=%.4f abstention_rate=%.4f",
            abstention_recommended["tau"],
            abstention_recommended["accuracy"],
            abstention_recommended["correct_count"],
            abstention_recommended["n_total"],
            abstention_recommended["macro_f1"],
            abstention_recommended["abstention_rate"],
        )
        logging.info(
            "  Pareto-optimal tau count: %d (see pareto_abstention in JSON)",
            len(pareto_abstention["pareto_optimal_thresholds"])
            if pareto_abstention is not None
            else 0,
        )

    if bootstrap_ci is not None:
        def _log_ci(name: str, block: dict):
            logging.info(
                "  %s: point=%.4f  CI[2.5,97.5]=[%.4f, %.4f]",
                name,
                block["point_estimate"],
                block["ci_low_2.5pct"],
                block["ci_high_97.5pct"],
            )

        logging.info("Bootstrap 95th percentile CIs (key metrics):")
        _log_ci("acc (cal, argmax)", bootstrap_ci["accuracy_calibrated_argmax"])
        _log_ci("macro-F1 (cal, argmax)", bootstrap_ci["macro_f1_calibrated_argmax"])
        _log_ci("ECE (cal)", bootstrap_ci["ece_max_conf_calibrated_T_star"])
        _log_ci("NLL true-class (cal)", bootstrap_ci["nll_mean_true_class_calibrated_T_star"])
        if "pareto_recommended_abstention_tau" in bootstrap_ci:
            _log_ci(
                "Pareto recommended abstention tau",
                bootstrap_ci["pareto_recommended_abstention_tau"],
            )
            _log_ci(
                "accuracy at that tau (test.py style)",
                bootstrap_ci["accuracy_at_pareto_recommended_tau"],
            )

    if _HAS_MPL:
        p_before = out_dir / "reliability_uncalibrated.png"
        p_after = out_dir / "reliability_calibrated.png"
        if plot_reliability(probs_uncal, labels_np, p_before, "(uncalibrated)"):
            logging.info("Saved %s", p_before)
        if plot_reliability(probs_cal, labels_np, p_after, f"(T={T_star:.3f})"):
            logging.info("Saved %s", p_after)

    logging.info("Done. Output directory: %s", out_dir)


if __name__ == "__main__":
    main()
