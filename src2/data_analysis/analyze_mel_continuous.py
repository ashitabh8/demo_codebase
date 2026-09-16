#!/usr/bin/env python3
"""
Mel hyperparameter sweep and spectral feature analysis for ACIDS continuous path.

Loads vehicle_classification val data, applies ch0 + continuous truncation,
sweeps mel settings, and runs spectral clustering / PCA / RF / XGBoost.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from data_augmenter.acids_continuous import AcidsContinuousTruncator
from data_augmenter.audio_channel_select import select_audio_channel
from data_augmenter.mel_preprocess import MelPreprocessor
from dataset_utils.MultiModalDataLoader import create_dataloaders
from data_analysis.analyze_mel_spectral_features import (
    build_combined_matrix,
    compute_mel_separability_metrics,
    compute_spectral_features,
    plot_class_precision_bars,
    plot_cluster_confusion,
    plot_confusion_matrix,
    plot_psd_by_class,
    run_classifier_cv,
    run_combined_pca,
    run_per_feature_clustering,
    write_high_precision_classes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TEMPORAL_CONFIGS = [
    {"rows": 7, "hop": 228, "window": 228, "name": "7x228"},
    {"rows": 10, "hop": 160, "window": 160, "name": "10x160"},
    {"rows": 8, "hop": 200, "window": 200, "name": "8x200"},
    {"rows": 16, "hop": 100, "window": 100, "name": "16x100"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mel hyperparameter sweep and spectral analysis (ACIDS continuous)"
    )
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="vehicle_classification")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_samples_per_class", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--include_classes", type=int, nargs="+", default=None)
    parser.add_argument(
        "--sweep",
        dest="sweep",
        action="store_true",
        help="Run mel hyperparameter sweep (default)",
    )
    parser.add_argument(
        "--no-sweep",
        dest="sweep",
        action="store_false",
        help="Skip mel sweep; use baseline mel params only",
    )
    parser.set_defaults(sweep=True)
    parser.add_argument(
        "--skip-spectral",
        action="store_true",
        help="Skip spectral clustering / PCA / RF / XGBoost",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Alias for --no-sweep",
    )
    parser.add_argument("--precision_threshold", type=float, default=0.8)
    parser.add_argument(
        "--spectral-hann",
        action="store_true",
        help="Apply Hann window before spectral FFT (n_fft samples)",
    )
    return parser.parse_args()


def load_config(args) -> dict:
    yaml_path = Path(args.yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["experiment_name"] = args.experiment_name
    config["yaml_path"] = str(yaml_path)
    config["device"] = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"

    exp = config["experiments"][args.experiment_name]
    if "task_name" not in exp:
        config["task_name"] = args.task_name
    else:
        config["task_name"] = exp["task_name"]

    if "dataloader" not in exp:
        exp["dataloader"] = "acids_vehicle"

    include = args.include_classes
    if include is None and "include_classes" in exp:
        include = exp["include_classes"]
    config["include_classes"] = include

    return config


def get_experiment_params(config: dict) -> dict:
    exp = config["experiments"][config["experiment_name"]]
    ds = config
    params = {
        "audio_channel_index": exp["audio_channel_index"] if "audio_channel_index" in exp else 0,
        "n_fft": exp["n_fft"] if "n_fft" in exp else ds["n_fft"],
        "mel_bins": ds["mel_bins"],
        "mel_fmin": ds["mel_fmin"],
        "mel_fmax": ds["mel_fmax"],
        "sample_rate": ds["sample_rate"],
        "mel_window_fn": ds["mel_window_fn"] if "mel_window_fn" in ds else "none",
        "acids_used_samples": ds["acids_used_samples"],
        "acids_padded_seg_len": ds["acids_padded_seg_len"],
        "num_segments": ds["num_segments"],
        "mel_window": ds["acids_mel_window"],
        "mel_hop": ds["acids_mel_hop"],
        "mel_rows": ds["acids_mel_rows"],
    }
    return params


def _label_to_int(label) -> int:
    if isinstance(label, torch.Tensor):
        if label.numel() == 1:
            return int(label.item())
        return int(label[0].item())
    return int(label)


def collect_val_streams(
    config: dict,
    split: str,
    max_samples_per_class: int,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    _, val_loader, test_loader = create_dataloaders(config)
    if split == "train":
        raise ValueError("Use val or test split for analysis; train not exposed here")
    loader = val_loader if split == "val" else test_loader

    task = config[config["task_name"]]
    class_names = task["class_names"]
    include = config["include_classes"]

    truncator = AcidsContinuousTruncator(
        num_segments=params["num_segments"],
        padded_seg_len=params["acids_padded_seg_len"],
        used_samples=params["acids_used_samples"],
        target_modalities=["audio"],
    )

    per_class: dict[int, list[np.ndarray]] = {i: [] for i in range(len(class_names))}
    ch_idx = params["audio_channel_index"]

    for batch in loader:
        data, labels, _ = batch
        for i in range(labels.shape[0]):
            y = _label_to_int(labels[i])
            if include is not None and y not in include:
                continue
            if len(per_class[y]) >= max_samples_per_class:
                continue
            sample_data = {}
            for loc in data:
                sample_data[loc] = {}
                for mod, tensor in data[loc].items():
                    sample_data[loc][mod] = tensor[i : i + 1]
            sample_data = select_audio_channel(sample_data, ch_idx)
            stream = truncator.truncate(sample_data)["shake"]["audio"]
            per_class[y].append(stream[0, 0].cpu().numpy())

        if all(len(per_class[c]) >= max_samples_per_class for c in per_class):
            if include is None or all(
                len(per_class[c]) >= max_samples_per_class for c in include
            ):
                break

    streams_list = []
    labels_list = []
    active_names = []
    if include is not None:
        remap = {old: new for new, old in enumerate(sorted(include))}
        for old_idx in sorted(include):
            for arr in per_class[old_idx]:
                streams_list.append(arr)
                labels_list.append(remap[old_idx])
            active_names.append(class_names[old_idx])
    else:
        active_names = class_names
        for c_idx in range(len(class_names)):
            for arr in per_class[c_idx]:
                streams_list.append(arr)
                labels_list.append(c_idx)

    if not streams_list:
        raise RuntimeError("No samples collected; check index files and include_classes")

    streams = np.stack(streams_list, axis=0).astype(np.float32)
    y = np.array(labels_list, dtype=np.int64)
    logger.info(
        "Collected %d samples (%d classes) from %s split",
        len(y),
        len(active_names),
        split,
    )
    return streams, y, active_names


def extract_mel_features(
    streams: np.ndarray,
    params: dict,
    mel_fmax: float,
    mel_bins: int,
    window_fn: str,
    temporal: dict,
    device: str = "cpu",
) -> np.ndarray:
    mel = MelPreprocessor(
        n_fft=params["n_fft"],
        n_mel=mel_bins,
        fmin=params["mel_fmin"],
        fmax=mel_fmax,
        sample_rate=params["sample_rate"],
        device=device,
        mel_window=temporal["window"],
        mel_hop=temporal["hop"],
        mel_rows=temporal["rows"],
        window_fn=window_fn,
    )
    batch_size = 32
    feats = []
    for start in range(0, len(streams), batch_size):
        chunk = streams[start : start + batch_size]
        t = torch.from_numpy(chunk).unsqueeze(1).to(device)
        out = mel.preprocess_continuous({"shake": {"audio": t}})
        flat = out["shake"]["audio"].reshape(len(chunk), -1)
        feats.append(flat.cpu().numpy())
    return np.concatenate(feats, axis=0)


def validate_temporal(temporal: dict, used_samples: int, n_fft: int) -> bool:
    del n_fft  # window may exceed n_fft; mel truncates/pads inside FFT
    return temporal["rows"] * temporal["hop"] <= used_samples


def build_sweep_grid(params: dict) -> list[dict]:
    mel_fmax_vals = list(range(350, 801, 50))
    mel_bins_vals = list(range(40, 81, 8))
    window_fns = ["none", "hann"]
    grid = []
    for fmax, bins, wf, temporal in product(
        mel_fmax_vals, mel_bins_vals, window_fns, TEMPORAL_CONFIGS
    ):
        if not validate_temporal(temporal, params["acids_used_samples"], params["n_fft"]):
            continue
        grid.append(
            {
                "mel_fmax": fmax,
                "mel_bins": bins,
                "mel_window_fn": wf,
                "temporal_name": temporal["name"],
                "mel_rows": temporal["rows"],
                "mel_hop": temporal["hop"],
                "mel_window": temporal["window"],
                "temporal": temporal,
            }
        )
    return grid


def run_sweep(
    streams: np.ndarray,
    y: np.ndarray,
    params: dict,
    output_dir: Path,
    device: str,
) -> list[dict]:
    grid = build_sweep_grid(params)
    results = []
    for cfg in tqdm(grid, desc="Mel sweep"):
        try:
            X = extract_mel_features(
                streams,
                params,
                mel_fmax=cfg["mel_fmax"],
                mel_bins=cfg["mel_bins"],
                window_fn=cfg["mel_window_fn"],
                temporal=cfg["temporal"],
                device=device,
            )
            metrics = compute_mel_separability_metrics(X, y)
            row = {**cfg, **metrics}
            del row["temporal"]
            results.append(row)
        except Exception as exc:
            logger.warning("Skipping config %s: %s", cfg, exc)

    results.sort(key=lambda r: r["knn_cv_accuracy"], reverse=True)
    csv_path = output_dir / "sweep_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
    return results


def plot_stream_truncation(streams: np.ndarray, y: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    axes = axes.flatten()
    for ax, cls in zip(axes, np.unique(y)[:6]):
        idx = np.where(y == cls)[0][0]
        ax.plot(streams[idx])
        ax.set_title(f"class {cls}")
        ax.set_xlabel("sample")
    fig.suptitle("Truncated ch0 streams (1600 samples)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sweep_heatmap(results: list[dict], path: Path) -> None:
    if not results:
        return
    baseline = [r for r in results if r["temporal_name"] == "7x228" and r["mel_window_fn"] == "none"]
    if not baseline:
        baseline = results
    fmax_vals = sorted({r["mel_fmax"] for r in baseline})
    bins_vals = sorted({r["mel_bins"] for r in baseline})
    mat = np.zeros((len(bins_vals), len(fmax_vals)))
    for i, b in enumerate(bins_vals):
        for j, f in enumerate(fmax_vals):
            matches = [
                r["knn_cv_accuracy"]
                for r in baseline
                if r["mel_bins"] == b and r["mel_fmax"] == f
            ]
            mat[i, j] = np.mean(matches) if matches else 0.0
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(fmax_vals)))
    ax.set_xticklabels(fmax_vals)
    ax.set_yticks(range(len(bins_vals)))
    ax.set_yticklabels(bins_vals)
    ax.set_xlabel("mel_fmax")
    ax.set_ylabel("mel_bins")
    ax.set_title("kNN CV accuracy (7x228, no Hann)")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sweep_line(
    results: list[dict],
    group_key: str,
    title: str,
    path: Path,
) -> None:
    if not results:
        return
    groups: dict = {}
    for r in results:
        if r["temporal_name"] != "7x228" or r["mel_window_fn"] != "none":
            continue
        g = r[group_key]
        groups.setdefault(g, []).append(r["knn_cv_accuracy"])
    if not groups:
        return
    xs = sorted(groups.keys())
    ys = [np.mean(groups[x]) for x in xs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(group_key)
    ax.set_ylabel("mean kNN CV accuracy")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_hann_compare(results: list[dict], path: Path) -> None:
    if not results:
        return
    acc = {"none": [], "hann": []}
    for r in results:
        if r["temporal_name"] != "7x228":
            continue
        acc[r["mel_window_fn"]].append(r["knn_cv_accuracy"])
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = []
    vals = []
    for k in ("none", "hann"):
        if acc[k]:
            labels.append(k)
            vals.append(np.mean(acc[k]))
    ax.bar(labels, vals)
    ax.set_ylabel("mean kNN CV accuracy")
    ax.set_title("Hann vs none (7x228 temporal)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_temporal_compare(results: list[dict], path: Path) -> None:
    if not results:
        return
    groups: dict = {}
    for r in results:
        if r["mel_window_fn"] != "none":
            continue
        groups.setdefault(r["temporal_name"], []).append(r["knn_cv_accuracy"])
    names = sorted(groups.keys())
    vals = [np.mean(groups[n]) for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, vals)
    ax.set_ylabel("mean kNN CV accuracy")
    ax.set_title("Temporal tiling (no Hann)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_class_mean_mel(
    streams: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    params: dict,
    cfg: dict,
    title: str,
    path: Path,
    device: str,
) -> None:
    X = extract_mel_features(
        streams,
        params,
        mel_fmax=cfg["mel_fmax"],
        mel_bins=cfg["mel_bins"],
        window_fn=cfg["mel_window_fn"],
        temporal={
            "rows": cfg["mel_rows"],
            "hop": cfg["mel_hop"],
            "window": cfg["mel_window"],
        },
        device=device,
    )
    mel_rows = cfg["mel_rows"]
    mel_bins = cfg["mel_bins"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for c_idx, name in enumerate(class_names):
        mask = y == c_idx
        if not np.any(mask):
            continue
        mean_mel = X[mask].mean(axis=0).reshape(mel_rows, mel_bins)
        ax.plot(mean_mel.mean(axis=0), label=name, alpha=0.8)
    ax.set_xlabel("mel bin")
    ax.set_ylabel("mean log-mel (avg over rows)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sample_mels_grid(
    streams: np.ndarray,
    y: np.ndarray,
    params: dict,
    cfg: dict,
    path: Path,
    device: str,
    n_per_class: int = 1,
) -> None:
    mel = MelPreprocessor(
        n_fft=params["n_fft"],
        n_mel=cfg["mel_bins"],
        fmin=params["mel_fmin"],
        fmax=cfg["mel_fmax"],
        sample_rate=params["sample_rate"],
        device=device,
        mel_window=cfg["mel_window"],
        mel_hop=cfg["mel_hop"],
        mel_rows=cfg["mel_rows"],
        window_fn=cfg["mel_window_fn"],
    )
    classes = np.unique(y)
    n_show = min(6, len(classes))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for ax, cls in zip(axes, classes[:n_show]):
        idx = np.where(y == cls)[0][0]
        t = torch.from_numpy(streams[idx : idx + 1]).unsqueeze(1).to(device)
        m = mel.preprocess_continuous({"shake": {"audio": t}})["shake"]["audio"][0, 0]
        ax.imshow(m.cpu().numpy(), aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"class {cls}")
    fig.suptitle("Sample mel spectrograms")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pca_mel(
    streams: np.ndarray,
    y: np.ndarray,
    params: dict,
    cfg: dict,
    path: Path,
    device: str,
) -> None:
    X = extract_mel_features(
        streams,
        params,
        mel_fmax=cfg["mel_fmax"],
        mel_bins=cfg["mel_bins"],
        window_fn=cfg["mel_window_fn"],
        temporal={
            "rows": cfg["mel_rows"],
            "hop": cfg["mel_hop"],
            "window": cfg["mel_window"],
        },
        device=device,
    )
    Xs = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="tab10", alpha=0.7, s=25)
    ax.set_title("PCA on mel features (best config)")
    plt.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_band_energy_by_class(
    streams: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    n_fft: int,
    sample_rate: int,
    path: Path,
) -> None:
    from data_analysis.analyze_mel_spectral_features import compute_power_spectrum

    power, freqs = compute_power_spectrum(streams, sample_rate, n_fft, use_hann=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    for c_idx, name in enumerate(class_names):
        mask = y == c_idx
        if not np.any(mask):
            continue
        ax.plot(freqs, power[mask].mean(axis=0), label=name)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean power")
    ax.set_title("Band energy by class (first n_fft samples)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_spectral_analysis(
    streams: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    params: dict,
    output_dir: Path,
    use_hann: bool,
    precision_threshold: float,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    features = compute_spectral_features(
        streams,
        sample_rate=params["sample_rate"],
        n_fft=params["n_fft"],
        use_hann=use_hann,
    )
    plot_psd_by_class(features, y, class_names, plots_dir / "psd_mean_by_class.png")

    cluster_results = run_per_feature_clustering(features, y, len(class_names))
    metrics_path = output_dir / "spectral_cluster_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["feature", "silhouette", "ari", "nmi"]
        )
        writer.writeheader()
        for r in cluster_results:
            writer.writerow(
                {
                    "feature": r["feature"],
                    "silhouette": r["silhouette"],
                    "ari": r["ari"],
                    "nmi": r["nmi"],
                }
            )
            plot_cluster_confusion(
                r["confusion"],
                f"KMeans vs labels ({r['feature']})",
                plots_dir / f"cluster_{r['feature']}.png",
            )

    run_combined_pca(features, y, class_names, plots_dir)

    X = build_combined_matrix(features)
    rf = run_classifier_cv(X, y, class_names, "rf")
    xgb = run_classifier_cv(X, y, class_names, "xgb")

    for name, result in [("rf", rf), ("xgb", xgb)]:
        csv_out = output_dir / f"spectral_class_precision_{name}.csv"
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["class_name", "precision", "recall", "f1", "support"],
            )
            writer.writeheader()
            writer.writerows(result["class_metrics"])
        plot_class_precision_bars(
            result["class_metrics"],
            f"Per-class precision ({name.upper()})",
            plots_dir / f"class_precision_{name}.png",
        )
        plot_confusion_matrix(
            result["confusion"],
            class_names,
            f"Confusion matrix ({name.upper()})",
            plots_dir / f"confusion_matrix_{name}.png",
        )

    write_high_precision_classes(
        rf["class_metrics"],
        xgb["class_metrics"],
        precision_threshold,
        output_dir / "spectral_high_precision_classes.txt",
    )

    return cluster_results, rf, xgb


def cfg_from_params(params: dict) -> dict:
    return {
        "mel_fmax": params["mel_fmax"],
        "mel_bins": params["mel_bins"],
        "mel_window_fn": params["mel_window_fn"],
        "mel_rows": params["mel_rows"],
        "mel_hop": params["mel_hop"],
        "mel_window": params["mel_window"],
    }


def cfg_from_sweep_row(row: dict) -> dict:
    return {
        "mel_fmax": row["mel_fmax"],
        "mel_bins": row["mel_bins"],
        "mel_window_fn": row["mel_window_fn"],
        "mel_rows": row["mel_rows"],
        "mel_hop": row["mel_hop"],
        "mel_window": row["mel_window"],
    }


def write_summary(
    output_dir: Path,
    sweep_results: list[dict],
    baseline_cfg: dict,
    rf_result,
    xgb_result,
    cluster_results,
) -> None:
    lines = ["Mel Hyperparameter Analysis Summary", "=" * 40, ""]
    if sweep_results:
        best = sweep_results[0]
        lines.append("Top mel config (by kNN CV accuracy):")
        lines.append(
            f"  mel_fmax={best['mel_fmax']} mel_bins={best['mel_bins']} "
            f"window_fn={best['mel_window_fn']} temporal={best['temporal_name']}"
        )
        lines.append(f"  knn_cv_accuracy={best['knn_cv_accuracy']:.4f}")
        lines.append(f"  fisher_ratio={best['fisher_ratio']:.4f}")
        lines.append("")
        lines.append("Top 5 mel configs:")
        for r in sweep_results[:5]:
            lines.append(
                f"  fmax={r['mel_fmax']} bins={r['mel_bins']} "
                f"{r['mel_window_fn']} {r['temporal_name']} "
                f"kNN={r['knn_cv_accuracy']:.4f}"
            )
    else:
        lines.append("Mel sweep skipped or empty.")
    lines.append("")
    lines.append(f"Baseline mel: {baseline_cfg}")
    if rf_result is not None:
        lines.append("")
        lines.append(
            f"Spectral RF: acc={rf_result['accuracy']:.4f} "
            f"macro_f1={rf_result['macro_f1']:.4f}"
        )
        lines.append(
            f"Spectral XGB: acc={xgb_result['accuracy']:.4f} "
            f"macro_f1={xgb_result['macro_f1']:.4f}"
        )
        best_rf = max(rf_result["class_metrics"], key=lambda x: x["precision"])
        worst_rf = min(rf_result["class_metrics"], key=lambda x: x["precision"])
        lines.append(
            f"Best RF precision class: {best_rf['class_name']} "
            f"({best_rf['precision']:.3f})"
        )
        lines.append(
            f"Worst RF precision class: {worst_rf['class_name']} "
            f"({worst_rf['precision']:.3f})"
        )
    if cluster_results:
        lines.append("")
        lines.append("Clustering (best ARI feature):")
        best_c = max(cluster_results, key=lambda x: x["ari"])
        lines.append(
            f"  {best_c['feature']}: ARI={best_c['ari']:.4f} "
            f"NMI={best_c['nmi']:.4f} silhouette={best_c['silhouette']:.4f}"
        )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index_html(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"

    report_files = sorted(output_dir.glob("*.csv")) + sorted(
        p for p in output_dir.glob("*.txt") if p.name != "summary.txt"
    )
    report_links = "".join(
        f'<li><a href="{p.relative_to(output_dir).as_posix()}">{p.name}</a></li>'
        for p in report_files
    )

    summary_html = ""
    summary_path = output_dir / "summary.txt"
    if summary_path.exists():
        summary_text = summary_path.read_text(encoding="utf-8").strip()
        summary_html = (
            f'<pre class="summary">{summary_text}</pre>'
        )

    plot_sections = [
        (
            "Classification (RF / XGBoost — full val sample)",
            [
                "confusion_matrix_rf.png",
                "confusion_matrix_xgb.png",
                "class_precision_rf.png",
                "class_precision_xgb.png",
            ],
            "wide",
        ),
        (
            "Spectral features",
            [
                "cluster_centroid.png",
                "cluster_mean_energy.png",
                "cluster_psd.png",
                "pca_spectral_combined_2d.png",
                "pca_spectral_combined_3d.png",
                "psd_mean_by_class.png",
                "band_energy_by_class.png",
                "pca_loadings.png",
            ],
            "normal",
        ),
        (
            "Mel sweep",
            [
                "sweep_heatmap_knn.png",
                "sweep_mel_fmax.png",
                "sweep_mel_bins.png",
                "sweep_hann_compare.png",
                "sweep_temporal.png",
            ],
            "normal",
        ),
        (
            "Mel spectrograms",
            [
                "class_mean_mel_baseline.png",
                "class_mean_mel_best.png",
                "sample_mels_grid.png",
                "pca_mel_best.png",
                "stream_truncation.png",
            ],
            "normal",
        ),
    ]

    def plot_item(name: str, size: str) -> str:
        path = plots_dir / name
        if not path.exists():
            return ""
        rel = f"plots/{name}"
        css = "item wide" if size == "wide" else "item"
        return (
            f'<div class="{css}"><a href="{rel}">'
            f'<img src="{rel}" alt="{name}"><br>{name}</a></div>'
        )

    sections_html = []
    for title, names, size in plot_sections:
        items = "".join(plot_item(n, size) for n in names)
        if items:
            sections_html.append(
                f"<h2>{title}</h2><div class=\"gallery\">{items}</div>"
            )

    listed = {n for _, names, _ in plot_sections for n in names}
    extra = sorted(
        p.name for p in plots_dir.glob("*.png") if p.name not in listed
    )
    if extra:
        items = "".join(plot_item(n, "normal") for n in extra)
        sections_html.append(f"<h2>Other plots</h2><div class=\"gallery\">{items}</div>")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Mel Analysis</title>
<style>
body {{ font-family: sans-serif; margin: 20px; max-width: 1400px; }}
.summary {{ background: #f6f6f6; padding: 12px 16px; border-radius: 6px;
            overflow-x: auto; font-size: 13px; line-height: 1.4; }}
.gallery {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 28px; }}
.item {{ text-align: center; font-size: 12px; }}
.item img {{ max-width: 320px; border: 1px solid #ccc; border-radius: 4px; }}
.item.wide img {{ max-width: 520px; }}
.item a {{ color: inherit; text-decoration: none; }}
.item a:hover img {{ outline: 2px solid #4a90d9; }}
h2 {{ margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
</style></head><body>
<h1>Mel Hyperparameter Analysis</h1>
<p>Output directory: <code>{output_dir.name}</code></p>
<h2>Summary</h2>
{summary_html}
<h2>Reports</h2>
<ul>{report_links}
<li><a href="summary.txt">summary.txt</a></li></ul>
{''.join(sections_html)}
</body></html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    if args.skip_sweep:
        args.sweep = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    params = get_experiment_params(config)
    device = config["device"]

    streams, y, class_names = collect_val_streams(
        config,
        args.split,
        args.max_samples_per_class,
        params,
    )

    plot_stream_truncation(streams, y, plots_dir / "stream_truncation.png")

    baseline_cfg = cfg_from_params(params)
    sweep_results: list[dict] = []
    rf_result = None
    xgb_result = None
    cluster_results = None

    if not args.skip_spectral:
        cluster_results, rf_result, xgb_result = run_spectral_analysis(
            streams,
            y,
            class_names,
            params,
            output_dir,
            args.spectral_hann,
            args.precision_threshold,
        )

    if args.sweep:
        sweep_results = run_sweep(streams, y, params, output_dir, device)
        plot_sweep_heatmap(sweep_results, plots_dir / "sweep_heatmap_knn.png")
        plot_sweep_line(
            sweep_results,
            "mel_fmax",
            "kNN vs mel_fmax",
            plots_dir / "sweep_mel_fmax.png",
        )
        plot_sweep_line(
            sweep_results,
            "mel_bins",
            "kNN vs mel_bins",
            plots_dir / "sweep_mel_bins.png",
        )
        plot_hann_compare(sweep_results, plots_dir / "sweep_hann_compare.png")
        plot_temporal_compare(sweep_results, plots_dir / "sweep_temporal.png")
        best_cfg = cfg_from_sweep_row(sweep_results[0]) if sweep_results else baseline_cfg
    else:
        best_cfg = baseline_cfg

    plot_class_mean_mel(
        streams,
        y,
        class_names,
        params,
        baseline_cfg,
        "Class mean mel (baseline)",
        plots_dir / "class_mean_mel_baseline.png",
        device,
    )
    plot_class_mean_mel(
        streams,
        y,
        class_names,
        params,
        best_cfg,
        "Class mean mel (best sweep)" if sweep_results else "Class mean mel (baseline)",
        plots_dir / "class_mean_mel_best.png",
        device,
    )
    plot_sample_mels_grid(
        streams, y, params, best_cfg, plots_dir / "sample_mels_grid.png", device
    )
    plot_pca_mel(streams, y, params, best_cfg, plots_dir / "pca_mel_best.png", device)
    plot_band_energy_by_class(
        streams,
        y,
        class_names,
        params["n_fft"],
        params["sample_rate"],
        plots_dir / "band_energy_by_class.png",
    )

    write_summary(
        output_dir,
        sweep_results,
        baseline_cfg,
        rf_result,
        xgb_result,
        cluster_results,
    )
    write_index_html(output_dir)
    logger.info("Analysis complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
