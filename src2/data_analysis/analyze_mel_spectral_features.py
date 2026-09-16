"""
Spectral feature extraction and analysis for ACIDS continuous streams.

Features: spectral centroid, mean spectral energy, PSD.
Clustering, PCA, Random Forest, and XGBoost helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _apply_hann(x: np.ndarray, n_fft: int, use_hann: bool) -> np.ndarray:
    if not use_hann:
        return x[..., :n_fft] if x.shape[-1] >= n_fft else np.pad(
            x, [(0, 0)] * (x.ndim - 1) + [(0, n_fft - x.shape[-1])]
        )
    win_len = min(x.shape[-1], n_fft)
    hann = np.hanning(win_len).astype(np.float32)
    out = x[..., :win_len] * hann
    if win_len < n_fft:
        pad_width = [(0, 0)] * (out.ndim - 1) + [(0, n_fft - win_len)]
        out = np.pad(out, pad_width)
    return out


def compute_power_spectrum(
    streams: np.ndarray,
    sample_rate: int,
    n_fft: int,
    use_hann: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        streams: [N, T] float32
    Returns:
        power: [N, n_fft//2 + 1]
        freqs: [n_fft//2 + 1] Hz
    """
    x = _apply_hann(streams.astype(np.float32), n_fft, use_hann)
    fft = np.fft.rfft(x, n=n_fft, axis=-1)
    power = (fft.real ** 2 + fft.imag ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return power, freqs


def compute_spectral_features(
    streams: np.ndarray,
    sample_rate: int = 1600,
    n_fft: int = 160,
    use_hann: bool = False,
) -> dict[str, np.ndarray]:
    """
    Args:
        streams: [N, T] truncated ch0 streams
    Returns:
        dict with centroid [N], mean_energy [N], psd [N, n_freqs]
    """
    power, freqs = compute_power_spectrum(streams, sample_rate, n_fft, use_hann)
    psd = power / (power.sum(axis=-1, keepdims=True) + 1e-12)
    centroid = (power * freqs).sum(axis=-1) / (power.sum(axis=-1) + 1e-12)
    mean_energy = power.mean(axis=-1)
    return {
        "centroid": centroid.astype(np.float32),
        "mean_energy": mean_energy.astype(np.float32),
        "psd": psd.astype(np.float32),
        "power": power.astype(np.float32),
        "freqs": freqs.astype(np.float32),
    }


def build_combined_matrix(features: dict[str, np.ndarray]) -> np.ndarray:
    """Stack centroid, mean_energy, and PSD into [N, 2 + n_freqs]."""
    return np.concatenate(
        [
            features["centroid"].reshape(-1, 1),
            features["mean_energy"].reshape(-1, 1),
            features["psd"],
        ],
        axis=1,
    )


def cluster_single_feature(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    feature_name: str,
) -> dict:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(Xs)
    sil = silhouette_score(Xs, cluster_ids) if len(np.unique(cluster_ids)) > 1 else 0.0
    ari = adjusted_rand_score(y, cluster_ids)
    nmi = normalized_mutual_info_score(y, cluster_ids)
    return {
        "feature": feature_name,
        "silhouette": float(sil),
        "ari": float(ari),
        "nmi": float(nmi),
        "cluster_ids": cluster_ids,
        "confusion": confusion_matrix(y, cluster_ids),
    }


def run_per_feature_clustering(
    features: dict[str, np.ndarray],
    y: np.ndarray,
    n_classes: int,
) -> list[dict]:
    results = []
    results.append(
        cluster_single_feature(
            features["centroid"].reshape(-1, 1), y, n_classes, "centroid"
        )
    )
    results.append(
        cluster_single_feature(
            features["mean_energy"].reshape(-1, 1), y, n_classes, "mean_energy"
        )
    )
    results.append(cluster_single_feature(features["psd"], y, n_classes, "psd"))
    return results


def run_combined_pca(
    features: dict[str, np.ndarray],
    y: np.ndarray,
    class_names: list[str],
    plots_dir: Path,
    n_components: int = 2,
) -> dict:
    X = build_combined_matrix(features)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(n_components, Xs.shape[1], Xs.shape[0]))
    coords = pca.fit_transform(Xs)

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=y, cmap="tab10", alpha=0.7, s=30
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA: centroid + mean energy + PSD (colored by class)")
    plt.colorbar(scatter, ax=ax, label="class index")
    fig.tight_layout()
    fig.savefig(plots_dir / "pca_spectral_combined_2d.png", dpi=120)
    plt.close(fig)

    if Xs.shape[1] >= 3 and Xs.shape[0] >= 3:
        pca3 = PCA(n_components=3)
        coords3 = pca3.fit_transform(Xs)
        fig = plt.figure(figsize=(10, 8))
        ax3 = fig.add_subplot(111, projection="3d")
        ax3.scatter(
            coords3[:, 0], coords3[:, 1], coords3[:, 2], c=y, cmap="tab10", alpha=0.7, s=20
        )
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.set_zlabel("PC3")
        ax3.set_title("PCA 3D: combined spectral features")
        fig.tight_layout()
        fig.savefig(plots_dir / "pca_spectral_combined_3d.png", dpi=120)
        plt.close(fig)

    pca2 = PCA(n_components=2)
    pca2.fit(Xs)
    loadings = pca2.components_.T
    fig, ax = plt.subplots(figsize=(12, 5))
    n_show = min(40, loadings.shape[0])
    ax.bar(range(n_show), loadings[:n_show, 0], alpha=0.7, label="PC1")
    ax.bar(range(n_show), loadings[:n_show, 1], alpha=0.7, label="PC2")
    ax.set_xlabel("Feature index (0=centroid, 1=mean_energy, 2+=PSD bins)")
    ax.set_ylabel("Loading")
    ax.set_title("PCA loadings (first features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "pca_loadings.png", dpi=120)
    plt.close(fig)

    return {
        "coords_2d": coords,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def _aggregate_classification_report(y_true, y_pred, class_names) -> list[dict]:
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    rows = []
    for name in class_names:
        if name in report:
            rows.append(
                {
                    "class_name": name,
                    "precision": report[name]["precision"],
                    "recall": report[name]["recall"],
                    "f1": report[name]["f1-score"],
                    "support": int(report[name]["support"]),
                }
            )
    return rows


def run_classifier_cv(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    model_name: str,
    n_splits: int = 5,
) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min()) if len(counts) else 0
    if min_class < 2:
        rows = [
            {
                "class_name": name,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": int((y == i).sum()),
            }
            for i, name in enumerate(class_names)
        ]
        return {
            "model": model_name,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "y_pred": np.zeros_like(y),
            "class_metrics": rows,
            "confusion": confusion_matrix(y, y),
        }
    n_splits = min(n_splits, min_class)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        )
    elif model_name == "xgb":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=len(class_names),
            eval_metric="mlogloss",
            random_state=42,
            n_estimators=200,
            max_depth=6,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    y_pred = cross_val_predict(clf, Xs, y, cv=skf)
    acc = float((y_pred == y).mean())
    macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
    rows = _aggregate_classification_report(y, y_pred, class_names)
    return {
        "model": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "y_pred": y_pred,
        "class_metrics": rows,
        "confusion": confusion_matrix(y, y_pred),
    }


def compute_mel_separability_metrics(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """kNN CV, Fisher ratio, silhouette on mel feature matrix."""
    classes, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min()) if len(counts) else 0
    if min_class < 2:
        return {
            "knn_cv_accuracy": 0.0,
            "fisher_ratio": 0.0,
            "silhouette": 0.0,
            "mean_pairwise_centroid_dist": 0.0,
        }
    n_splits = min(n_splits, min_class)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    knn = KNeighborsClassifier(n_neighbors=5)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(knn, Xs, y, cv=skf)
    knn_acc = float((y_pred == y).mean())

    classes = np.unique(y)
    global_mean = Xs.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in classes:
        Xc = Xs[y == c]
        if len(Xc) == 0:
            continue
        class_mean = Xc.mean(axis=0)
        between += len(Xc) * np.sum((class_mean - global_mean) ** 2)
        within += np.sum((Xc - class_mean) ** 2)
    fisher = between / (within + 1e-12)

    sil = silhouette_score(Xs, y) if len(classes) > 1 else 0.0

    centroids = []
    for c in classes:
        centroids.append(Xs[y == c].mean(axis=0))
    centroids = np.stack(centroids, axis=0)
    dists = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    mean_pairwise_dist = float(np.mean(dists)) if dists else 0.0

    return {
        "knn_cv_accuracy": knn_acc,
        "fisher_ratio": float(fisher),
        "silhouette": float(sil),
        "mean_pairwise_centroid_dist": mean_pairwise_dist,
    }


def plot_cluster_confusion(
    cm: np.ndarray,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Cluster id")
    ax.set_ylabel("True class index")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_class_precision_bars(rows: list[dict], title: str, path: Path) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["precision"], reverse=True)
    names = [r["class_name"] for r in rows_sorted]
    prec = [r["precision"] for r in rows_sorted]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 5))
    ax.barh(names[::-1], prec[::-1])
    ax.set_xlabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str, path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_psd_by_class(
    features: dict[str, np.ndarray],
    y: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    freqs = features["freqs"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for c_idx, name in enumerate(class_names):
        mask = y == c_idx
        if not np.any(mask):
            continue
        mean_psd = features["psd"][mask].mean(axis=0)
        ax.plot(freqs, mean_psd, label=name)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized power")
    ax.set_title("Mean PSD by class")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_high_precision_classes(
    rf_rows: list[dict],
    xgb_rows: list[dict],
    threshold: float,
    path: Path,
) -> None:
    lines = [f"Precision threshold: {threshold}", ""]
    lines.append("Random Forest:")
    for r in sorted(rf_rows, key=lambda x: x["precision"], reverse=True):
        if r["precision"] >= threshold:
            lines.append(
                f"  {r['class_name']}: precision={r['precision']:.3f} "
                f"recall={r['recall']:.3f} f1={r['f1']:.3f}"
            )
    lines.append("")
    lines.append("XGBoost:")
    for r in sorted(xgb_rows, key=lambda x: x["precision"], reverse=True):
        if r["precision"] >= threshold:
            lines.append(
                f"  {r['class_name']}: precision={r['precision']:.3f} "
                f"recall={r['recall']:.3f} f1={r['f1']:.3f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
