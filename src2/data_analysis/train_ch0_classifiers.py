#!/usr/bin/env python3
"""
Train ch0 continuous-path classifiers on vehicle_classification.

All three models use the same spectral features (no mel):
  centroid + mean energy + PSD (83-dim, n_fft=160 on truncated ch0 stream)

1. Random Forest
2. XGBoost
3. PyTorch MLP (spectral neural network)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from xgboost import XGBClassifier

src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from data_augmenter.acids_continuous import AcidsContinuousTruncator
from data_augmenter.audio_channel_select import select_audio_channel
from data_analysis.analyze_mel_spectral_features import (
    build_combined_matrix,
    compute_spectral_features,
    plot_confusion_matrix,
)
from dataset_utils.MultiModalDataLoader import create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SpectralMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RF, XGB, and spectral MLP on ch0 continuous (no mel)"
    )
    parser.add_argument("--yaml_path", type=str, default="../data/ACIDS.yaml")
    parser.add_argument("--output_dir", type=str, default="../tmp/ch0_classifiers")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mlp_epochs", type=int, default=50)
    parser.add_argument("--mlp_batch_size", type=int, default=256)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--skip_mlp", action="store_true")
    parser.add_argument("--spectral-hann", action="store_true")
    return parser.parse_args()


def load_config(args) -> dict:
    with open(args.yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    exp_name = "classifier_ch0_continuous"
    config["experiment_name"] = exp_name
    config["experiments"][exp_name] = {
        "dataloader": "acids_vehicle",
        "task_name": "vehicle_classification",
    }
    config["task_name"] = "vehicle_classification"
    config["yaml_path"] = str(args.yaml_path)
    config["device"] = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
    config["include_classes"] = None
    return config


def get_params(config: dict) -> dict:
    ds = config
    return {
        "audio_channel_index": 0,
        "n_fft": ds["n_fft"],
        "sample_rate": ds["sample_rate"],
        "num_segments": ds["num_segments"],
        "acids_padded_seg_len": ds["acids_padded_seg_len"],
        "acids_used_samples": ds["acids_used_samples"],
    }


def _label_to_int(label) -> int:
    if isinstance(label, torch.Tensor):
        return int(label.item()) if label.numel() == 1 else int(label[0].item())
    return int(label)


def collect_streams(loader, params: dict, desc: str) -> tuple[np.ndarray, np.ndarray]:
    truncator = AcidsContinuousTruncator(
        num_segments=params["num_segments"],
        padded_seg_len=params["acids_padded_seg_len"],
        used_samples=params["acids_used_samples"],
        target_modalities=["audio"],
    )
    ch_idx = params["audio_channel_index"]
    streams_list = []
    labels_list = []
    for batch in tqdm(loader, desc=desc):
        data, labels, _ = batch
        for i in range(labels.shape[0]):
            sample = {}
            for loc in data:
                sample[loc] = {mod: tensor[i : i + 1] for mod, tensor in data[loc].items()}
            sample = select_audio_channel(sample, ch_idx)
            stream = truncator.truncate(sample)["shake"]["audio"]
            streams_list.append(stream[0, 0].cpu().numpy())
            labels_list.append(_label_to_int(labels[i]))
    return (
        np.stack(streams_list, axis=0).astype(np.float32),
        np.array(labels_list, dtype=np.int64),
    )


def fit_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
) -> tuple[StandardScaler, RandomForestClassifier, XGBClassifier]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(Xs, y_train)
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42,
        n_estimators=200,
        max_depth=6,
        n_jobs=-1,
    )
    xgb.fit(Xs, y_train)
    return scaler, rf, xgb


def eval_sklearn(
    name: str,
    model,
    scaler: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    split: str,
    plots_dir: Path,
) -> dict:
    preds = model.predict(scaler.transform(X))
    cm = confusion_matrix(y, preds, labels=list(range(len(class_names))))
    acc = float(accuracy_score(y, preds))
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    plot_confusion_matrix(
        cm,
        class_names,
        f"{name} — {split} (acc={acc:.3f}, macro-F1={macro_f1:.3f})",
        plots_dir / f"confusion_{split}_{name.lower().replace(' ', '_')}.png",
    )
    return {
        "split": split,
        "model": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
    }


def sklearn_model_size(model, path: Path) -> dict:
    joblib.dump(model, path)
    size_bytes = path.stat().st_size
    info = {"file_bytes": size_bytes, "file_kb": round(size_bytes / 1024, 2)}
    if hasattr(model, "n_estimators"):
        info["n_estimators"] = int(model.n_estimators)
    if hasattr(model, "n_features_in_"):
        info["n_features_in"] = int(model.n_features_in_)
    if hasattr(model, "estimators_"):
        info["n_trees"] = len(model.estimators_)
    return info


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[SpectralMLP, StandardScaler]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    model = SpectralMLP(X_train.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val_acc = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(yb)
        train_metrics = eval_mlp_arrays(model, scaler, X_train, y_train, device)
        val_metrics = eval_mlp_arrays(model, scaler, X_val, y_val, device)
        logger.info(
            "MLP epoch %d/%d loss=%.4f train_acc=%.4f val_acc=%.4f",
            epoch + 1,
            epochs,
            total_loss / len(y_train),
            train_metrics["accuracy"],
            val_metrics["accuracy"],
        )
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler


def eval_mlp_arrays(
    model: SpectralMLP,
    scaler: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,
    device: str,
) -> dict:
    model.eval()
    Xs = torch.from_numpy(scaler.transform(X).astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(Xs)
        preds = logits.argmax(dim=-1).cpu().numpy()
    acc = float(accuracy_score(y, preds))
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    return {"accuracy": acc, "macro_f1": macro_f1, "predictions": preds}


def eval_mlp(
    model: SpectralMLP,
    scaler: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    split: str,
    device: str,
    plots_dir: Path,
) -> dict:
    result = eval_mlp_arrays(model, scaler, X, y, device)
    cm = confusion_matrix(y, result["predictions"], labels=list(range(len(class_names))))
    plot_confusion_matrix(
        cm,
        class_names,
        f"Spectral MLP — {split} (acc={result['accuracy']:.3f}, macro-F1={result['macro_f1']:.3f})",
        plots_dir / f"confusion_{split}_mlp.png",
    )
    return {
        "split": split,
        "model": "spectral_mlp",
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "confusion_matrix": cm.tolist(),
    }


def write_report(
    output_dir: Path,
    class_names: list[str],
    sizes: dict,
    metrics: list[dict],
) -> None:
    lines = [
        "Ch0 Continuous Classifier Comparison (no mel)",
        "=" * 44,
        "",
        "Features (all models):",
        "  spectral centroid + mean energy + PSD (83-dim, n_fft=160)",
        "  ch0 → continuous truncate [:1600] → FFT power spectrum",
        "",
        "Model sizes:",
    ]
    for name, info in sizes.items():
        lines.append(f"  {name}:")
        for k, v in info.items():
            lines.append(f"    {k}: {v}")
    lines.append("")
    lines.append("Metrics (train / val):")
    for m in metrics:
        lines.append(
            f"  {m['model']} {m['split']}: acc={m['accuracy']:.4f} "
            f"macro_f1={m['macro_f1']:.4f}"
        )
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {"class_names": class_names, "sizes": sizes, "metrics": metrics}
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    plot_items = sorted((output_dir / "plots").glob("*.png"))
    imgs = "".join(
        f'<div class="item"><a href="plots/{p.name}">'
        f'<img src="plots/{p.name}"><br>{p.name}</a></div>'
        for p in plot_items
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Ch0 Spectral Classifiers</title>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
pre {{ background: #f6f6f6; padding: 12px; }}
.gallery {{ display: flex; flex-wrap: wrap; gap: 16px; }}
.item img {{ max-width: 480px; border: 1px solid #ccc; }}
</style></head><body>
<h1>Ch0 Spectral Classifiers (no mel)</h1>
<pre>{(output_dir / "report.txt").read_text(encoding="utf-8")}</pre>
<h2>Confusion matrices</h2>
<div class="gallery">{imgs}</div>
</body></html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args)
    params = get_params(config)
    task = config["task_name"]
    class_names = config[task]["class_names"]
    num_classes = config[task]["num_classes"]
    device = config["device"]

    train_loader, val_loader, _ = create_dataloaders(config)

    logger.info("Collecting train streams...")
    train_streams, y_train = collect_streams(train_loader, params, "Train streams")
    logger.info("Collecting val streams...")
    val_streams, y_val = collect_streams(val_loader, params, "Val streams")

    train_feats = compute_spectral_features(
        train_streams,
        sample_rate=params["sample_rate"],
        n_fft=params["n_fft"],
        use_hann=args.spectral_hann,
    )
    val_feats = compute_spectral_features(
        val_streams,
        sample_rate=params["sample_rate"],
        n_fft=params["n_fft"],
        use_hann=args.spectral_hann,
    )
    X_train = build_combined_matrix(train_feats)
    X_val = build_combined_matrix(val_feats)

    logger.info("Training RF and XGB on %d train samples...", len(y_train))
    scaler, rf, xgb = fit_sklearn(X_train, y_train, num_classes)

    sizes = {
        "scaler": {"n_features": int(X_train.shape[1])},
        "random_forest": sklearn_model_size(rf, models_dir / "random_forest.joblib"),
        "xgboost": sklearn_model_size(xgb, models_dir / "xgboost.joblib"),
    }
    joblib.dump(scaler, models_dir / "spectral_scaler.joblib")
    sizes["scaler"]["file_kb"] = round(
        (models_dir / "spectral_scaler.joblib").stat().st_size / 1024, 2
    )

    metrics = []
    metrics.append(eval_sklearn("RF", rf, scaler, X_train, y_train, class_names, "train", plots_dir))
    metrics.append(eval_sklearn("RF", rf, scaler, X_val, y_val, class_names, "val", plots_dir))
    metrics.append(eval_sklearn("XGB", xgb, scaler, X_train, y_train, class_names, "train", plots_dir))
    metrics.append(eval_sklearn("XGB", xgb, scaler, X_val, y_val, class_names, "val", plots_dir))

    if not args.skip_mlp:
        logger.info("Training spectral MLP (%d epochs)...", args.mlp_epochs)
        mlp, mlp_scaler = train_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            num_classes,
            device,
            args.mlp_epochs,
            args.mlp_batch_size,
            args.mlp_lr,
        )
        mlp_path = models_dir / "spectral_mlp.pth"
        torch.save(
            {
                "model_state_dict": mlp.state_dict(),
                "scaler_mean": mlp_scaler.mean_,
                "scaler_scale": mlp_scaler.scale_,
                "in_dim": int(X_train.shape[1]),
                "num_classes": num_classes,
            },
            mlp_path,
        )
        num_params = sum(p.numel() for p in mlp.parameters())
        sizes["spectral_mlp"] = {
            "num_params": num_params,
            "checkpoint_kb": round(mlp_path.stat().st_size / 1024, 2),
            "architecture": "MLP(83 -> 128 -> 128 -> 10)",
        }
        metrics.append(
            eval_mlp(mlp, mlp_scaler, X_train, y_train, class_names, "train", device, plots_dir)
        )
        metrics.append(
            eval_mlp(mlp, mlp_scaler, X_val, y_val, class_names, "val", device, plots_dir)
        )

    write_report(output_dir, class_names, sizes, metrics)
    logger.info("Done. Results in %s", output_dir)


if __name__ == "__main__":
    main()
