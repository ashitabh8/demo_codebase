#!/usr/bin/env python3
"""
Train Spectral MLP from already-flat float32[1600] samples_txt streams.

Uses Welch-averaged spectral features (n_fft=160, hop=80) over the full second
→ 83-dim → StandardScaler → MLP(83→128→128→10).

Usage:
  python train_from_1600.py --data_root data --output_dir artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))

from load_1600_txt import load_dir
from welch_features import FEATURE_DIM, streams_to_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "background",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


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
    parser = argparse.ArgumentParser(description="Train spectral MLP from 1600-stream data")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Directory with train/*.txt and val/*.txt (samples_txt layout)",
    )
    parser.add_argument(
        "--export_root",
        type=Path,
        default=None,
        help="Directory with train_X.npy/train_y.npy and val_X.npy/val_y.npy "
        "(from export_acids_indices.py)",
    )
    parser.add_argument("--train_dir", type=Path, default=None)
    parser.add_argument("--val_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=PKG / "artifacts")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=10)
    return parser.parse_args()


def load_split_streams(
    export_root: Path | None,
    data_root: Path | None,
    train_dir: Path | None,
    val_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train_streams, y_train, val_streams, y_val."""
    if export_root is not None:
        export_root = Path(export_root)
        train_x = export_root / "train_X.npy"
        train_y = export_root / "train_y.npy"
        val_x = export_root / "val_X.npy"
        val_y = export_root / "val_y.npy"
        for p in (train_x, train_y, val_x, val_y):
            if not p.exists():
                raise FileNotFoundError(f"missing {p}; run export_acids_indices.py first")
        return (
            np.load(train_x).astype(np.float32),
            np.load(train_y).astype(np.int64),
            np.load(val_x).astype(np.float32),
            np.load(val_y).astype(np.int64),
        )

    tdir = train_dir or (Path(data_root) / "train" if data_root else None)
    vdir = val_dir or (Path(data_root) / "val" if data_root else None)
    if tdir is None or vdir is None:
        raise ValueError("Provide --export_root or --data_root / --train_dir+--val_dir")
    _, train_streams, y_train = load_dir(tdir)
    _, val_streams, y_val = load_dir(vdir)
    return train_streams, y_train, val_streams, y_val


def eval_arrays(model, scaler, X, y, device):
    model.eval()
    Xs = torch.from_numpy(scaler.transform(X).astype(np.float32)).to(device)
    with torch.no_grad():
        preds = model(Xs).argmax(dim=-1).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "predictions": preds,
    }


def train_mlp(X_train, y_train, X_val, y_val, num_classes, device, epochs, batch_size, lr):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
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
        train_m = eval_arrays(model, scaler, X_train, y_train, device)
        val_m = eval_arrays(model, scaler, X_val, y_val, device)
        logger.info(
            "epoch %d/%d loss=%.4f train_acc=%.4f val_acc=%.4f",
            epoch + 1,
            epochs,
            total_loss / max(len(y_train), 1),
            train_m["accuracy"],
            val_m["accuracy"],
        )
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler


def main():
    args = parse_args()
    device = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_streams, y_train, val_streams, y_val = load_split_streams(
        args.export_root, args.data_root, args.train_dir, args.val_dir
    )
    if train_streams.ndim != 2 or train_streams.shape[1] != 1600:
        raise ValueError(f"train streams expected (N,1600), got {train_streams.shape}")
    if val_streams.ndim != 2 or val_streams.shape[1] != 1600:
        raise ValueError(f"val streams expected (N,1600), got {val_streams.shape}")
    logger.info("Loaded train=%d val=%d streams", len(y_train), len(y_val))

    X_train = streams_to_features(train_streams)
    X_val = streams_to_features(val_streams)
    if X_train.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"expected {FEATURE_DIM} features, got {X_train.shape[1]}")

    model, scaler = train_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        args.num_classes,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
    )

    train_m = eval_arrays(model, scaler, X_train, y_train, device)
    val_m = eval_arrays(model, scaler, X_val, y_val, device)

    ckpt_path = args.output_dir / "spectral_mlp.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
            "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
            "in_dim": int(X_train.shape[1]),
            "num_classes": int(args.num_classes),
            "frontend": "welch",
            "n_fft": 160,
            "hop": 80,
            "sample_rate": 1600,
            "class_names": CLASS_NAMES,
        },
        ckpt_path,
    )
    logger.info("Wrote %s", ckpt_path)

    report = {
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "train_accuracy": train_m["accuracy"],
        "train_macro_f1": train_m["macro_f1"],
        "val_accuracy": val_m["accuracy"],
        "val_macro_f1": val_m["macro_f1"],
        "checkpoint": str(ckpt_path),
        "frontend": "welch n_fft=160 hop=80",
        "feature_dim": FEATURE_DIM,
    }
    (args.output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Done. train_acc=%.4f val_acc=%.4f",
        train_m["accuracy"],
        val_m["accuracy"],
    )


if __name__ == "__main__":
    main()
