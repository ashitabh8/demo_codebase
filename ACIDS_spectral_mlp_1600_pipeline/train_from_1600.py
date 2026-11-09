#!/usr/bin/env python3
"""
Train Spectral MLP from already-flat float32[1600] samples_txt streams.

Uses Welch-averaged spectral features (n_fft=160, hop=80) over the full second
→ 83-dim → StandardScaler → MLP(83→128→128→C).

C (num_classes) defaults to max(label)+1 over train+val. Override with
--num_classes / --class_names when your dataset needs a fixed head size.

Usage:
  python train_from_1600.py --data_root data --output_dir artifacts
  python train_from_1600.py --export_root exported_acids --num_classes 15
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

# Default ACIDS vehicle_classification names (used when num_classes == 10 and
# no --class_names file is given).
DEFAULT_CLASS_NAMES = [
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
# Backward-compatible alias for scripts that import CLASS_NAMES.
CLASS_NAMES = DEFAULT_CLASS_NAMES


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


def load_class_names_file(path: Path) -> list[str]:
    """Load one class name per non-empty line from a text file."""
    names = [
        ln.strip()
        for ln in Path(path).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not names:
        raise ValueError(f"class names file is empty: {path}")
    return names


def resolve_num_classes(
    y_train: np.ndarray,
    y_val: np.ndarray,
    num_classes: int | None = None,
) -> int:
    """
    Resolve head size from labels.

    Default: max(train, val) + 1.
    Override: --num_classes N requires every label in 0 .. N-1.
    """
    y = np.concatenate([np.asarray(y_train, dtype=np.int64), np.asarray(y_val, dtype=np.int64)])
    if y.size == 0:
        raise ValueError("cannot resolve num_classes: no train/val labels")
    if int(y.min()) < 0:
        raise ValueError(f"labels must be >= 0, got min={int(y.min())}")

    inferred = int(y.max()) + 1
    if num_classes is None:
        n = inferred
        logger.info("num_classes=%d (from max label + 1)", n)
    else:
        n = int(num_classes)
        if n < 1:
            raise ValueError(f"--num_classes must be >= 1, got {n}")
        if inferred > n:
            raise ValueError(
                f"labels require at least {inferred} classes "
                f"(max label={int(y.max())}), but --num_classes={n}"
            )
        logger.info("num_classes=%d (forced via --num_classes; data max+1=%d)", n, inferred)

    present = set(int(v) for v in np.unique(y).tolist())
    missing = [i for i in range(n) if i not in present]
    if missing:
        logger.warning(
            "labels 0..%d missing from train+val (head still has %d logits): %s",
            n - 1,
            n,
            missing,
        )
    return n


def resolve_class_names(
    num_classes: int,
    class_names_path: Path | None = None,
) -> list[str]:
    """
    Resolve display names for class ids 0 .. num_classes-1.

    --class_names file (one name per line) wins when given.
    Else if num_classes == 10, use DEFAULT_CLASS_NAMES.
    Else name classes "0" .. "N-1".
    """
    if class_names_path is not None:
        names = load_class_names_file(class_names_path)
        if len(names) != num_classes:
            raise ValueError(
                f"class names file has {len(names)} names, expected {num_classes}: "
                f"{class_names_path}"
            )
        return names
    if num_classes == len(DEFAULT_CLASS_NAMES):
        return list(DEFAULT_CLASS_NAMES)
    return [str(i) for i in range(num_classes)]


def load_class_names_from_ckpt(
    ckpt_path: Path,
    class_names_path: Path | None = None,
) -> list[str]:
    """Load class names from a checkpoint, with optional file override."""
    if class_names_path is not None:
        return load_class_names_file(class_names_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "class_names" in ckpt and ckpt["class_names"]:
        names = [str(n) for n in ckpt["class_names"]]
        n = int(ckpt.get("num_classes", len(names)))
        if len(names) != n:
            raise ValueError(
                f"checkpoint class_names length {len(names)} != num_classes {n}: {ckpt_path}"
            )
        return names
    n = int(ckpt["num_classes"])
    return resolve_class_names(n, None)


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

    num_classes = resolve_num_classes(y_train, y_val, args.num_classes)
    class_names = resolve_class_names(num_classes, args.class_names)
    logger.info("class_names=%s", class_names)

    X_train = streams_to_features(train_streams)
    X_val = streams_to_features(val_streams)
    if X_train.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"expected {FEATURE_DIM} features, got {X_train.shape[1]}")

    model, scaler = train_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes,
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
            "num_classes": int(num_classes),
            "frontend": "welch",
            "n_fft": 160,
            "hop": 80,
            "sample_rate": 1600,
            "class_names": class_names,
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
        "num_classes": int(num_classes),
        "class_names": class_names,
        "checkpoint": str(ckpt_path),
        "frontend": "welch n_fft=160 hop=80",
        "feature_dim": FEATURE_DIM,
    }
    (args.output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Done. train_acc=%.4f val_acc=%.4f num_classes=%d",
        train_m["accuracy"],
        val_m["accuracy"],
        num_classes,
    )


if __name__ == "__main__":
    main()
