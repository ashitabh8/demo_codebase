#!/usr/bin/env python3
"""
Evaluate the ch0 continuous spectral MLP on the ACIDS test split.

Uses the same feature pipeline as train_ch0_classifiers.py:
  ch0 -> continuous truncate [:1600] -> spectral features (83-dim) -> StandardScaler -> MLP
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

src2_path = Path(__file__).parent.parent
sys.path.insert(0, str(src2_path))

from data_analysis.analyze_mel_spectral_features import (
    build_combined_matrix,
    compute_spectral_features,
)
from data_analysis.train_ch0_classifiers import (
    SpectralMLP,
    collect_streams,
    eval_mlp,
    get_params,
    load_config,
)
from dataset_utils.MultiModalDataLoader import create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test spectral MLP on ACIDS vehicle_classification test split"
    )
    parser.add_argument("--yaml_path", type=str, default="../data/ACIDS.yaml")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../tmp/ch0_classifiers/models/spectral_mlp.pth",
    )
    parser.add_argument("--output_dir", type=str, default="../tmp/ch0_classifiers")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--spectral-hann", action="store_true")
    return parser.parse_args()


def load_mlp_checkpoint(path: Path, device: str) -> tuple[SpectralMLP, StandardScaler]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    in_dim = int(ckpt["in_dim"])
    num_classes = int(ckpt["num_classes"])
    model = SpectralMLP(in_dim, num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
    scaler.n_features_in_ = in_dim
    return model, scaler


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    config = load_config(args)
    params = get_params(config)
    task = config["task_name"]
    class_names = config[task]["class_names"]
    device = config["device"]

    _, _, test_loader = create_dataloaders(config)
    logger.info("Collecting test streams...")
    test_streams, y_test = collect_streams(test_loader, params, "Test streams")
    test_feats = compute_spectral_features(
        test_streams,
        sample_rate=params["sample_rate"],
        n_fft=params["n_fft"],
        use_hann=args.spectral_hann,
    )
    X_test = build_combined_matrix(test_feats)

    model, scaler = load_mlp_checkpoint(checkpoint_path, device)
    metrics = eval_mlp(
        model,
        scaler,
        X_test,
        y_test,
        class_names,
        "test",
        device,
        plots_dir,
    )

    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "num_test_samples": int(len(y_test)),
        "class_names": class_names,
        "metrics": metrics,
    }
    report_path = output_dir / "test_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(
        "spectral_mlp test: acc=%.4f macro_f1=%.4f (%d samples)",
        metrics["accuracy"],
        metrics["macro_f1"],
        len(y_test),
    )
    logger.info("Confusion matrix plot: %s", plots_dir / "confusion_test_mlp.png")
    logger.info("Results written to %s", report_path)


if __name__ == "__main__":
    main()
