#!/usr/bin/env python3
"""
Compare frozen spectral MLP under snapshot vs Welch feature extractors.

Snapshot (deploy default): single RFFT on stream[:, :n_fft] (first 160 samples).
Welch: mean power spectrum over overlapping n_fft windows across the full 1600 stream.

The MLP weights stay frozen. By default the training StandardScaler is reused for
both extractors (distribution shift for Welch is expected). Use --refit-welch-scaler
to fit a Welch-only scaler on train features while keeping the MLP frozen.

Also runs stress variants (tail snapshot) when --stress is set.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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
    eval_mlp_arrays,
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
        description="Snapshot vs Welch spectral features with frozen spectral MLP"
    )
    parser.add_argument("--yaml_path", type=str, default="../data/ACIDS.yaml")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../tmp/ch0_classifiers/models/spectral_mlp.pth",
    )
    parser.add_argument("--output_dir", type=str, default="../tmp/ch0_classifiers")
    parser.add_argument("--split", choices=["test", "val", "both"], default="test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-fft", type=int, default=160)
    parser.add_argument("--welch-hop", type=int, default=80)
    parser.add_argument("--spectral-hann", action="store_true")
    parser.add_argument(
        "--refit-welch-scaler",
        action="store_true",
        help="Fit StandardScaler on train Welch features; MLP stays frozen",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Also evaluate tail snapshot (last n_fft samples only)",
    )
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


def _stats_from_power(power: np.ndarray, freqs: np.ndarray) -> dict[str, np.ndarray]:
    psd = power / (power.sum(axis=-1, keepdims=True) + 1e-12)
    centroid = (power * freqs).sum(axis=-1) / (power.sum(axis=-1) + 1e-12)
    mean_energy = power.mean(axis=-1)
    return {
        "centroid": centroid.astype(np.float32),
        "mean_energy": mean_energy.astype(np.float32),
        "psd": psd.astype(np.float32),
    }


def compute_welch_spectral_features(
    streams: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop: int,
    use_hann: bool = False,
) -> dict[str, np.ndarray]:
    """Average periodogram over overlapping windows covering each full stream."""
    if hop <= 0:
        raise ValueError("welch hop must be positive")
    n_samples, t_len = streams.shape
    n_bins = n_fft // 2 + 1
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate).astype(np.float64)
    avg_power = np.zeros((n_samples, n_bins), dtype=np.float64)

    for i in range(n_samples):
        stream = streams[i].astype(np.float32)
        frame_powers = []
        last_start = t_len - n_fft
        if last_start < 0:
            starts = [0]
        else:
            starts = list(range(0, last_start + 1, hop))
            if starts[-1] != last_start:
                starts.append(last_start)
        for start in starts:
            frame = stream[start : start + n_fft]
            if frame.shape[0] < n_fft:
                pad = np.zeros(n_fft - frame.shape[0], dtype=np.float32)
                frame = np.concatenate([frame, pad], axis=0)
            if use_hann:
                win = np.hanning(n_fft).astype(np.float32)
                frame = frame * win
            fft = np.fft.rfft(frame, n=n_fft)
            frame_powers.append((fft.real ** 2 + fft.imag ** 2).astype(np.float64))
        avg_power[i] = np.mean(np.stack(frame_powers, axis=0), axis=0)

    return _stats_from_power(avg_power, freqs)


def compute_tail_snapshot_features(
    streams: np.ndarray,
    sample_rate: int,
    n_fft: int,
    use_hann: bool = False,
) -> dict[str, np.ndarray]:
    """Single FFT on the last n_fft samples (stress / robustness check)."""
    tail = streams[:, -n_fft:].astype(np.float32)
    if tail.shape[1] < n_fft:
        pad_width = n_fft - tail.shape[1]
        tail = np.pad(tail, ((0, 0), (pad_width, 0)))
    feats = compute_spectral_features(
        tail,
        sample_rate=sample_rate,
        n_fft=n_fft,
        use_hann=use_hann,
    )
    return feats


def evaluate_variant(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    model: SpectralMLP,
    scaler: StandardScaler,
    class_names: list[str],
    device: str,
) -> dict:
    result = eval_mlp_arrays(model, scaler, X, y, device)
    preds = result["predictions"]
    cm = confusion_matrix(y, preds, labels=list(range(len(class_names))))
    per_class = classification_report(
        y,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "variant": name,
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def print_summary(metrics: dict) -> None:
    logger.info(
        "%s: acc=%.4f macro_f1=%.4f",
        metrics["variant"],
        metrics["accuracy"],
        metrics["macro_f1"],
    )
    for cname, row in metrics["per_class"].items():
        if cname in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if not isinstance(row, dict):
            continue
        logger.info(
            "  class %-12s  f1=%.3f  recall=%.3f  support=%d",
            cname,
            row["f1-score"],
            row["recall"],
            int(row["support"]),
        )


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    config = load_config(args)
    params = get_params(config)
    task = config["task_name"]
    class_names = config[task]["class_names"]
    device = config["device"]
    sample_rate = params["sample_rate"]
    n_fft = args.n_fft

    train_loader, val_loader, test_loader = create_dataloaders(config)
    splits = []
    if args.split in ("val", "both"):
        splits.append(("val", val_loader))
    if args.split in ("test", "both"):
        splits.append(("test", test_loader))

    model, frozen_scaler = load_mlp_checkpoint(checkpoint_path, device)
    welch_scaler = frozen_scaler
    if args.refit_welch_scaler:
        logger.info("Collecting train streams for Welch scaler refit...")
        train_streams, _ = collect_streams(train_loader, params, "Train streams")
        train_welch = build_combined_matrix(
            compute_welch_spectral_features(
                train_streams,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop=args.welch_hop,
                use_hann=args.spectral_hann,
            )
        )
        welch_scaler = StandardScaler()
        welch_scaler.fit(train_welch)
        logger.info("Refit Welch StandardScaler on train features")

    all_results = []

    for split_name, loader in splits:
        logger.info("Collecting %s streams...", split_name)
        streams, y = collect_streams(loader, params, f"{split_name} streams")

        snapshot_feats = compute_spectral_features(
            streams,
            sample_rate=sample_rate,
            n_fft=n_fft,
            use_hann=args.spectral_hann,
        )
        welch_feats = compute_welch_spectral_features(
            streams,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop=args.welch_hop,
            use_hann=args.spectral_hann,
        )

        variants = [
            ("snapshot_first160", build_combined_matrix(snapshot_feats), frozen_scaler),
            ("welch_full_second", build_combined_matrix(welch_feats), welch_scaler),
        ]
        if args.stress:
            tail_feats = compute_tail_snapshot_features(
                streams,
                sample_rate=sample_rate,
                n_fft=n_fft,
                use_hann=args.spectral_hann,
            )
            variants.append(
                ("snapshot_tail160", build_combined_matrix(tail_feats), frozen_scaler)
            )

        split_metrics = []
        for variant_name, X, scaler in variants:
            full_name = f"{split_name}_{variant_name}"
            metrics = evaluate_variant(
                full_name, X, y, model, scaler, class_names, device
            )
            print_summary(metrics)
            split_metrics.append(metrics)

        all_results.append(
            {
                "split": split_name,
                "num_samples": int(len(y)),
                "n_fft": n_fft,
                "welch_hop": args.welch_hop,
                "refit_welch_scaler": bool(args.refit_welch_scaler),
                "variants": split_metrics,
            }
        )

    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "class_names": class_names,
        "description": (
            "Frozen MLP; snapshot uses first n_fft samples, "
            "Welch averages overlapping FFTs over full 1600-sample stream"
        ),
        "results": all_results,
    }
    out_path = output_dir / "snapshot_vs_welch_results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
