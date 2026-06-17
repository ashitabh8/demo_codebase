#!/usr/bin/env python3
"""
WestPoint stability training launcher.

The base WestPoint.yaml already enables inverse-frequency balanced sampling
(WeightedRandomSampler via use_balanced_sampling + compute_sample_weights_for_balanced_sampling).

This script adds a *separate* experiment and training recipe that additionally:
  - applies class-weighted CE inside ce_supcon (training_config ce_class_weights)
  - uses a slightly tighter SupCon temperature and lower supcon_weight
  - selects best checkpoints by pr_macro_mean (single_label_best_metric)

It writes a merged YAML to a temp file and invokes train.py as a subprocess so
the normal argparse path is unchanged.

Usage (from repo root or src2/train_test):
  cd src2/train_test
  python3 train_westpoint_stability.py --gpu 0
  python3 train_westpoint_stability.py --smoke --gpu 0
  python3 train_westpoint_stability.py --base_yaml ../data/WestPoint.yaml --class_counts 757,693,41006 --gpu 0
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_SRC2 = _THIS_DIR.parent
_DEFAULT_BASE = _SRC2 / "data" / "WestPoint.yaml"
_TRAIN_PY = _THIS_DIR / "train.py"

STABILITY_EXPERIMENT = "only_audio_deepsense_dw_large_mel_stability"
STABILITY_TRAINING_KEY = "vanilla_supervised_stability"
_BASE_EXPERIMENT_KEY = "only_audio_deepsense_dw_large_mel"


def _parse_counts(s: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--class_counts must be three integers: kona,wagoneer,background")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _ce_weights_from_counts(
    counts: tuple[int, int, int], mode: str
) -> list[float] | None:
    """
    Compute CE class weights from raw counts.

    mode:
      - "none": no CE weights (safe when WeightedRandomSampler already balances).
      - "inverse": majority_count / class_count (aggressive; can destroy majority).
      - "sqrt_inverse": sqrt(majority_count / class_count) (milder).
    """
    if mode == "none":
        return None
    n0, n1, n2 = counts
    m = max(n0, n1, n2)
    raw = [float(m) / float(n0), float(m) / float(n1), float(m) / float(n2)]
    if mode == "inverse":
        return raw
    if mode == "sqrt_inverse":
        return [float(r) ** 0.5 for r in raw]
    raise ValueError(
        f"unsupported ce_weight_mode '{mode}'; expected one of none/inverse/sqrt_inverse"
    )


def build_stability_config(
    base: dict,
    *,
    class_counts: tuple[int, int, int],
    ce_weight_mode: str,
    disable_balanced_sampling: bool,
    supcon_temperature: float,
    supcon_weight: float,
    smoke: bool,
) -> dict:
    merged = copy.deepcopy(base)
    experiments = merged["experiments"]
    if _BASE_EXPERIMENT_KEY not in experiments:
        raise KeyError(
            f"Base experiment '{_BASE_EXPERIMENT_KEY}' not found under experiments"
        )

    src_tc = merged["training_configs"]["vanilla_supervised_contrastive"]
    stability_tc = copy.deepcopy(src_tc)

    ce_w = _ce_weights_from_counts(class_counts, ce_weight_mode)
    if ce_w is None:
        stability_tc.pop("ce_class_weights", None)
    else:
        stability_tc["ce_class_weights"] = ce_w

    stability_tc["single_label_best_metric"] = "pr_macro_mean"
    stability_tc["supcon_temperature"] = float(supcon_temperature)
    stability_tc["supcon_weight"] = float(supcon_weight)
    if smoke:
        stability_tc["epochs"] = 1
    merged["training_configs"][STABILITY_TRAINING_KEY] = stability_tc

    if disable_balanced_sampling:
        merged["use_balanced_sampling"] = False

    new_exp = copy.deepcopy(experiments[_BASE_EXPERIMENT_KEY])
    new_exp["training"] = STABILITY_TRAINING_KEY
    experiments[STABILITY_EXPERIMENT] = new_exp

    return merged


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Launch WestPoint stability training")
    parser.add_argument(
        "--base_yaml",
        type=str,
        default=str(_DEFAULT_BASE),
        help="Path to base WestPoint.yaml",
    )
    parser.add_argument(
        "--class_counts",
        type=str,
        default="757,693,41006",
        help="Train-set counts kona,wagoneer,background (for CE weights)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--ce_weight_mode",
        type=str,
        default="none",
        choices=["none", "inverse", "sqrt_inverse"],
        help=(
            "CE class weighting: "
            "'none' (recommended when WeightedRandomSampler is already on), "
            "'inverse' (majority/count — aggressive, can destroy majority), "
            "'sqrt_inverse' (milder)."
        ),
    )
    parser.add_argument(
        "--no_balanced_sampling",
        action="store_true",
        help=(
            "Disable WeightedRandomSampler (use_balanced_sampling=False). "
            "Use this if you rely on CE class weights to handle imbalance."
        ),
    )
    parser.add_argument(
        "--supcon_temperature",
        type=float,
        default=0.07,
        help="SupCon temperature (lower = sharper).",
    )
    parser.add_argument(
        "--supcon_weight",
        type=float,
        default=1.0,
        help="Weight on SupCon term vs CE.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 epoch only (overrides stability training epochs)",
    )
    parser.add_argument(
        "--keep_merged_yaml",
        action="store_true",
        help="Do not delete the generated merged YAML after the run",
    )
    args = parser.parse_args()

    base_path = Path(args.base_yaml).expanduser().resolve()
    if not base_path.is_file():
        logging.error("Base YAML not found: %s", base_path)
        return 1

    with open(base_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    counts = _parse_counts(args.class_counts)
    ce_w = _ce_weights_from_counts(counts, args.ce_weight_mode)
    merged = build_stability_config(
        base_cfg,
        class_counts=counts,
        ce_weight_mode=args.ce_weight_mode,
        disable_balanced_sampling=args.no_balanced_sampling,
        supcon_temperature=args.supcon_temperature,
        supcon_weight=args.supcon_weight,
        smoke=args.smoke,
    )

    use_balanced_sampling = merged["use_balanced_sampling"]
    if use_balanced_sampling and ce_w is not None and args.ce_weight_mode == "inverse":
        logging.warning(
            "Both WeightedRandomSampler (use_balanced_sampling=True) and "
            "inverse CE class weights are enabled. This double-balances and "
            "can push the model to never predict the majority class. "
            "Prefer --ce_weight_mode none (sampler only) or --no_balanced_sampling."
        )

    logging.info("Stability recipe: experiment=%s", STABILITY_EXPERIMENT)
    logging.info("  training_config key: %s", STABILITY_TRAINING_KEY)
    logging.info("  class_counts: %s", list(counts))
    logging.info("  ce_weight_mode: %s", args.ce_weight_mode)
    logging.info("  ce_class_weights: %s", ce_w)
    logging.info("  use_balanced_sampling: %s", use_balanced_sampling)
    logging.info(
        "  supcon_temperature: %s",
        merged["training_configs"][STABILITY_TRAINING_KEY]["supcon_temperature"],
    )
    logging.info(
        "  supcon_weight: %s",
        merged["training_configs"][STABILITY_TRAINING_KEY]["supcon_weight"],
    )
    logging.info(
        "  epochs: %s",
        merged["training_configs"][STABILITY_TRAINING_KEY]["epochs"],
    )

    fd, tmp_name = tempfile.mkstemp(suffix="_westpoint_stability_merged.yaml", text=True)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w") as out:
            yaml.safe_dump(merged, out, default_flow_style=False, sort_keys=False)

        cmd = [
            sys.executable,
            str(_TRAIN_PY),
            "--experiment_name",
            STABILITY_EXPERIMENT,
            "--yaml_path",
            str(tmp_path),
            "--gpu",
            str(args.gpu),
        ]
        logging.info("Running: %s", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(_THIS_DIR))
        return int(proc.returncode)
    finally:
        if not args.keep_merged_yaml and tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
