#!/usr/bin/env python3
"""
Export ACIDS .pt index files to flat ch0 continuous streams (1600 @ 1600 Hz).

Writes:
  <output_root>/train_X.npy  (N,1600) float32
  <output_root>/train_y.npy  (N,) int64
  <output_root>/val_X.npy
  <output_root>/val_y.npy
  <output_root>/test_X.npy
  <output_root>/test_y.npy
  <output_root>/test_names.txt
  <output_root>/test_txt/*.txt   # for C packaging / spectral_txt_to_bin
  <output_root>/test_bin/*.bin   # raw float32[1600] for spectral_infer

No src2 dependency: ch0 flatten + [:1600] matches AcidsContinuousTruncator.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

PKG = Path(__file__).resolve().parent
NUM_SEGMENTS = 7
PADDED_SEG_LEN = 256
USED_SAMPLES = 1600
EXPECTED_FLAT = NUM_SEGMENTS * PADDED_SEG_LEN  # 1792


def read_index(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def assert_disjoint_splits(
    train_paths: list[str],
    val_paths: list[str],
    test_paths: list[str],
    *,
    allow_overlap: bool,
) -> None:
    """Require a true train/val/test split unless allow_overlap is set."""
    train_set, val_set, test_set = set(train_paths), set(val_paths), set(test_paths)
    overlaps = {
        "train∩val": len(train_set & val_set),
        "train∩test": len(train_set & test_set),
        "val∩test": len(val_set & test_set),
    }
    if val_paths == test_paths:
        msg = (
            "val_index and test_index are identical — not a train/val/test split. "
            "Point paths.yaml at fixed_partition_index_vehicle_classification "
            "(disjoint), or pass --allow_overlapping_splits."
        )
        if allow_overlap:
            print(f"WARNING: {msg}")
            return
        raise SystemExit(f"ERROR: {msg}")
    bad = {k: n for k, n in overlaps.items() if n > 0}
    if bad:
        msg = f"overlapping samples across splits: {bad}"
        if allow_overlap:
            print(f"WARNING: {msg}")
            return
        raise SystemExit(
            f"ERROR: {msg}. Use disjoint indices or --allow_overlapping_splits."
        )


def safe_stem(pt_path: str) -> str:
    stem = Path(pt_path).stem.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "_", stem)


def load_ch0_stream(pt_path: str) -> tuple[np.ndarray, int]:
    sample = torch.load(pt_path, map_location="cpu", weights_only=False)
    audio = sample["data"]["shake"]["audio"]
    if not torch.is_tensor(audio):
        audio = torch.as_tensor(audio)
    audio = audio.float()
    if audio.dim() != 3:
        raise ValueError(f"{pt_path}: expected audio (C,7,256), got {tuple(audio.shape)}")
    if audio.shape[1] != NUM_SEGMENTS or audio.shape[2] != PADDED_SEG_LEN:
        raise ValueError(
            f"{pt_path}: expected (*,{NUM_SEGMENTS},{PADDED_SEG_LEN}), got {tuple(audio.shape)}"
        )
    stream = audio[0].reshape(-1)[:USED_SAMPLES].cpu().numpy().astype(np.float32)
    if stream.shape != (USED_SAMPLES,):
        raise ValueError(f"{pt_path}: bad stream shape {stream.shape}")

    label = sample["label"]["vehicle_type"]
    if torch.is_tensor(label):
        label_id = int(label.item()) if label.numel() == 1 else int(label[0].item())
    else:
        label_id = int(label)
    return stream, label_id


def export_split(
    index_path: Path,
    split_name: str,
    output_root: Path,
    write_test_files: bool,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths = read_index(index_path)
    if max_samples > 0:
        paths = paths[:max_samples]

    streams = []
    labels = []
    names = []
    used_names: set[str] = set()

    txt_dir = output_root / "test_txt"
    bin_dir = output_root / "test_bin"
    if write_test_files:
        txt_dir.mkdir(parents=True, exist_ok=True)
        bin_dir.mkdir(parents=True, exist_ok=True)

    for pt_path in tqdm(paths, desc=f"export {split_name}"):
        stream, label_id = load_ch0_stream(pt_path)
        stem = safe_stem(pt_path)
        base = stem
        k = 1
        while stem in used_names:
            stem = f"{base}_{k}"
            k += 1
        used_names.add(stem)

        streams.append(stream)
        labels.append(label_id)
        names.append(stem)

        if write_test_files:
            txt_path = txt_dir / f"{stem}.txt"
            with txt_path.open("w", encoding="utf-8") as f:
                f.write(f"{stem}\n{label_id}\nRAW_AUDIO_STREAM\n{USED_SAMPLES}\n")
                for v in stream.tolist():
                    f.write(f"{float(v):.9g}\n")
            stream.astype(np.float32).tofile(bin_dir / f"{stem}.bin")

    X = np.stack(streams, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    np.save(output_root / f"{split_name}_X.npy", X)
    np.save(output_root / f"{split_name}_y.npy", y)
    (output_root / f"{split_name}_names.txt").write_text(
        "\n".join(names) + "\n", encoding="utf-8"
    )
    print(f"  {split_name}: X={X.shape} y={y.shape} -> {output_root}")
    return X, y, names


def load_paths_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_index_path(path: Path, base_dir: Path) -> Path:
    """Resolve index paths relative to the package (or paths.yaml) directory."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def main():
    parser = argparse.ArgumentParser(description="Export ACIDS indices to 1600 streams")
    parser.add_argument("--paths_yaml", type=Path, default=PKG / "paths.yaml")
    parser.add_argument("--train_index", type=Path, default=None)
    parser.add_argument("--val_index", type=Path, default=None)
    parser.add_argument("--test_index", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=PKG / "exported_acids")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Cap each split (0 = all). Useful for smoke tests.",
    )
    parser.add_argument(
        "--allow_overlapping_splits",
        action="store_true",
        help="Allow val==test or other overlap (not recommended).",
    )
    args = parser.parse_args()

    cfg = {}
    paths_yaml = Path(args.paths_yaml)
    base_dir = paths_yaml.resolve().parent if paths_yaml.exists() else PKG
    if paths_yaml.exists():
        cfg = load_paths_yaml(paths_yaml)

    train_index = resolve_index_path(
        args.train_index or Path(cfg["train_index_file"]), base_dir
    )
    val_index = resolve_index_path(
        args.val_index or Path(cfg["val_index_file"]), base_dir
    )
    test_index = resolve_index_path(
        args.test_index or Path(cfg["test_index_file"]), base_dir
    )

    train_paths = read_index(train_index)
    val_paths = read_index(val_index)
    test_paths = read_index(test_index)
    assert_disjoint_splits(
        train_paths,
        val_paths,
        test_paths,
        allow_overlap=args.allow_overlapping_splits,
    )
    print(
        f"splits: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)} "
        f"(disjoint)"
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "train_index_file": str(train_index),
        "val_index_file": str(val_index),
        "test_index_file": str(test_index),
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "n_test": len(test_paths),
        "used_samples": USED_SAMPLES,
        "expected_flat_before_truncate": EXPECTED_FLAT,
        "channel": 0,
    }
    (args.output_root / "export_meta.yaml").write_text(
        yaml.safe_dump(meta), encoding="utf-8"
    )

    export_split(train_index, "train", args.output_root, False, args.max_samples)
    export_split(val_index, "val", args.output_root, False, args.max_samples)
    export_split(test_index, "test", args.output_root, True, args.max_samples)
    print(f"Done. Exported under {args.output_root}")


if __name__ == "__main__":
    main()
