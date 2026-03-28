"""
Analyze label distribution for GracesQuarters multiclass finetuning data.

Reads train/val/test index files, loads each .pt sample, keeps audio-flagged
samples only, and reports single- vs multi-label stats and per-label /
combination counts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch


DEFAULT_INDEX_DIR = (
    "/data/misra8/GracesQuarters/index_files/2024-08-06-GQ-split-multiclass"
)


def read_index_paths(index_file: Path) -> list[str]:
    with open(index_file) as f:
        return [line.strip() for line in f if line.strip()]


def sample_has_audio_flag(sample: dict) -> bool:
    flag = sample["flag"]
    for _loc, mods in flag.items():
        if not isinstance(mods, dict):
            continue
        if "audio" not in mods:
            continue
        if mods["audio"] is True:
            return True
    return False


def label_tuple_from_sample(sample: dict) -> tuple[str, ...]:
    """Unique labels as a sorted tuple (one row per sample for combination stats)."""
    arr = sample["label"]
    items: list[str] = []
    for i in range(int(arr.size)):
        x = arr.flat[i]
        items.append(str(x))
    return tuple(sorted(set(items)))


def iter_distance_meters(obj: object) -> list[float]:
    """Collect numeric distances from distance (flat or nested dict of dicts)."""
    out: list[float] = []
    if not isinstance(obj, dict):
        return out
    for v in obj.values():
        if isinstance(v, dict):
            out.extend(iter_distance_meters(v))
        elif isinstance(v, (int, float)):
            out.append(float(v))
    return out


def min_distance_meters(sample: dict) -> float | None:
    """Minimum distance in meters across all entries under distance, or None if absent/empty."""
    if "distance" not in sample:
        return None
    vals = iter_distance_meters(sample["distance"])
    if not vals:
        return None
    return min(vals)


DISTANCE_NEAR_M = 10.0


def print_sample_structure(sample: dict, indent: int = 0) -> None:
    """Print nested keys, types, and tensor/array shapes for one loaded sample."""
    prefix = "  " * indent

    def line(msg: str) -> None:
        print(f"{prefix}{msg}")

    if isinstance(sample, dict):
        for k in sorted(sample.keys()):
            v = sample[k]
            if isinstance(v, dict):
                line(f"{k}: dict, keys = {sorted(v.keys())!s}")
                print_sample_structure(v, indent + 1)
            elif hasattr(v, "shape"):
                line(
                    f"{k}: array/tensor shape={tuple(v.shape)}, dtype={v.dtype}"
                )
                if k == "label":
                    vals = [str(v.flat[i]) for i in range(int(v.size))]
                    line(f"  label strings: {vals}")
            else:
                line(f"{k}: {type(v).__name__} = {v!r}")
    elif hasattr(sample, "shape"):
        line(f"shape={tuple(sample.shape)}, dtype={sample.dtype}")
    else:
        line(f"{type(sample).__name__} = {sample!r}")


def print_structure_overview() -> None:
    print("Expected .pt top-level structure (one representative audio sample):")
    print("  Top-level keys typically: flag, label, distance, data")
    print("  flag[location][modality] -> bool (e.g. audio / seismic present)")
    print("  label -> numpy object array of string class names (length 1+)")
    print("  distance -> dict: label/name -> float (meters); may nest like other fields")
    print("  data[location][modality] -> tensor (e.g. audio [1, 10, 1600])")
    print()
    print("Concrete nested layout from first loadable train path:")
    print()


def analyze_split(paths: list[str], split_name: str) -> dict:
    total = len(paths)
    audio_paths: list[str] = []
    load_errors = 0
    single_class = 0
    multi_class = 0
    combo_counts: dict[tuple[str, ...], int] = defaultdict(int)
    per_label_counts: dict[str, int] = defaultdict(int)
    dist_min_lt_10 = 0
    dist_min_ge_10 = 0
    dist_missing = 0

    for p in paths:
        try:
            sample = torch.load(p, weights_only=False)
        except Exception as e:
            print(f"  ERROR loading {p}: {e}")
            load_errors += 1
            continue

        if not sample_has_audio_flag(sample):
            continue
        audio_paths.append(p)

        try:
            combo = label_tuple_from_sample(sample)
        except Exception as e:
            print(f"  ERROR reading label from {p}: {e}")
            load_errors += 1
            continue

        combo_counts[combo] += 1
        if len(combo) == 1:
            single_class += 1
        else:
            multi_class += 1

        for lbl in combo:
            per_label_counts[lbl] += 1

        dmin = min_distance_meters(sample)
        if dmin is None:
            dist_missing += 1
        elif dmin < DISTANCE_NEAR_M:
            dist_min_lt_10 += 1
        else:
            dist_min_ge_10 += 1

    n_audio = len(audio_paths)
    return {
        "split_name": split_name,
        "total_listed": total,
        "audio_samples": n_audio,
        "single_class": single_class,
        "multi_class": multi_class,
        "combo_counts": dict(combo_counts),
        "per_label_counts": dict(per_label_counts),
        "load_errors": load_errors,
        "dist_min_lt_10": dist_min_lt_10,
        "dist_min_ge_10": dist_min_ge_10,
        "dist_missing": dist_missing,
    }


def pct(part: int, whole: int) -> str:
    if whole == 0:
        return "n/a"
    return f"{100.0 * part / whole:.1f}%"


def print_split_report(stats: dict) -> None:
    name = stats["split_name"]
    total = stats["total_listed"]
    n_audio = stats["audio_samples"]
    sc = stats["single_class"]
    mc = stats["multi_class"]

    print(f"=== {name.upper()} ({total} listed in index, {n_audio} with audio flag) ===")
    print(f"Single-label samples:  {sc:5d}  ({pct(sc, n_audio)})")
    print(f"Multi-label samples:   {mc:5d}  ({pct(mc, n_audio)})")
    print()

    lt10 = stats["dist_min_lt_10"]
    ge10 = stats["dist_min_ge_10"]
    dmis = stats["dist_missing"]
    print(
        f"Distance (min over all entries in sample['distance'], vs {DISTANCE_NEAR_M:g} m):"
    )
    print(f"  min distance < {DISTANCE_NEAR_M:g} m:  {lt10:5d}  ({pct(lt10, n_audio)})")
    print(f"  min distance >= {DISTANCE_NEAR_M:g} m: {ge10:5d}  ({pct(ge10, n_audio)})")
    print(f"  missing/empty distance:        {dmis:5d}  ({pct(dmis, n_audio)})")
    print()

    print("Per-label counts (audio samples that contain label):")
    for lbl in sorted(stats["per_label_counts"]):
        print(f"  {lbl:20s} : {stats['per_label_counts'][lbl]}")
    print()

    print("Label combination counts:")
    combos = sorted(stats["combo_counts"].items(), key=lambda x: (-x[1], x[0]))
    for combo, c in combos:
        print(f"  {combo!s:40s} : {c}")
    print()

    if stats["load_errors"]:
        print(f"Load/label errors: {stats['load_errors']}")
        print()


def merge_combo_dicts(
    a: dict[tuple[str, ...], int], b: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out[k] + v if k in out else v
    return out


def merge_label_dicts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out[k] + v if k in out else v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="GQ multiclass label distribution")
    parser.add_argument(
        "--index_dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help="Directory containing train_index.txt, val_index.txt, test_index.txt",
    )
    args = parser.parse_args()
    index_dir = Path(args.index_dir)

    splits = [
        ("train", index_dir / "train_index.txt"),
        ("val", index_dir / "val_index.txt"),
        ("test", index_dir / "test_index.txt"),
    ]

    train_path = index_dir / "train_index.txt"
    if train_path.is_file():
        print_structure_overview()
        for p in read_index_paths(train_path):
            try:
                sample = torch.load(p, weights_only=False)
            except Exception:
                continue
            if not sample_has_audio_flag(sample):
                continue
            print(f"  file: {p}")
            print_sample_structure(sample)
            print()
            break
        else:
            print("  (no loadable audio sample found for structure dump)")
            print()

    all_stats: list[dict] = []
    for split_name, path in splits:
        if not path.is_file():
            print(f"Missing index file: {path}")
            continue
        paths = read_index_paths(path)
        stats = analyze_split(paths, split_name)
        all_stats.append(stats)
        print_split_report(stats)

    if not all_stats:
        return

    total_listed = sum(s["total_listed"] for s in all_stats)
    n_audio = sum(s["audio_samples"] for s in all_stats)
    sc = sum(s["single_class"] for s in all_stats)
    mc = sum(s["multi_class"] for s in all_stats)
    combo_merged: dict[tuple[str, ...], int] = {}
    label_merged: dict[str, int] = {}
    err = sum(s["load_errors"] for s in all_stats)
    lt10_all = sum(s["dist_min_lt_10"] for s in all_stats)
    ge10_all = sum(s["dist_min_ge_10"] for s in all_stats)
    dmis_all = sum(s["dist_missing"] for s in all_stats)
    for s in all_stats:
        combo_merged = merge_combo_dicts(combo_merged, s["combo_counts"])
        label_merged = merge_label_dicts(label_merged, s["per_label_counts"])

    combined = {
        "split_name": "all_splits",
        "total_listed": total_listed,
        "audio_samples": n_audio,
        "single_class": sc,
        "multi_class": mc,
        "combo_counts": combo_merged,
        "per_label_counts": label_merged,
        "load_errors": err,
        "dist_min_lt_10": lt10_all,
        "dist_min_ge_10": ge10_all,
        "dist_missing": dmis_all,
    }
    print("=" * 60)
    print_split_report(combined)


if __name__ == "__main__":
    main()
