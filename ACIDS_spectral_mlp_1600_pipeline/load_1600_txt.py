"""Load samples_txt-style RAW_AUDIO_STREAM files (1600 floats + label)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

EXPECTED_COUNT = 1600
LAYOUT_TAG = "RAW_AUDIO_STREAM"


def load_one_txt(path: Path) -> tuple[str, int, np.ndarray]:
    """Return (sample_name, label_id, stream[1600] float32)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4 + EXPECTED_COUNT:
        raise ValueError(f"{path}: expected at least {4 + EXPECTED_COUNT} lines, got {len(lines)}")

    name = lines[0].strip()
    label = int(lines[1].strip())
    layout = lines[2].strip()
    if layout != LAYOUT_TAG:
        raise ValueError(f"{path}: expected layout {LAYOUT_TAG!r}, got {layout!r}")
    count = int(lines[3].strip())
    if count != EXPECTED_COUNT:
        raise ValueError(f"{path}: expected count {EXPECTED_COUNT}, got {count}")

    vals: list[float] = []
    for line in lines[4:]:
        s = line.strip()
        if not s:
            continue
        vals.append(float(s))
    if len(vals) != EXPECTED_COUNT:
        raise ValueError(f"{path}: expected {EXPECTED_COUNT} floats, got {len(vals)}")
    return name, label, np.asarray(vals, dtype=np.float32)


def load_dir(dir_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load all *.txt in a directory (sorted by name).

    Returns:
        names: list of sample names
        streams: [N, 1600] float32
        labels: [N] int64
    """
    dir_path = Path(dir_path)
    paths = sorted(dir_path.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"no .txt files in {dir_path}")

    names: list[str] = []
    streams: list[np.ndarray] = []
    labels: list[int] = []
    for path in paths:
        name, label, stream = load_one_txt(path)
        names.append(name)
        streams.append(stream)
        labels.append(label)
    return (
        names,
        np.stack(streams, axis=0).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
    )
