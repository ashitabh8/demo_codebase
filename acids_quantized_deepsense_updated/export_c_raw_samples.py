#!/usr/bin/env python3
"""Export ACIDS packaged C samples as (3, 7, 25) float32 @ 1600 Hz (post-decimation)."""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SRC2 = Path(__file__).resolve().parent.parent / "src2"
sys.path.insert(0, str(SRC2))

from data_augmenter.audio_downsample import AudioDownsampler
from dataset_utils.multimodal_core import normalize_sample_data_layout

PKG = Path(__file__).resolve().parent
NUM_CHANNELS = 3
NUM_SEGMENTS = 7
SAMPLES_PER_SEGMENT = 25

SAMPLE_NAMES = [
    "Gv3c1090_96",
    "Gv6d1090_125",
    "Gv3c1090_231",
    "Gv1c2020_66",
    "Gv8c1220_99",
    "Gv1c2020_70",
    "Gv7c1056_4",
    "Gv3c1090_114",
    "Gv7c1056_178",
    "Gv9d1126_58",
    "Gv8c1220_180",
    "Gv9d1126_126",
    "Gv6d1090_6",
    "Gv3c1090_362",
    "Gv3c1090_215",
]

LABEL_IDS = [0, 2, 0, 0, 3, 0, 0, 0, 0, 4, 3, 4, 0, 0, 0]


def load_yaml_index(yaml_path: Path) -> list[str]:
    with open(yaml_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    index_path = config["vehicle_classification"]["test_index_file"]
    lines = np.loadtxt(index_path, dtype=str, ndmin=1)
    return [str(x).strip() for x in np.atleast_1d(lines).ravel() if str(x).strip()]


def find_pt_path(sample_name: str, index_paths: list[str]) -> str:
    for path in index_paths:
        if Path(path).stem.replace(" ", "_") == sample_name:
            return path
    raise FileNotFoundError(f"sample {sample_name!r} not found in test index")


def load_audio_chw_16k(pt_path: str) -> np.ndarray:
    sample = torch.load(pt_path, weights_only=False)
    normalize_sample_data_layout(
        sample,
        location_names=["shake"],
        loc_modalities={"shake": ["seismic", "audio"]},
    )
    audio = sample["data"]["shake"]["audio"].float()
    if audio.dim() == 3:
        chw = audio.numpy()
    else:
        chw = audio.reshape(NUM_CHANNELS, NUM_SEGMENTS, -1).numpy()
    if chw.shape != (NUM_CHANNELS, NUM_SEGMENTS, 256):
        raise ValueError(f"expected (3, 7, 256) in {pt_path}, got {chw.shape}")
    return chw.astype(np.float32)


def decimate_chw(chw_16k: np.ndarray, downsampler: AudioDownsampler) -> np.ndarray:
    x = torch.from_numpy(chw_16k).unsqueeze(0)
    out = downsampler.downsample({"shake": {"audio": x}})["shake"]["audio"]
    decimated = out.squeeze(0).numpy()
    if decimated.shape != (NUM_CHANNELS, NUM_SEGMENTS, SAMPLES_PER_SEGMENT):
        raise ValueError(f"unexpected decimated shape {decimated.shape}")
    return decimated.astype(np.float32)


def write_sample_txt(path: Path, sample_name: str, label_id: int, audio_chw: np.ndarray) -> None:
    flat = audio_chw.reshape(-1)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{sample_name}\n")
        handle.write(f"{int(label_id)}\n")
        handle.write("RAW_AUDIO_CHW\n")
        handle.write(f"{NUM_CHANNELS},{NUM_SEGMENTS},{SAMPLES_PER_SEGMENT}\n")
        for value in flat:
            handle.write(f"{float(value):.9g}\n")


def write_sample_bin(path: Path, audio_chw: np.ndarray) -> None:
    audio_chw.astype(np.float32).tofile(path)


def main() -> None:
    yaml_path = SRC2 / "data" / "ACIDS.yaml"
    index_paths = load_yaml_index(yaml_path)
    txt_dir = PKG / "samples_txt"
    bin_dir = PKG / "samples_bin"
    txt_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    downsampler = AudioDownsampler(16000, 1600, target_modalities=["audio"])

    for sample_name, label_id in zip(SAMPLE_NAMES, LABEL_IDS):
        pt_path = find_pt_path(sample_name, index_paths)
        chw_16k = load_audio_chw_16k(pt_path)
        chw_1600 = decimate_chw(chw_16k, downsampler)
        write_sample_txt(txt_dir / f"{sample_name}.txt", sample_name, label_id, chw_1600)
        write_sample_bin(bin_dir / f"{sample_name}.bin", chw_1600)
        print(f"wrote {sample_name} from {pt_path}")


if __name__ == "__main__":
    main()
