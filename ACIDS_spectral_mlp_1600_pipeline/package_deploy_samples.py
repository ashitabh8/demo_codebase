#!/usr/bin/env python3
"""
Package labeled 1600-stream .txt samples into deploy/samples, samples_txt, acids_samples.h.

Usage:
  python package_deploy_samples.py --sample_dir data/val --output_dir deploy --max_samples 32
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))

from load_1600_txt import load_one_txt

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


def safe_stem(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def write_acids_samples_h(
    output_dir: Path,
    stems: list[str],
    label_ids: list[int],
) -> None:
    n = len(stems)
    names_c = ",\n".join(f'    "{s}"' for s in stems)
    paths_c = ",\n".join(f'    "samples/{s}.bin"' for s in stems)
    ids_c = ",\n".join(f"    {i}" for i in label_ids)
    label_names = []
    for i in label_ids:
        if 0 <= i < len(CLASS_NAMES):
            label_names.append(CLASS_NAMES[i])
        else:
            label_names.append(str(i))
    names_labels_c = ",\n".join(f'    "{n}"' for n in label_names)

    body = f"""/* Auto-generated packaged sample manifest for spectral MLP C inference. */
#ifndef ACIDS_SAMPLES_H
#define ACIDS_SAMPLES_H

#define ACIDS_NUM_PACKAGED_SAMPLES {n}

static const char *ACIDS_SAMPLE_NAMES[ACIDS_NUM_PACKAGED_SAMPLES] = {{
{names_c}
}};

static const char *ACIDS_SAMPLE_PATHS[ACIDS_NUM_PACKAGED_SAMPLES] = {{
{paths_c}
}};

static const int ACIDS_SAMPLE_LABEL_IDS[ACIDS_NUM_PACKAGED_SAMPLES] = {{
{ids_c}
}};

static const char *ACIDS_SAMPLE_LABEL_NAMES[ACIDS_NUM_PACKAGED_SAMPLES] = {{
{names_labels_c}
}};

#endif /* ACIDS_SAMPLES_H */
"""
    (output_dir / "acids_samples.h").write_text(body, encoding="utf-8")
    print(f"  wrote acids_samples.h ({n} samples)")


def package_samples(sample_dir: Path, output_dir: Path, max_samples: int) -> None:
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)
    bin_dir = output_dir / "samples"
    txt_dir = output_dir / "samples_txt"
    if bin_dir.exists():
        shutil.rmtree(bin_dir)
    if txt_dir.exists():
        shutil.rmtree(txt_dir)
    bin_dir.mkdir(parents=True)
    txt_dir.mkdir(parents=True)

    paths = sorted(sample_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"no .txt in {sample_dir}")
    if max_samples > 0:
        paths = paths[:max_samples]

    stems: list[str] = []
    label_ids: list[int] = []
    used: set[str] = set()

    for path in paths:
        name, label, stream = load_one_txt(path)
        stem = safe_stem(name) or safe_stem(path.stem)
        base = stem
        k = 1
        while stem in used:
            stem = f"{base}_{k}"
            k += 1
        used.add(stem)

        txt_out = txt_dir / f"{stem}.txt"
        with txt_out.open("w", encoding="utf-8") as f:
            f.write(f"{stem}\n{int(label)}\nRAW_AUDIO_STREAM\n1600\n")
            for v in stream.tolist():
                f.write(f"{v}\n")

        bin_out = bin_dir / f"{stem}.bin"
        with bin_out.open("wb") as f:
            f.write(struct.pack(f"{len(stream)}f", *stream.tolist()))

        stems.append(stem)
        label_ids.append(int(label))
        print(f"  packaged {stem} label={label}")

    write_acids_samples_h(output_dir, stems, label_ids)
    print(f"Done. {len(stems)} samples in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Package 1600 txt samples into deploy/")
    parser.add_argument("--sample_dir", type=Path, default=PKG / "data" / "val")
    parser.add_argument("--output_dir", type=Path, default=PKG / "deploy")
    parser.add_argument("--max_samples", type=int, default=32)
    args = parser.parse_args()
    package_samples(args.sample_dir, args.output_dir, args.max_samples)


if __name__ == "__main__":
    main()
