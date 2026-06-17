"""
analyze_crossday_shift.py
=========================
Cross-day domain shift analysis for GracesQuarters audio classifier.
Compares Day 1 (2024-08-06) vs Day 2 (2024-08-07) on:
  1. Class balance
  2. Signal-level stats  (RMS/dBFS, dynamic range, per-mic level variance)
  2c. Temporal envelope  (peak position, envelope CV, clip rate — gain / limiter sanity)
  3. Spectral stats      (centroid, rolloff, flatness, ZCR; onset-strength / flux)
  3b. Background floor   (RMS + flatness for label ``background`` only — BGN level vs color)
  4. CMVN stats          (per-bin log-mel mean & std — reveals min/max scaler leakage)
  5. Mean log-mel heatmaps
  6. Feature-space       (PCA + UMAP of mean log-mel vectors, colored by day & class)
  7. Linear-frequency subband energy (FFT band ratios — B3 EQ / shelf tilt)
  8. Text summary        (00_summary.txt)

Usage
-----
    # Recommended: merge train + val + test for each day (same order as index files).
    python analyze_data.py \
        --day1_indices \
            /data/misra8/GracesQuarters/index_files/2024-08-06-GQ-split-multiclass/train_index.txt \
            /data/misra8/GracesQuarters/index_files/2024-08-06-GQ-split-multiclass/val_index.txt \
            /data/misra8/GracesQuarters/index_files/2024-08-06-GQ-split-multiclass/test_index.txt \
        --day2_indices \
            /data/misra8/GracesQuarters/index_files/2024-08-07-GQ-split-multiclass/train_index.txt \
            /data/misra8/GracesQuarters/index_files/2024-08-07-GQ-split-multiclass/val_index.txt \
            /data/misra8/GracesQuarters/index_files/2024-08-07-GQ-split-multiclass/test_index.txt \
        --max_per_class 120 \
        --sr 16000 \
        --n_mels 64 \
        --out_dir ./crossday_analysis

    # Same as above, using split directories (expects train_index.txt, val_index.txt, test_index.txt).
    python analyze_data.py \
        --day1_split_dir /data/misra8/GracesQuarters/index_files/2024-08-06-GQ-split-multiclass \
        --day2_split_dir /data/misra8/GracesQuarters/index_files/2024-08-07-GQ-split-multiclass \
        --out_dir ./crossday_analysis

    # Legacy: single index file per day (test-only).
    python analyze_data.py --day1_index .../test_index.txt --day2_index .../test_index.txt

Optional flags
--------------
    --mic_idx      which mic channel to use for mono analysis (default: 0)
    --no_umap      skip UMAP (faster; still produces PCA plot)
    --seed         random seed for reproducible subsampling
"""

import argparse
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=FutureWarning)

# ── optional heavy deps ────────────────────────────────────────────────────────
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[warn] librosa not found — spectral feature extraction disabled. "
          "Install with: pip install librosa")

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[warn] umap-learn not found — UMAP plot disabled. "
          "Install with: pip install umap-learn")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── palette ───────────────────────────────────────────────────────────────────
DAY_COLOR   = {"day1": "#2196F3", "day2": "#FF5722"}   # blue / orange
CLASS_CMAP  = plt.get_cmap("tab10")

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_index(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def load_merged_indices(paths):
    """
    Load multiple index files and merge into one list.
    Order: first file first; duplicate lines (same .pt path) appear once.
    """
    ordered_unique = []
    seen = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        for line in load_index(str(path)):
            if line not in seen:
                seen.add(line)
                ordered_unique.append(line)
    return ordered_unique


def indices_from_split_dir(dir_path):
    """
    Return [train, val, test] index paths for a GracesQuarters-style split directory.
    """
    d = Path(dir_path)
    names = ("train_index.txt", "val_index.txt", "test_index.txt")
    paths = [d / n for n in names]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"Expected {p} when using --day*_split_dir (need train/val/test index files)."
            )
    return [str(p) for p in paths]


def load_sample(path, mic_idx=0):
    """Return (label_str, audio_1d_np) for one .pt file."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    label = str(d["label"][0])
    # shape [1, n_mics, T]
    audio = d["data"]["shake"]["audio"][0, mic_idx].numpy().astype(np.float32)
    return label, audio


def collect_dataset(index_paths, max_per_class, mic_idx, seed):
    """
    Load up to max_per_class clips per class.
    Returns dict: label -> list of 1-D numpy arrays
    """
    rng = random.Random(seed)
    # group paths by label first (requires a cheap peek — use filename heuristic
    # then verify on load)
    rng.shuffle(index_paths)

    per_class = defaultdict(list)
    skipped = 0
    for path in index_paths:
        # fast-skip classes already full (check after first real load)
        try:
            label, audio = load_sample(path, mic_idx)
        except Exception as e:
            skipped += 1
            continue
        if len(per_class[label]) < max_per_class:
            per_class[label].append(audio)
        if all(len(v) >= max_per_class for v in per_class.values()) and \
                len(per_class) >= 2:
            # keep scanning — new classes may still appear
            pass
    if skipped:
        print(f"  [warn] skipped {skipped} files on load error")
    return dict(per_class)


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def rms_dbfs(audio):
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def dynamic_range_db(audio):
    peak = np.max(np.abs(audio)) + 1e-12
    floor = np.percentile(np.abs(audio) + 1e-12, 5)
    return 20 * np.log10(peak / floor)


def compute_logmel(audio, sr, n_mels, n_fft=512, hop_length=160):
    if not HAS_LIBROSA:
        return None
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=sr,
        n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        fmin=20, fmax=sr // 2,
    )
    return np.log(mel + 1e-6)   # shape [n_mels, T]


def spectral_features(audio, sr):
    """Returns dict of scalar spectral descriptors."""
    if not HAS_LIBROSA:
        return {}
    centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
    rolloff   = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85).mean()
    flatness  = librosa.feature.spectral_flatness(y=audio).mean()
    zcr       = librosa.feature.zero_crossing_rate(audio).mean()
    # Onset strength envelope ~ spectral-flux scale; useful B7 reverb / transient cue
    try:
        onset = librosa.onset.onset_strength(y=audio.astype(np.float32), sr=sr)
        spectral_flux = float(np.mean(onset)) if onset is not None and len(onset) > 0 else 0.0
    except Exception:
        spectral_flux = 0.0
    return dict(
        centroid=float(centroid),
        rolloff=float(rolloff),
        flatness=float(flatness),
        zcr=float(zcr),
        spectral_flux=spectral_flux,
    )


def temporal_envelope_features(audio, sr, hop=160, frame_len=512):
    """
    Short-time RMS envelope shape + clipping sanity (gain-saturation confound).
    Uses librosa.util.frame when available; numpy framing otherwise.
    """
    audio = np.asarray(audio, dtype=np.float32)
    n = int(audio.shape[0])
    clip_rate = float((np.abs(audio) > 0.98).mean()) if n > 0 else 0.0
    if n < frame_len:
        return dict(peak_rel=float("nan"), clip_rate=clip_rate, env_cv=float("nan"))

    n_frames = 1 + (n - frame_len) // hop
    if n_frames < 1:
        return dict(peak_rel=float("nan"), clip_rate=clip_rate, env_cv=float("nan"))

    if HAS_LIBROSA:
        frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop)
    else:
        frames = np.zeros((frame_len, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop
            frames[:, i] = audio[start : start + frame_len]

    env = np.sqrt((frames ** 2).mean(axis=0) + 1e-12)
    peak_rel = float(np.argmax(env) / max(len(env), 1))
    env_cv = float(env.std() / (env.mean() + 1e-12))
    return dict(peak_rel=peak_rel, clip_rate=clip_rate, env_cv=env_cv)


def fft_subband_ratios(
    audio,
    sr,
    bands=None,
):
    """
    Normalized linear-frequency subband energy (rFFT magnitude^2).
    Keys are stable column names: sb_20_500, sb_500_2000, ...
    """
    if bands is None:
        bands = [(20, 500), (500, 2000), (2000, 4000), (4000, 8000)]
    audio = np.asarray(audio, dtype=np.float64)
    n = audio.shape[0]
    if n < 8:
        return {f"sb_{lo}_{hi}": float("nan") for lo, hi in bands}
    w = np.hanning(n).astype(np.float64)
    spec = np.fft.rfft(audio * w)
    mag2 = (np.abs(spec) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    total = mag2.sum() + 1e-12
    out = {}
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        out[f"sb_{lo}_{hi}"] = float(mag2[m].sum() / total)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Per-day feature tables
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_table(per_class, sr, n_mels):
    """
    Returns:
        rows      list of dicts (one per clip)
        mels      dict label -> list of log-mel arrays [n_mels, T]
    """
    rows = []
    mels = defaultdict(list)
    for label, clips in per_class.items():
        for audio in clips:
            row = {"label": label, "rms_db": rms_dbfs(audio),
                   "dr_db": dynamic_range_db(audio)}
            row.update(spectral_features(audio, sr))
            row.update(temporal_envelope_features(audio, sr))
            row.update(fft_subband_ratios(audio, sr))
            rows.append(row)
            lm = compute_logmel(audio, sr, n_mels)
            if lm is not None:
                mels[label].append(lm)
    return rows, dict(mels)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def violin_compare(ax, d1_vals, d2_vals, title, ylabel):
    parts = ax.violinplot([d1_vals, d2_vals], positions=[0, 1],
                          showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], [DAY_COLOR["day1"], DAY_COLOR["day2"]]):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Day 1", "Day 2"])
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def mean_logmel_heatmap(ax, mel_list, title, vmin=None, vmax=None):
    stack = np.stack([m.mean(axis=-1) for m in mel_list])   # [N, n_mels]
    mean_spec = stack.mean(axis=0)
    im = ax.imshow(mean_spec[::-1, np.newaxis], aspect="auto",
                   vmin=vmin, vmax=vmax, cmap="magma")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_ylabel("Mel bin", fontsize=8)
    return im, mean_spec


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 – Class balance
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_balance(d1, d2, out_dir):
    all_classes = sorted(set(list(d1.keys()) + list(d2.keys())))
    x = np.arange(len(all_classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    c1 = [len(d1.get(c, [])) for c in all_classes]
    c2 = [len(d2.get(c, [])) for c in all_classes]
    ax.bar(x - w/2, c1, w, label="Day 1", color=DAY_COLOR["day1"], alpha=0.85)
    ax.bar(x + w/2, c2, w, label="Day 2", color=DAY_COLOR["day2"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(all_classes, rotation=20)
    ax.set_ylabel("Clip count"); ax.set_title("Class balance — Day 1 vs Day 2")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "01_class_balance.png", dpi=150)
    plt.close(fig)
    print("  [1] class balance saved")
    return all_classes


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 – Signal-level stats
# ══════════════════════════════════════════════════════════════════════════════

def plot_signal_stats(rows1, rows2, out_dir):
    metrics = [("rms_db", "RMS level (dBFS)"),
               ("dr_db",  "Dynamic range (dB)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 4))
    for ax, (key, label) in zip(axes, metrics):
        v1 = [r[key] for r in rows1 if key in r]
        v2 = [r[key] for r in rows2 if key in r]
        violin_compare(ax, v1, v2, label, label)

    fig.suptitle("Signal-level statistics — Day 1 vs Day 2", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "02_signal_stats.png", dpi=150)
    plt.close(fig)

    # also per-class RMS
    all_labels = sorted(set(r["label"] for r in rows1 + rows2))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, lbl in enumerate(all_labels):
        v1 = [r["rms_db"] for r in rows1 if r["label"] == lbl]
        v2 = [r["rms_db"] for r in rows2 if r["label"] == lbl]
        offset = i * 3
        if v1: ax.scatter([offset]*len(v1), v1, color=DAY_COLOR["day1"],
                           s=8, alpha=0.5)
        if v2: ax.scatter([offset+1]*len(v2), v2, color=DAY_COLOR["day2"],
                           s=8, alpha=0.5)
        if v1: ax.hlines(np.median(v1), offset-0.4, offset+0.4,
                          colors=DAY_COLOR["day1"], linewidths=2)
        if v2: ax.hlines(np.median(v2), offset+0.6, offset+1.4,
                          colors=DAY_COLOR["day2"], linewidths=2)
    tick_pos = [i*3 + 0.5 for i in range(len(all_labels))]
    ax.set_xticks(tick_pos); ax.set_xticklabels(all_labels, rotation=20)
    ax.set_ylabel("RMS (dBFS)")
    ax.set_title("Per-class RMS level — Day 1 (blue) vs Day 2 (orange)")
    legend_els = [Line2D([0],[0], color=DAY_COLOR["day1"], lw=2, label="Day 1"),
                  Line2D([0],[0], color=DAY_COLOR["day2"], lw=2, label="Day 2")]
    ax.legend(handles=legend_els); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "02b_perclass_rms.png", dpi=150)
    plt.close(fig)
    print("  [2] signal stats saved")


# ══════════════════════════════════════════════════════════════════════════════
# Section 2c – Temporal envelope (B4 / gain confounds)
# ══════════════════════════════════════════════════════════════════════════════

def plot_temporal_envelope(rows1, rows2, out_dir):
    """Violin: peak_rel, env_cv; bar: mean clip_rate per class per day."""
    keys_env = [("peak_rel", "Envelope peak position (0=front, 1=tail)"),
                ("env_cv", "Envelope coeff. of variation")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (key, title) in zip(axes, keys_env):
        v1 = [r[key] for r in rows1 if key in r and not np.isnan(r[key])]
        v2 = [r[key] for r in rows2 if key in r and not np.isnan(r[key])]
        if not v1 and not v2:
            ax.set_title(f"{title}\n(no data)")
            continue
        violin_compare(ax, v1, v2, title, "")
    fig.suptitle("Temporal envelope — Day 1 vs Day 2", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "02c_temporal_envelope_violins.png", dpi=150)
    plt.close(fig)

    all_labels = sorted(set(r["label"] for r in rows1 + rows2))
    x = np.arange(len(all_labels))
    w = 0.35
    means1, means2 = [], []
    for lbl in all_labels:
        c1 = [r["clip_rate"] for r in rows1 if r["label"] == lbl and "clip_rate" in r]
        c2 = [r["clip_rate"] for r in rows2 if r["label"] == lbl and "clip_rate" in r]
        means1.append(float(np.mean(c1)) if c1 else 0.0)
        means2.append(float(np.mean(c2)) if c2 else 0.0)
    fig, ax = plt.subplots(figsize=(max(8, len(all_labels) * 1.2), 4))
    ax.bar(x - w / 2, means1, w, label="Day 1", color=DAY_COLOR["day1"], alpha=0.85)
    ax.bar(x + w / 2, means2, w, label="Day 2", color=DAY_COLOR["day2"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=20)
    ax.set_ylabel("Mean clip rate (|x|>0.98)")
    ax.set_title("Clip rate by class — sanity check for gain / saturation between days")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "02d_clip_rate_by_class.png", dpi=150)
    plt.close(fig)
    print("  [2c] temporal envelope + clip rate saved")


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 – Spectral descriptors
# ══════════════════════════════════════════════════════════════════════════════

def plot_spectral_stats(rows1, rows2, out_dir):
    if not HAS_LIBROSA:
        print("  [3] skipped — librosa not available")
        return
    keys = [("centroid", "Spectral centroid (Hz)"),
            ("rolloff",  "Spectral rolloff 85% (Hz)"),
            ("flatness", "Spectral flatness"),
            ("zcr",      "Zero-crossing rate")]
    fig, axes = plt.subplots(1, len(keys), figsize=(14, 4))
    for ax, (key, label) in zip(axes, keys):
        v1 = [r[key] for r in rows1 if key in r]
        v2 = [r[key] for r in rows2 if key in r]
        violin_compare(ax, v1, v2, label, "")
    fig.suptitle("Spectral descriptors — Day 1 vs Day 2", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "03_spectral_stats.png", dpi=150)
    plt.close(fig)
    print("  [3] spectral stats saved")


# ══════════════════════════════════════════════════════════════════════════════
# Section 3b – Background floor (B2 SNR / BGN level vs color)
# ══════════════════════════════════════════════════════════════════════════════

def plot_background_floor(rows1, rows2, out_dir):
    """RMS + spectral flatness for ``background`` clips only."""
    bg1 = [r for r in rows1 if str(r["label"]).lower() == "background"]
    bg2 = [r for r in rows2 if str(r["label"]).lower() == "background"]
    if not bg1 and not bg2:
        print("  [3b] skipped — no background-class samples in either day")
        return
    if not HAS_LIBROSA:
        print("  [3b] skipped — librosa required for flatness comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    v1_rms = [r["rms_db"] for r in bg1]
    v2_rms = [r["rms_db"] for r in bg2]
    v1_flat = [r["flatness"] for r in bg1 if "flatness" in r]
    v2_flat = [r["flatness"] for r in bg2 if "flatness" in r]
    violin_compare(axes[0], v1_rms, v2_rms, "Background RMS (dBFS)", "dBFS")
    violin_compare(axes[1], v1_flat, v2_flat, "Background spectral flatness", "flatness")
    fig.suptitle("Background class — level vs timbral noise (B2 / BGN diagnostic)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "03b_background_floor.png", dpi=150)
    plt.close(fig)
    print("  [3b] background floor saved")


# ══════════════════════════════════════════════════════════════════════════════
# Section 3c – Spectral flux (B7 reverb / transient cue)
# ══════════════════════════════════════════════════════════════════════════════

def plot_spectral_flux(rows1, rows2, out_dir):
    if not HAS_LIBROSA:
        print("  [3c] skipped — librosa not available")
        return
    v1 = [r["spectral_flux"] for r in rows1 if "spectral_flux" in r]
    v2 = [r["spectral_flux"] for r in rows2 if "spectral_flux" in r]
    if not v1 and not v2:
        print("  [3c] skipped — no spectral_flux values")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    violin_compare(ax, v1, v2, "Onset strength mean (spectral-flux scale)", "")
    fig.suptitle("Spectral flux proxy — Day 1 vs Day 2", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "03c_spectral_flux.png", dpi=150)
    plt.close(fig)
    print("  [3c] spectral flux saved")


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 – CMVN / per-bin log-mel stats (leakage diagnostic)
# ══════════════════════════════════════════════════════════════════════════════

def plot_cmvn_stats(mels1, mels2, all_classes, out_dir):
    if not HAS_LIBROSA:
        print("  [4] skipped — librosa not available")
        return

    # flatten all clips per day
    def stack_all(mels_dict):
        clips = []
        for v in mels_dict.values():
            clips.extend(v)
        if not clips:
            return None
        # time-average each clip -> [n_mels], then stack -> [N, n_mels]
        return np.stack([c.mean(axis=-1) for c in clips])

    all1 = stack_all(mels1)
    all2 = stack_all(mels2)
    if all1 is None or all2 is None:
        print("  [4] skipped — no mel data")
        return

    n_mels = all1.shape[1]
    bins = np.arange(n_mels)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 4a — corpus-level mean per bin
    ax = axes[0, 0]
    ax.plot(bins, all1.mean(0), color=DAY_COLOR["day1"], label="Day 1")
    ax.plot(bins, all2.mean(0), color=DAY_COLOR["day2"], label="Day 2")
    ax.fill_between(bins,
                    all1.mean(0) - all1.std(0),
                    all1.mean(0) + all1.std(0),
                    alpha=0.2, color=DAY_COLOR["day1"])
    ax.fill_between(bins,
                    all2.mean(0) - all2.std(0),
                    all2.mean(0) + all2.std(0),
                    alpha=0.2, color=DAY_COLOR["day2"])
    ax.set_title("Per-bin log-mel mean ± 1σ (corpus level)")
    ax.set_xlabel("Mel bin"); ax.set_ylabel("log-mel value")
    ax.legend(); ax.grid(alpha=0.3)

    # 4b — corpus-level std per bin
    ax = axes[0, 1]
    ax.plot(bins, all1.std(0), color=DAY_COLOR["day1"], label="Day 1")
    ax.plot(bins, all2.std(0), color=DAY_COLOR["day2"], label="Day 2")
    ax.set_title("Per-bin log-mel std (corpus level)")
    ax.set_xlabel("Mel bin"); ax.set_ylabel("std")
    ax.legend(); ax.grid(alpha=0.3)

    # 4c — delta (Day2 - Day1) mean per bin
    ax = axes[1, 0]
    delta_mean = all2.mean(0) - all1.mean(0)
    ax.bar(bins, delta_mean,
           color=np.where(delta_mean >= 0, DAY_COLOR["day2"], DAY_COLOR["day1"]),
           alpha=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Day2 − Day1 mean log-mel per bin\n(systematic shift = min/max scaler leakage)")
    ax.set_xlabel("Mel bin"); ax.set_ylabel("Δ log-mel")
    ax.grid(axis="y", alpha=0.3)

    # 4d — per-class mean spectra overlay
    ax = axes[1, 1]
    for cls in all_classes:
        m1 = mels1.get(cls, [])
        m2 = mels2.get(cls, [])
        if m1:
            mu1 = np.stack([c.mean(-1) for c in m1]).mean(0)
            ax.plot(bins, mu1, color=DAY_COLOR["day1"],
                    alpha=0.7, lw=1.5,
                    linestyle="-", label=f"{cls} D1" if cls == all_classes[0] else "_")
        if m2:
            mu2 = np.stack([c.mean(-1) for c in m2]).mean(0)
            ax.plot(bins, mu2, color=DAY_COLOR["day2"],
                    alpha=0.7, lw=1.5,
                    linestyle="--", label=f"{cls} D2" if cls == all_classes[0] else "_")
    ax.set_title("Per-class mean log-mel (solid=D1, dashed=D2)")
    ax.set_xlabel("Mel bin"); ax.set_ylabel("log-mel")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("Log-mel per-bin statistics — cross-day CMVN diagnostic", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "04_cmvn_stats.png", dpi=150)
    plt.close(fig)
    print("  [4] CMVN stats saved")


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 – Mean log-mel heatmaps per class per day
# ══════════════════════════════════════════════════════════════════════════════

def plot_mel_heatmaps(mels1, mels2, all_classes, out_dir):
    if not HAS_LIBROSA:
        print("  [5] skipped — librosa not available")
        return

    n_cls = len(all_classes)
    fig, axes = plt.subplots(n_cls, 2, figsize=(8, 3 * n_cls))
    if n_cls == 1:
        axes = axes[np.newaxis, :]

    # compute global vmin/vmax for consistent colorscale
    all_means = []
    for cls in all_classes:
        for mels in [mels1, mels2]:
            clips = mels.get(cls, [])
            if clips:
                all_means.append(np.stack([c.mean(-1) for c in clips]).mean(0))
    vmin = min(m.min() for m in all_means)
    vmax = max(m.max() for m in all_means)

    for i, cls in enumerate(all_classes):
        for j, (day_mels, day_lbl) in enumerate([(mels1, "Day 1"), (mels2, "Day 2")]):
            ax = axes[i, j]
            clips = day_mels.get(cls, [])
            if clips:
                mean_spec = np.stack([c.mean(-1) for c in clips]).mean(0)
                ax.imshow(mean_spec[::-1, np.newaxis], aspect="auto",
                          vmin=vmin, vmax=vmax, cmap="magma")
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(f"{cls} — {day_lbl}", fontsize=9)
            ax.set_xticks([]); ax.set_ylabel("Mel bin ↑", fontsize=7)

    fig.suptitle("Mean log-mel per class per day (same colorscale)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "05_mel_heatmaps.png", dpi=150)
    plt.close(fig)
    print("  [5] mel heatmaps saved")


# ══════════════════════════════════════════════════════════════════════════════
# Section 6 – PCA / UMAP of mean log-mel feature vectors
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_space(mels1, mels2, all_classes, out_dir, no_umap):
    if not HAS_LIBROSA:
        print("  [6] skipped — librosa not available")
        return

    # build feature matrix
    X, day_labels, class_labels = [], [], []
    for cls in all_classes:
        for clips, day in [(mels1.get(cls, []), "day1"),
                           (mels2.get(cls, []), "day2")]:
            for c in clips:
                X.append(c.mean(-1))      # [n_mels]
                day_labels.append(day)
                class_labels.append(cls)

    if len(X) < 4:
        print("  [6] skipped — too few samples")
        return

    X = np.array(X)
    X_scaled = StandardScaler().fit_transform(X)

    n_plots = 1 + (1 if (HAS_UMAP and not no_umap) else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    day_arr   = np.array(day_labels)
    class_arr = np.array(class_labels)
    class_ids = {c: i for i, c in enumerate(all_classes)}

    def scatter_ax(ax, coords, title):
        for cls in all_classes:
            for day in ["day1", "day2"]:
                mask = (class_arr == cls) & (day_arr == day)
                if mask.sum() == 0:
                    continue
                color = CLASS_CMAP(class_ids[cls])
                marker = "o" if day == "day1" else "^"
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=[color], marker=marker,
                           s=25, alpha=0.6,
                           label=f"{cls} ({'D1' if day=='day1' else 'D2'})")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    scatter_ax(axes[0], pca_coords,
               f"PCA  (PC1 {pca.explained_variance_ratio_[0]:.1%}, "
               f"PC2 {pca.explained_variance_ratio_[1]:.1%})\n"
               "circles=Day1  triangles=Day2")

    # UMAP
    if HAS_UMAP and not no_umap:
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15,
                       min_dist=0.1)
        umap_coords = reducer.fit_transform(X_scaled)
        scatter_ax(axes[1], umap_coords,
                   "UMAP\ncircles=Day1  triangles=Day2")

    fig.suptitle("Feature-space: mean log-mel per clip", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "06_feature_space.png", dpi=150)
    plt.close(fig)
    print("  [6] feature space saved")


# ══════════════════════════════════════════════════════════════════════════════
# Section 7 – Linear-frequency subband energy (B3 EQ / shelf tilt)
# ══════════════════════════════════════════════════════════════════════════════

def plot_subband_ratios(rows1, rows2, all_classes, out_dir):
    """
    Grouped bars: per class, per subband, mean energy fraction — Day 1 vs Day 2.
    """
    band_keys = sorted(
        {k for r in rows1 + rows2 for k in r if k.startswith("sb_")}
    )
    if not band_keys:
        print("  [7] skipped — no subband features (clips too short?)")
        return

    bar_w = 0.16
    gap = 0.04
    n_b = len(band_keys)
    cluster_w = n_b * (2 * bar_w + gap) - gap
    between = 0.45
    fig, ax = plt.subplots(figsize=(max(11, len(all_classes) * (cluster_w + between)), 5.2))
    x0 = 0.0
    tick_pos = []
    for cls in all_classes:
        tick_pos.append(x0 + cluster_w / 2)
        for bi, bk in enumerate(band_keys):
            vx = x0 + bi * (2 * bar_w + gap)
            vals1 = [
                float(r[bk])
                for r in rows1
                if r["label"] == cls and bk in r and not np.isnan(float(r[bk]))
            ]
            vals2 = [
                float(r[bk])
                for r in rows2
                if r["label"] == cls and bk in r and not np.isnan(float(r[bk]))
            ]
            m1 = float(np.mean(vals1)) if vals1 else 0.0
            m2 = float(np.mean(vals2)) if vals2 else 0.0
            ax.bar(vx, m1, bar_w, color=DAY_COLOR["day1"], alpha=0.9)
            ax.bar(vx + bar_w, m2, bar_w, color=DAY_COLOR["day2"], alpha=0.9)
        x0 += cluster_w + between

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(all_classes, rotation=15)
    ax.set_ylabel("Fraction of total rFFT energy")
    ax.set_ylim(0, 1.02)
    band_lbl = ", ".join(bk.replace("sb_", "").replace("_", "–") + " Hz" for bk in band_keys)
    ax.set_title("Linear-frequency subband ratios — EQ / shelf tilt (B3)\n" + band_lbl, fontsize=9)
    ax.legend(
        handles=[
            Patch(facecolor=DAY_COLOR["day1"], edgecolor="none", label="Day 1"),
            Patch(facecolor=DAY_COLOR["day2"], edgecolor="none", label="Day 2"),
        ],
        loc="upper right",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "07_subband_ratios.png", dpi=150)
    plt.close(fig)
    print("  [7] subband ratios saved")


# ══════════════════════════════════════════════════════════════════════════════
# Section 8 – Summary stats table (printed + saved as text)
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(rows1, rows2, mels1, mels2, all_classes, out_dir):
    lines = []
    lines.append("=" * 68)
    lines.append("CROSS-DAY SHIFT SUMMARY")
    lines.append("=" * 68)

    # class counts
    lines.append("\n── Class balance ─────────────────────────────────────────────")
    for cls in all_classes:
        n1 = len([r for r in rows1 if r["label"] == cls])
        n2 = len([r for r in rows2 if r["label"] == cls])
        lines.append(f"  {cls:20s}  Day1: {n1:4d}   Day2: {n2:4d}")

    # signal stats
    for key, lbl in [("rms_db", "RMS (dBFS)"), ("dr_db", "Dyn range (dB)")]:
        v1 = [r[key] for r in rows1 if key in r]
        v2 = [r[key] for r in rows2 if key in r]
        if v1 and v2:
            lines.append(f"\n── {lbl} ─────────────────────────────────────────")
            lines.append(f"  Day1  mean={np.mean(v1):+6.2f}  std={np.std(v1):.2f}  "
                         f"median={np.median(v1):+6.2f}")
            lines.append(f"  Day2  mean={np.mean(v2):+6.2f}  std={np.std(v2):.2f}  "
                         f"median={np.median(v2):+6.2f}")
            lines.append(f"  Δ mean = {np.mean(v2)-np.mean(v1):+.2f} dB")

    # CMVN shift
    if HAS_LIBROSA and mels1 and mels2:
        def corpus_mean(md):
            clips = [c for v in md.values() for c in v]
            if not clips: return None
            return np.stack([c.mean(-1) for c in clips]).mean(0)
        mu1, mu2 = corpus_mean(mels1), corpus_mean(mels2)
        if mu1 is not None and mu2 is not None:
            delta = mu2 - mu1
            lines.append("\n── Log-mel per-bin shift (Day2 − Day1) ───────────────────────")
            lines.append(f"  max |Δ|  = {np.abs(delta).max():.3f}  at bin {np.abs(delta).argmax()}")
            lines.append(f"  mean |Δ| = {np.abs(delta).mean():.3f}")
            lines.append(f"  Bins with |Δ| > 1.0: {(np.abs(delta)>1.0).sum()}")
            lines.append(f"  → {'SEVERE' if np.abs(delta).mean()>0.5 else 'MODERATE' if np.abs(delta).mean()>0.2 else 'MILD'} "
                         f"systematic log-mel shift between days")

    # Temporal envelope / clip (2c)
    cr1 = [r["clip_rate"] for r in rows1 if "clip_rate" in r]
    cr2 = [r["clip_rate"] for r in rows2 if "clip_rate" in r]
    if cr1 and cr2:
        lines.append("\n── Clip rate (all classes, clip mean) ─────────────────────────")
        lines.append(f"  Day1 mean clip_rate: {np.mean(cr1):.4f}   Day2: {np.mean(cr2):.4f}")
        lines.append(f"  Δ (D2−D1): {np.mean(cr2)-np.mean(cr1):+.4f}  (high → possible gain / limiting)")

    pr1 = [r["peak_rel"] for r in rows1 if "peak_rel" in r and not np.isnan(r["peak_rel"])]
    pr2 = [r["peak_rel"] for r in rows2 if "peak_rel" in r and not np.isnan(r["peak_rel"])]
    if pr1 and pr2:
        lines.append("\n── Envelope peak position (0=front, 1=tail) ─────────────────")
        lines.append(f"  Day1 mean peak_rel: {np.mean(pr1):.3f}   Day2: {np.mean(pr2):.3f}")

    # Background floor (3b)
    bg1 = [r for r in rows1 if str(r["label"]).lower() == "background"]
    bg2 = [r for r in rows2 if str(r["label"]).lower() == "background"]
    if bg1 or bg2:
        lines.append("\n── Background class (floor diagnostic) ───────────────────────")
        lines.append(f"  Counts  Day1: {len(bg1):4d}   Day2: {len(bg2):4d}")
        if bg1 and bg2 and HAS_LIBROSA:
            rms_b1 = [r["rms_db"] for r in bg1]
            rms_b2 = [r["rms_db"] for r in bg2]
            fl1 = [r["flatness"] for r in bg1 if "flatness" in r]
            fl2 = [r["flatness"] for r in bg2 if "flatness" in r]
            lines.append(f"  RMS (dBFS)  D1 mean={np.mean(rms_b1):+.2f}  D2 mean={np.mean(rms_b2):+.2f}")
            if fl1 and fl2:
                lines.append(f"  Flatness    D1 mean={np.mean(fl1):.4f}  D2 mean={np.mean(fl2):.4f}")

    # Subband corpus shift (7)
    band_keys = sorted({k for r in rows1 + rows2 for k in r if k.startswith("sb_")})
    if band_keys:
        lines.append("\n── Linear subband energy (corpus mean fraction) ──────────────")
        for bk in band_keys:
            v1 = [float(r[bk]) for r in rows1 if bk in r and not np.isnan(float(r[bk]))]
            v2 = [float(r[bk]) for r in rows2 if bk in r and not np.isnan(float(r[bk]))]
            if v1 and v2:
                lines.append(
                    f"  {bk}:  D1={np.mean(v1):.3f}  D2={np.mean(v2):.3f}  Δ={np.mean(v2)-np.mean(v1):+.3f}"
                )

    lines.append("\n" + "=" * 68)
    text = "\n".join(lines)
    print(text)
    (out_dir / "00_summary.txt").write_text(text)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def _resolve_day_index_sources(args, day_prefix):
    """Return list of index file paths for day1 or day2. Exactly one mode must be set."""
    split_dir = getattr(args, f"{day_prefix}_split_dir", None)
    indices = getattr(args, f"{day_prefix}_indices", None)
    single = getattr(args, f"{day_prefix}_index", None)
    modes = sum(
        1
        for v in (split_dir, indices, single)
        if v is not None and (not isinstance(v, list) or len(v) > 0)
    )
    if modes == 0:
        raise ValueError(
            f"Specify exactly one of --{day_prefix}_split_dir, --{day_prefix}_indices, "
            f"or --{day_prefix}_index"
        )
    if modes > 1:
        raise ValueError(
            f"Use only one of --{day_prefix}_split_dir, --{day_prefix}_indices, "
            f"--{day_prefix}_index (not multiple)"
        )
    if split_dir:
        return indices_from_split_dir(split_dir)
    if indices:
        return list(indices)
    return [single]


def run_crossday_analysis(
    day1_index_paths,
    day2_index_paths,
    max_per_class=120,
    sr=16000,
    n_mels=64,
    mic_idx=0,
    out_dir="./crossday_analysis",
    no_umap=False,
    seed=42,
):
    """
    Programmatic entry point (CLI and web UI).
    day1_index_paths / day2_index_paths: list of index file paths (merged in order, deduped).
    Returns output directory as Path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx1_sources = [str(Path(p)) for p in day1_index_paths]
    idx2_sources = [str(Path(p)) for p in day2_index_paths]

    print(f"\n{'='*60}")
    print("Cross-day shift analysis")
    print(f"  Day 1 index files ({len(idx1_sources)}):")
    for p in idx1_sources:
        print(f"    {p}")
    print(f"  Day 2 index files ({len(idx2_sources)}):")
    for p in idx2_sources:
        print(f"    {p}")
    print(f"  max/class   : {max_per_class}")
    print(f"  mic channel : {mic_idx}")
    print(f"  output      : {out_dir}")
    print(f"{'='*60}\n")

    print("Loading Day 1 (merged) ...")
    idx1 = load_merged_indices(idx1_sources)
    print(f"  merged unique paths: {len(idx1)}")
    d1 = collect_dataset(idx1, max_per_class, mic_idx, seed)
    print(f"  classes: { {k: len(v) for k, v in d1.items()} }")

    print("Loading Day 2 (merged) ...")
    idx2 = load_merged_indices(idx2_sources)
    print(f"  merged unique paths: {len(idx2)}")
    d2 = collect_dataset(idx2, max_per_class, mic_idx, seed)
    print(f"  classes: { {k: len(v) for k, v in d2.items()} }")

    print("\nExtracting features ...")
    rows1, mels1 = build_feature_table(d1, sr, n_mels)
    rows2, mels2 = build_feature_table(d2, sr, n_mels)

    all_classes = sorted(set(list(d1.keys()) + list(d2.keys())))

    print("\nGenerating plots ...")
    plot_class_balance(d1, d2, out_dir)
    plot_signal_stats(rows1, rows2, out_dir)
    plot_temporal_envelope(rows1, rows2, out_dir)
    plot_spectral_stats(rows1, rows2, out_dir)
    plot_background_floor(rows1, rows2, out_dir)
    plot_spectral_flux(rows1, rows2, out_dir)
    plot_cmvn_stats(mels1, mels2, all_classes, out_dir)
    plot_mel_heatmaps(mels1, mels2, all_classes, out_dir)
    plot_feature_space(mels1, mels2, all_classes, out_dir, no_umap)
    plot_subband_ratios(rows1, rows2, all_classes, out_dir)
    print_summary(rows1, rows2, mels1, mels2, all_classes, out_dir)

    print(f"\nDone. All outputs in: {out_dir}/")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Cross-day domain shift analysis")
    parser.add_argument(
        "--day1_split_dir",
        default=None,
        help="Directory with train_index.txt, val_index.txt, test_index.txt for day 1",
    )
    parser.add_argument(
        "--day2_split_dir",
        default=None,
        help="Directory with train_index.txt, val_index.txt, test_index.txt for day 2",
    )
    parser.add_argument(
        "--day1_indices",
        nargs="+",
        default=None,
        help="One or more index files for day 1 (merged, deduped)",
    )
    parser.add_argument(
        "--day2_indices",
        nargs="+",
        default=None,
        help="One or more index files for day 2 (merged, deduped)",
    )
    parser.add_argument(
        "--day1_index",
        default=None,
        help="Single index file for day 1 (legacy; test-only if one file)",
    )
    parser.add_argument(
        "--day2_index",
        default=None,
        help="Single index file for day 2 (legacy)",
    )
    parser.add_argument("--max_per_class", type=int, default=120)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument(
        "--mic_idx",
        type=int,
        default=0,
        help="Which mic channel to analyse (0-9)",
    )
    parser.add_argument("--out_dir", default="./crossday_analysis")
    parser.add_argument("--no_umap", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        day1_sources = _resolve_day_index_sources(args, "day1")
        day2_sources = _resolve_day_index_sources(args, "day2")
    except ValueError as e:
        parser.error(str(e))

    run_crossday_analysis(
        day1_sources,
        day2_sources,
        max_per_class=args.max_per_class,
        sr=args.sr,
        n_mels=args.n_mels,
        mic_idx=args.mic_idx,
        out_dir=args.out_dir,
        no_umap=args.no_umap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()