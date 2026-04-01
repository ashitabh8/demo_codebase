"""
Dataloader Inspection Script
=============================
This script loads ONE batch from the training dataloader using the exact same
setup as finetune.py, then prints out everything in plain English so you can
verify the data is flowing through correctly.

Run from src2/:
    python inspect_dataloader.py

No GPU needed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import numpy as np
from collections import Counter

# ── 1. Load the same YAML the training scripts use ──────────────────────────
YAML_PATH = os.path.join(os.path.dirname(__file__), "data", "Parkland.yaml")
EXPERIMENT_NAME = "finetune_audio_deepsense_dw_large_mel_supcon_difflr"

with open(YAML_PATH) as f:
    config = yaml.safe_load(f)

config["experiment_name"] = EXPERIMENT_NAME
exp_cfg = config["experiments"][EXPERIMENT_NAME]
config["task_name"] = exp_cfg["task_name"]         # "fine_tune_vehicle_classification"
config["dataloader"] = exp_cfg["dataloader"]       # "parkland_single_label"

task_cfg = config[config["task_name"]]
CLASS_NAMES = task_cfg["class_names"]              # ["polaris", "warthog", "truck", "background"]
NUM_CLASSES = task_cfg["num_classes"]              # 4

print("=" * 70)
print("STEP 1 — What classes are we predicting?")
print("=" * 70)
print()
print(f"  Number of classes : {NUM_CLASSES}")
print(f"  Class names       : {CLASS_NAMES}")
print()
print("  In plain English:")
print("  Each audio sample is assigned ONE label from the list above.")
print("  The model will output 4 numbers (one per class); whichever is")
print("  highest determines the predicted class.")
print()

# ── 2. Build the dataloader (exactly as finetune.py does) ───────────────────
from dataset_utils.dataloader_factory import create_dataloaders

# dataloader_factory needs a few top-level keys
config.setdefault("num_workers", 0)   # set to 0 for simpler debugging
orig_workers = config["num_workers"]
config["num_workers"] = 0             # avoid forked processes in a script

train_loader, val_loader, test_loader = create_dataloaders(config)

config["num_workers"] = orig_workers  # restore

print("=" * 70)
print("STEP 2 — How many samples are in each split?")
print("=" * 70)
print()
print(f"  Training samples   : {len(train_loader.dataset)}")
print(f"  Validation samples : {len(val_loader.dataset)}")
print(f"  Test samples       : {len(test_loader.dataset)}")
print()
print("  (Multi-label samples like 'polaris + warthog' are DROPPED because")
print("   we are doing single-label classification.)")
print()

# ── 3. Count class distribution ──────────────────────────────────────────────
print("=" * 70)
print("STEP 3 — Class distribution (how many samples per class?)")
print("=" * 70)
print()

for split_name, loader in [("Train", train_loader), ("Val", val_loader)]:
    counter = Counter()
    for _, labels, _ in loader:
        for lbl in labels.tolist():
            counter[lbl] += 1
    total = sum(counter.values())
    print(f"  {split_name} set ({total} samples):")
    for idx, name in enumerate(CLASS_NAMES):
        count = counter.get(idx, 0)
        bar = "█" * int(30 * count / max(list(counter.values()) + [1]))
        print(f"    [{idx}] {name:<12} {count:>5} samples  {bar}")
    print()

print("  IMPORTANT: If 'background' shows 0 samples, it means the index")
print("  files for this dataset split contain NO background (vehicle-absent)")
print("  samples at all. The background class is defined but never used.")
print("  That is why the confusion matrix shows an empty background row —")
print("  there is simply no data for it in this split.")
print()

# ── 4. Print one batch in detail ─────────────────────────────────────────────
print("=" * 70)
print("STEP 4 — Inspect one real training batch")
print("=" * 70)
print()

batch = next(iter(train_loader))
data_dict, labels, indices = batch

print("  A 'batch' is a group of samples processed together.")
print(f"  Batch size = {labels.shape[0]} samples\n")

print("  DATA (the audio tensors fed to the model):")
for location, modality_dict in data_dict.items():
    for modality, tensor in modality_dict.items():
        print(f"    Location='{location}', Modality='{modality}'")
        print(f"      Shape : {list(tensor.shape)}")
        print(f"             = [batch_size={tensor.shape[0]}, "
              f"channels={tensor.shape[1]}, "
              f"time_frames={tensor.shape[2]}, "
              f"freq_bins={tensor.shape[3]}]")
        print(f"      dtype : {tensor.dtype}")
        print(f"      range : [{tensor.min():.3f}, {tensor.max():.3f}]")
        print()
        print("    In plain English:")
        print("    Each sample is a mel-spectrogram — a 2D picture of sound.")
        print(f"    The {tensor.shape[3]} rows are frequency bands (low→high pitch).")
        print(f"    The {tensor.shape[2]} columns are time windows.")
        print(f"    Think of it like a heatmap: bright = loud at that freq/time.")
        print()

print("  LABELS (the class index assigned to each sample):")
print(f"    Raw label tensor : {labels.tolist()}")
print(f"    dtype            : {labels.dtype}  (long = integer)")
print()
print("    Mapping back to class names:")
for i, (lbl_idx, sample_idx) in enumerate(zip(labels.tolist(), indices.tolist())):
    class_name = CLASS_NAMES[lbl_idx]
    print(f"      Sample {i:>2} (dataset index {sample_idx:>4}) → "
          f"label index {lbl_idx} = '{class_name}'")
print()

# ── 5. Spot-check: load one sample directly from disk ────────────────────────
print("=" * 70)
print("STEP 5 — Spot-check: compare raw file label vs dataloader label")
print("=" * 70)
print()
print("  We pick 3 samples, load them directly from disk, and verify the")
print("  dataloader assigned the correct class index.\n")

dataset = train_loader.dataset
for i in range(min(3, len(dataset))):
    raw_data, label_tensor, idx = dataset[i]
    sample_path = dataset.sample_files[i]
    raw_sample = torch.load(sample_path, weights_only=False)
    raw_label = raw_sample["label"]
    assigned_name = CLASS_NAMES[label_tensor.item()]
    distance = raw_sample.get("distance", "N/A")

    print(f"  Sample {i}:")
    print(f"    File path      : .../{os.path.basename(sample_path)}")
    print(f"    Raw label      : {raw_label!r}  (as stored on disk)")
    print(f"    Distance info  : {distance}")
    print(f"    Assigned index : {label_tensor.item()} → '{assigned_name}'")
    print()
    if assigned_name == "background":
        print("    ← This is a BACKGROUND sample (no vehicle label on disk)")
    else:
        print(f"    ← Model will be trained to predict '{assigned_name}' for this sample")
    print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
train_counter = Counter()
for _, labels, _ in train_loader:
    for lbl in labels.tolist():
        train_counter[lbl] += 1
bg_count = train_counter.get(CLASS_NAMES.index("background"), 0)
if bg_count == 0:
    print("  ⚠  BACKGROUND CLASS IS EMPTY in this dataset split.")
    print()
    print("  The index files in:")
    print(f"    {task_cfg['train_index_file']}")
    print("  contain only vehicle samples (polaris, warthog, truck).")
    print("  No 'far from vehicle' / background samples are present.")
    print()
    print("  This is why the confusion matrix has an empty background row —")
    print("  the model never sees background in training or validation,")
    print("  so it never predicts it.")
    print()
    print("  To fix: create a new index file that includes samples where no")
    print("  vehicle is within the detection range, and add those paths to")
    print("  train_index_file / val_index_file in Parkland.yaml.")
else:
    print(f"  Background samples in training : {bg_count}")
    print(f"  Total training samples         : {sum(train_counter.values())}")
