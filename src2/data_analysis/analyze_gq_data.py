import torch
from pathlib import Path
from collections import defaultdict

index_path = "/home/tkimura4/data/indices/single/2024-08-06-GQ-split-multiclass/train_index.txt"

with open(index_path) as f:
    paths = [line.strip() for line in f if line.strip()]

total_samples = len(paths)
print(f"Total samples in train_index.txt: {total_samples}")

# Filename pattern: run{run_id}_gq-{sensor_id}_{time_segment}.pt
# Extract run_id from stem to use as grouping key (one sample per run)
run_sample_path = {}
for p in paths:
    stem = Path(p).stem  # e.g. run2_gq-2_55
    run_id = stem.split("_")[0]  # e.g. run2
    if run_id not in run_sample_path:
        run_sample_path[run_id] = p

print(f"Unique run IDs: {len(run_sample_path)}")
print()

# Load one sample to inspect full structure
first_path = paths[0]
print(f"Inspecting sample: {Path(first_path).name}")
sample = torch.load(first_path, weights_only=False)

def print_structure(obj, indent=0):
    prefix = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}: dict with keys {list(v.keys())}")
                print_structure(v, indent + 1)
            elif hasattr(v, "shape"):
                print(f"{prefix}{k}: tensor shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"{prefix}{k}: {type(v).__name__} = {v}")
    elif hasattr(obj, "shape"):
        print(f"{prefix}tensor shape={tuple(obj.shape)}, dtype={obj.dtype}")
    else:
        print(f"{prefix}{type(obj).__name__} = {obj}")

print_structure(sample)
print()

# Collect label stats across one sample per run
label_to_runs = defaultdict(list)
load_errors = 0

for run_id, path in sorted(run_sample_path.items()):
    try:
        s = torch.load(path, weights_only=False)
        label = int(s["label"])
        label_to_runs[label].append(run_id)
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        load_errors += 1

print("Label -> Run IDs:")
for label in sorted(label_to_runs):
    runs = sorted(label_to_runs[label])
    print(f"  label {label:2d} ({len(runs):3d} runs): {runs[:5]}{'...' if len(runs) > 5 else ''}")

print()
print("Label distribution across all samples:")
label_to_count = defaultdict(int)
for p in paths:
    stem = Path(p).stem
    run_id = stem.split("_")[0]
    # Use cached label from run_sample_path lookup
    # (all segments of a run share the same label)
    if run_id in run_sample_path:
        try:
            label = None
            for lbl, runs in label_to_runs.items():
                if run_id in runs:
                    label = lbl
                    break
            if label is not None:
                label_to_count[label] += 1
        except Exception:
            pass

for label in sorted(label_to_count):
    print(f"  label {label:2d}: {label_to_count[label]:4d} samples")

if load_errors:
    print(f"\n{load_errors} files failed to load.")
