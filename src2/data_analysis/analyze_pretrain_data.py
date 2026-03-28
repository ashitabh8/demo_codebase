import torch
from pathlib import Path
from collections import defaultdict

index_path = "/data/kara4/Parkland/time_data_partition/pretrain_index.txt"

# Parse all paths, extract vehicle name from filename (part before _rs)
vehicle_to_label = {}  # vehicle_name -> set of labels seen
label_to_vehicles = defaultdict(set)
total_samples = 0
load_errors = 0

with open(index_path) as f:
    paths = [line.strip() for line in f if line.strip()]

total_samples = len(paths)

# Extract vehicle names from filenames
all_vehicle_names = set()
for p in paths:
    stem = Path(p).stem  # e.g. tesla_rs1_210
    # vehicle name is everything before the last _rs<digit>
    parts = stem.rsplit("_rs", 1)
    vehicle = parts[0]
    all_vehicle_names.add(vehicle)

# Sample one file per unique vehicle name to get label mapping
# Build a map: vehicle_name -> one representative path
vehicle_sample_path = {}
for p in paths:
    stem = Path(p).stem
    parts = stem.rsplit("_rs", 1)
    vehicle = parts[0]
    if vehicle not in vehicle_sample_path:
        vehicle_sample_path[vehicle] = p

# Load one sample per vehicle to get its label
for vehicle, path in sorted(vehicle_sample_path.items()):
    try:
        sample = torch.load(path, weights_only=False)
        label = int(sample["label"])
        vehicle_to_label[vehicle] = label
        label_to_vehicles[label].add(vehicle)
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        load_errors += 1

print(f"Total samples in pretrain_index.txt: {total_samples}")
print(f"Unique vehicle types: {len(all_vehicle_names)}")
print()
print("Vehicle name -> Label:")
for vehicle, label in sorted(vehicle_to_label.items(), key=lambda x: x[1]):
    print(f"  label {label:2d}  {vehicle}")
print()
print("Label -> Vehicle names:")
for label in sorted(label_to_vehicles):
    print(f"  label {label:2d} -> {sorted(label_to_vehicles[label])}")
if load_errors:
    print(f"\n{load_errors} files failed to load.")
