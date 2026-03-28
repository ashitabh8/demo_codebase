import os
import shutil

INDEX_DIR = "/data/misra8/GracesQuarters/index_files/2024-08-06-GQ-split-multiclass"
OLD_PREFIX = "/home/tkimura4/data/datasets/MOD/GracesQuarters/2024-08-06-GQ/individual_time_samples"
NEW_PREFIX = "/data/misra8/GracesQuarters/data/2024-08-06-GQ/individual_time_samples"

index_files = [f for f in os.listdir(INDEX_DIR) if f.endswith(".txt") and not f.endswith("_original.txt")]

for filename in index_files:
    filepath = os.path.join(INDEX_DIR, filename)
    original_path = os.path.join(INDEX_DIR, filename.replace(".txt", "_original.txt"))

    shutil.copy2(filepath, original_path)
    print(f"Saved original: {original_path}")

    with open(filepath, "r") as f:
        lines = f.readlines()

    updated_lines = [line.replace(OLD_PREFIX, NEW_PREFIX) for line in lines]

    replaced_count = sum(1 for old, new in zip(lines, updated_lines) if old != new)

    with open(filepath, "w") as f:
        f.writelines(updated_lines)

    print(f"Updated {filepath}  ({replaced_count}/{len(lines)} lines replaced)")

print("\nDone.")
