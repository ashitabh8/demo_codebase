# Dataloader configuration (fail-fast routing)

Supervised training uses `create_dataloaders(config)` from `dataset_utils.MultiModalDataLoader` (implemented in `dataloader_factory.py`).

## Required YAML

1. **Top-level** `dataloader_configs`: named blocks. Each block **must** set `type` to one of:
   - `legacy_multiclass` — return labels as stored in `.pt` (`weights_only=True` load when possible).
   - `single_label_only` — drop samples with more than one label; map string labels to class indices via task `class_names`.
   - `multilabel_distance` — per-class binary targets from task `class_names` + sample distance; requires extra keys below.

2. **Per experiment** (`experiments.<experiment_name>`): **must** set `dataloader` to a string key that exists in `dataloader_configs`. There is **no default**; missing or unknown keys raise immediately.

3. **Task block** (`config[task_name]`): must define `num_classes`, `train_index_file`, `val_index_file`, `test_index_file`, and for `single_label_only` / `multilabel_distance` a non-empty `class_names` list with `len(class_names) == num_classes`.

## `dataloader_configs` entries by type

### `legacy_multiclass`

Only the key `type: legacy_multiclass` is allowed (no other keys).

### `single_label_only`

Only `type: single_label_only` is allowed.

### `multilabel_distance`

Required keys:

- `distance_threshold_m` (float)
- `distance_key` (string, e.g. `distance`)

Optional experiment flag: `balance_background: true` on the **experiment** (not inside `dataloader_configs`) downsamples all-zero (background) training samples after building the multilabel dataset.

## Pretrain

SSL pretraining still uses `create_pretrain_dataloader` and does **not** use `dataloader_configs`.
