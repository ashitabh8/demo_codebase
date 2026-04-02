# Data Export Plan

## Scope
Prepare a deterministic Parkland demo dataset containing mel-preprocessed audio-only features for classes Polaris, Warhog, and Truck, plus labels/metadata for firmware and UI consumers.

## Owner
Person A (support: Person B for label/format review)

## Dependencies
- `src2/data/Parkland.yaml`
- Existing dataloader pipeline in `src2/dataset_utils`
- Existing preprocessing pipeline in `src2/data_augmenter`

## Implementation Tasks
1. Add exporter script `src2/gen_code/export_demo_samples.py`.
2. Load config from `Parkland.yaml`, use a Parkland experiment with `preprocess_mode: mel`.
3. Filter to target classes (`Polaris`, `Warhog`, `Truck`) and remap labels to `0..2`.
4. Use val/test split to build a balanced sample set with deterministic seed.
5. Export:
   - `src2/gen_code/demo_data/demo_samples.csv` (sample_id, flattened features, target)
   - `src2/gen_code/demo_data/demo_labels.csv` (sample_id,target,class_name)
   - `src2/gen_code/demo_data/demo_metadata.json` (shape/class map/config)
6. Add optional header export (`demo_samples.h`) for direct firmware embedding.
7. Add class-balance and shape checks before writing outputs.

## Validation Commands
- `python src2/gen_code/export_demo_samples.py --help`
- `python src2/gen_code/export_demo_samples.py --yaml_path src2/data/Parkland.yaml --output_dir src2/gen_code/demo_data --num_samples 90 --split test`
- `python src2/gen_code/validate_demo_export.py --data_dir src2/gen_code/demo_data`

## Exit Criteria
- Output files exist and are parseable.
- Class counts are balanced (or logged if insufficient source samples).
- Labels are remapped only to `0,1,2`.
- Metadata clearly documents feature flatten order and source config.

## Handoff Artifacts
- Export script and validator script.
- Generated demo data files for firmware/UI integration.
