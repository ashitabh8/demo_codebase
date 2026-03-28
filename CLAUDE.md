# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Coding Suggestions
1. Do not use .get() while retrieving values from a dict. Use [] instead.
2. Try not add "defaults" inside the code we would like all values to be 
explicitly set in the YAML by the user.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate cenv
```

Key dependencies: PyTorch 2.6.0, TensorFlow 2.20.0, CUDA 12.9, scikit-learn, tensorboard.

## Common Commands

**Training:**
```bash
cd src2/train_test
python train.py -experiment_name <name> -yaml_path ../data/ACIDS.yaml -gpu 0
# Use -gpu -1 for CPU
```

**Testing a trained model:**
```bash
python test.py --experiment_dir ../experiments/<experiment_dir> --gpu 0
# Use --checkpoint_path to specify a non-default checkpoint
```

**TensorBoard monitoring:**
```bash
tensorboard --logdir=src2/experiments --port=6006
```

**Quick debugging:**
```bash
python main.py --model ResNet --model_variant resnet18 --yaml_path ../data/ACIDS.yaml --gpu 0
```

## Architecture Overview

This is a **model distillation / compression pipeline** for multi-modal time-series classification (audio + seismic sensor data). The primary dataset is ACIDS (vehicle classification).

### Configuration-Driven Design

Everything flows from a single YAML file (e.g., `src2/data/ACIDS.yaml`), which has three sections:
1. **Dataset config** — file paths, batch size, modality names, class counts
2. **Model zoo** — named model definitions (teacher/student ResNets with layer sizes, quantization settings, early exit positions)
3. **Distillation experiments** — named experiments referencing models and defining training stages, optimizer, and LR schedule

The `-experiment_name` CLI arg selects which experiment in the `distillation:` section of the YAML to run.

### Data Flow

```
YAML config → train.py → MultiModalDataLoader → Augmenter → model forward pass → loss → optimizer
```

- `MultiModalDataLoader` (`src2/dataset_utils/`) lazy-loads sensor data from disk. Input tensors are structured as dicts: `{location: {modality: tensor}}`.
- `Augmenter` (`src2/data_augmenter/`) applies 20+ augmentation techniques (time/frequency domain).
- Models return a dict: `{'logits': tensor, 'exits': [tensor, ...], 'features': tensor}`.

### Model Architecture

- **`src2/models/create_models.py`** — factory function `create_single_modal_model()` dispatches to model classes based on `model_type` in YAML.
- **`ResNetSimple.py`** — primary model class; configurable depth, filter sizes, stem, dropout, early exits, and weight-only quantization.
- **`QuantModules.py` / `WeightOnlyQuant.py`** — quantization-aware modules (2-bit, 4-bit weight quantization).

### Training Logic

`src2/train_test/train_test_utils.py` (750+ lines) contains the core training loop. It dispatches to `train()` or `train_with_early_exits()` depending on whether the model has intermediate exit heads. Early exit models use a weighted sum of per-exit losses (`exit_weights` in YAML).

Outputs are saved to `src2/experiments/<timestamp>_<experiment_name>/`:
- `models/best_model.pth`, `models/last_model.pth`
- `logs/train.log`
- TensorBoard event files

### Key Abstractions

| Concern | Location |
|---|---|
| YAML parsing & CLI args | `src2/dataset_utils/parse_args_utils.py` |
| Data loading | `src2/dataset_utils/MultiModalDataLoader.py` |
| Augmentation | `src2/data_augmenter/Augmenter.py` + subclasses |
| Model creation | `src2/models/create_models.py` |
| Loss functions | `src2/train_test/loss.py` |
| Training/eval loops | `src2/train_test/train_test_utils.py` |
| Experiment orchestration | `src2/train_test/train.py` |
