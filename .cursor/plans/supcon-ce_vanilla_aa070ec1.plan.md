---
name: supcon-ce vanilla
overview: Add a supervised contrastive (two-view SupCon) objective combined with CE for the new `vanilla_supervised_contrastive` training type, wire it through `train.py`, and update Parkland config so the experiment can be run and tested.
todos:
  - id: loss-ce-supcon
    content: Implement `ce_supcon` loss in `src2/train_test/loss.py` that supports (out1,out2) for training and single-out dict for test (CE only).
    status: pending
  - id: train-loop-two-views
    content: Add `train_vanilla_supervised_contrastive` (and matching validation) in `src2/train_test/train_test_utils.py` that creates two augmented views per batch and calls the new loss.
    status: pending
  - id: train-py-branch
    content: Update `src2/train_test/train.py` to handle `train_type == "vanilla_supervised_contrastive"` and call the new training function.
    status: pending
  - id: parklnd-yaml-wiring
    content: Update `src2/data/Parkland.yaml` with a new `training_configs` entry (tau=0.07, supcon_weight=1.0) and a new experiment key pointing to it.
    status: pending
  - id: test-vanilla-fallback
    content: Patch `src2/train_test/test.py` to support vanilla experiment directories (fallback when no `distillation` section exists), using the right `training_config` for `get_loss_function`.
    status: pending
isProject: false
---

## Design (what will change)

- Extend `src2/train_test/loss.py` with a new loss (e.g. `ce_supcon`) that:
  - Accepts either a single model output dict (for test-time compatibility) or a tuple `(outputs_view1, outputs_view2)` (for training).
  - Computes `CE(logits)` averaged across the two views.
  - Computes Supervised Contrastive Loss on embeddings using `outputs['features']` from both views, temperature `tau=0.07`, and contrastive weight `supcon_weight=1.0`.
- Add a new training function in `src2/train_test/train_test_utils.py` that:
  - For each batch, creates two augmented views by calling the existing augmentation function twice.
  - Runs two forward passes, then calls the new loss with `(outputs_view1, outputs_view2)`.
  - Uses averaged logits from both views for accuracy/confusion matrix.
  - Validates using the same two-view logic.
- Update `src2/train_test/train.py` to:
  - Add `elif train_type == "vanilla_supervised_contrastive":` which calls the new training function.
- Update `src2/data/Parkland.yaml` to:
  - Add a new `training_configs` entry with `type: "vanilla_supervised_contrastive"`, `loss_name: "ce_supcon"`, and `supcon_temperature: 0.07`, `supcon_weight: 1.0`.
  - Add a new experiment key (e.g. `only_audio_resnet18_supcon`) pointing at that training config.
- (For “then we will test it”) ensure `src2/train_test/test.py` can load vanilla experiment configs (it currently assumes a `distillation` section). Minimal fallback logic will be added so it can pick the trained model + training config when `distillation` is absent.

## Data flow (conceptual)

```mermaid
flowchart LR
  A[YAML config] --> B[train.py selects train_type]
  B --> C[create_augmenter + augmenter utils]
  C --> D[train_test_utils: vanilla_supervised_contrastive]
  D --> E[Two augmentations per batch -> view1, view2]
  E --> F[Two forward passes -> outputs_view1, outputs_view2]
  F --> G[loss.py: CE + SupCon on outputs['features']]
  G --> H[optimizer step + metrics]
```



## Implementation checklist (files & key hooks)

- `src2/train_test/loss.py`
  - Add `class CrossEntropyPlusSupConLoss(nn.Module)` (or similar)
  - Update `get_loss_function(training_config)` to handle `loss_name == "ce_supcon"`
  - Ensure `forward()` supports both:
    - `forward((out1, out2), labels)` for training
    - `forward(out, labels)` for test-time (return CE only, supcon term = 0)
- `src2/train_test/train_test_utils.py`
  - Add `train_vanilla_supervised_contrastive(...)`
  - Add `validate_vanilla_supervised_contrastive(...)` or inline validation
  - Two-view augmentation: call `apply_augmentation_fn(augmenter, data, labels)` twice per batch
  - Loss call: `loss_fn((outputs_view1, outputs_view2), loss_labels)`
- `src2/train_test/train.py`
  - Add `elif train_type == "vanilla_supervised_contrastive":` branch
  - Call the new training function
- `src2/data/Parkland.yaml`
  - Add training config recipe (epochs can mirror `vanilla_supervised_ce`)
  - Add new experiment entry for selection during training
- `src2/train_test/test.py`
  - Add fallback config parsing for vanilla experiments (no `distillation` key)
  - Use `get_loss_function(training_config)` rather than `stage_config` when in vanilla mode

## Success criteria

- `python train.py --experiment_name <new_key> --yaml_path ../data/Parkland.yaml` starts and runs without runtime errors.
- Training logs show both CE and SupCon contributing (at least once supcon_term > 0 when labels repeat in batch).
- `python test.py --experiment_dir <...>` can run for the resulting experiment directory (accuracy computed correctly).

