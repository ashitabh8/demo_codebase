# Remove Early Exit Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all early exit functionality from the codebase while keeping all model types (ResNet, ConvOnly, ResNetSimple) functional as standard classifiers.

**Architecture:** Delete early exit classes, parameters, and branching code from models, loss, training, testing, and config files. Models will return `{'logits': ..., 'features': ...}` without an `'exits'` key. ConvOnly remains but without exit branches.

**Tech Stack:** PyTorch, Python, YAML

---

## Task 1: `src2/models/ResNet.py` — Remove EarlyExitBranch and exit logic

**Files:**
- Modify: `src2/models/ResNet.py`

- [ ] Delete the `EarlyExitBranch` class (lines 717–742)
- [ ] In `ConfigurableResNet.__init__`: remove `early_exit_layers` param, `self.early_exit_layers`, and the `exit_branches` ModuleDict block (lines 797, 836–841)
- [ ] Remove the exits collection loop from `ConfigurableResNet.forward` (lines 913–917); keep final head; remove `'exits'` key from returned dict
- [ ] Update `ConfigurableResNet` docstring: remove `early_exit_layers` from Args and Example
- [ ] In `SingleModalResNet.__init__`: remove `early_exit_layers` param and its pass-through to `ConfigurableResNet` (lines 977–978, 995)
- [ ] Update `SingleModalResNet` docstring: remove early exit references
- [ ] Verify: `forward()` now returns `{'logits': logits, 'features': features}` only

---

## Task 2: `src2/models/ConvOnlyModels.py` — Remove exit branches

**Files:**
- Modify: `src2/models/ConvOnlyModels.py`

- [ ] Remove `from models.ResNet import EarlyExitBranch` import (line 40)
- [ ] In `ConvOnlyNet.__init__`: remove `early_exit_layers`, `early_exit_type` params; remove `self.early_exit_layers`, `self.early_exit_type`, and `exit_branches` ModuleDict block (lines 113–114, 135–136, 175–189)
- [ ] Remove exits collection loop from `ConvOnlyNet.forward` (lines 242–246); remove `'exits'` key from returned dict
- [ ] In `SingleModalConvOnly.__init__`: remove `early_exit_layers`, `early_exit_type` params and their pass-through (lines 310–311, 329–330)
- [ ] Update module docstring and class docstrings: remove early exit mentions, remove early exit example usages in `__main__` block (lines 356–428 that reference exits)
- [ ] Verify: `forward()` returns `{'logits': logits, 'features': features}` only

---

## Task 3: `src2/models/create_models.py` — Remove early exit factory logic and functions

**Files:**
- Modify: `src2/models/create_models.py`

- [ ] In `create_single_modal_model`: remove `early_exit_layers` extraction (lines 86–97); remove `early_exit_layers` from `SingleModalResNet(...)` call; for ConvOnly: remove `kernel_sizes`, `strides` optional gets, `early_exit_type`, `early_exit_layers` from `SingleModalConvOnly(...)` call — wait, keep `kernel_sizes` and `strides` (those are not early exit); remove only `early_exit_layers` and `early_exit_type`
- [ ] Remove `get_early_exit_memory()` function entirely (lines 311–549)
- [ ] In `log_memory_info()`: remove the early exit table branch; keep only the "no early exits" final branch (simplify to always show final model memory)
- [ ] In `get_model_config()`: remove `early_exits` from result dict, remove early exit fields from convonly section
- [ ] Update `create_single_modal_model` docstring to remove early_exits default mention

---

## Task 4: `src2/train_test/loss.py` — Remove CrossEntropyLossWithEarlyExits

**Files:**
- Modify: `src2/train_test/loss.py`

- [ ] Delete `CrossEntropyLossWithEarlyExits` class (lines 61–130)
- [ ] In `get_loss_function`: remove `has_early_exits` parameter and the `if has_early_exits:` branch; always return `CrossEntropyLossForDictOutput()` for `cross_entropy` loss name
- [ ] Update `get_loss_function` docstring to remove early exit references

---

## Task 5: `src2/train_test/train_test_utils.py` — Remove early exit training/validation functions

**Files:**
- Modify: `src2/train_test/train_test_utils.py`

- [ ] Delete `validate_with_early_exits()` function (lines 470–601)
- [ ] Delete `log_early_exits_to_tensorboard()` function (lines 604–773 approximately, including commented-out legacy version)
- [ ] Delete `train_with_early_exits()` function (lines 1156–1401 approximately)
- [ ] Verify remaining exports: `setup_experiment_dir`, `train`, `setup_optimizer`, `setup_scheduler`, `load_checkpoint`, `validate` are intact

---

## Task 6: `src2/train_test/train.py` — Remove early exit detection and routing

**Files:**
- Modify: `src2/train_test/train.py`

- [ ] Remove `train_with_early_exits` from import (line 29)
- [ ] Remove `has_early_exits` detection (lines 217–219)
- [ ] Remove the `if has_early_exits:` branch that calls `train_with_early_exits` (lines 234–250); keep only the `else` branch calling `train(...)`, remove the else
- [ ] Simplify loss setup: remove `has_early_exits=has_early_exits` from `get_loss_function` call (line 222)

---

## Task 7: `src2/train_test/test.py` — Remove early exit test branches

**Files:**
- Modify: `src2/train_test/test.py`

- [ ] Remove `validate_with_early_exits`, `log_early_exits_to_tensorboard` from import (lines 49–51)
- [ ] Remove section 10 "Detect Early Exits" block (lines 181–189)
- [ ] Remove section 12b memory calculation block (lines 211–237) — the `if has_early_exits:` block
- [ ] Simplify section 13 loss setup: remove `has_early_exits=has_early_exits` from `get_loss_function` call
- [ ] Replace section 14 testing: remove the `if has_early_exits:` branch entirely; keep only the standard model testing block (lines 304–378)
- [ ] Update section 16 save results: remove early exit conditional; always use standard results format
- [ ] Update section 17 final summary: remove early exit TensorBoard reference
- [ ] Update module docstring to remove early exit references; update the usage example path
- [ ] Remove `from torch.utils.tensorboard import SummaryWriter` if no longer needed (check — it's only used in the early exit branch)

---

## Task 8: `src2/train_test/analyze_memory.py` — Switch to get_total_memory

**Files:**
- Modify: `src2/train_test/analyze_memory.py`

- [ ] Update imports: replace `get_early_exit_memory` with `get_total_memory`; keep `get_input_memory`, `log_memory_info`
- [ ] In `analyze_model`: replace `get_early_exit_memory(model, input_dict, unit=unit)` call with `get_total_memory(model, input_dict, unit=unit)` — note: `get_total_memory` returns `{'parameter_memory', 'activation_memory', 'total_memory', 'unit'}`, not the same shape as `get_early_exit_memory`. Rewrite to print the values directly instead of calling `log_memory_info` (which expects the old shape).

---

## Task 9: `src2/data/ACIDS.yaml` — Remove early exit config fields

**Files:**
- Modify: `src2/data/ACIDS.yaml`

- [ ] In `student_audio_convonly` model definition: remove `early_exits: [1, 2]` and `early_exit_type: "gap_linear"` fields
- [ ] In `teacher_audio_resnet18` model definition: remove `early_exits: []` field
- [ ] In `student_audio_resnet` model definition: remove `early_exits: []` field
- [ ] Remove entire `only_audio_convonly_early_exit` experiment block (lines 197–220)
- [ ] Remove entire `only_audio_resnet_early_exit` experiment block (lines 225–248)
- [ ] Remove `exit_weights` from any remaining experiment stages if present
