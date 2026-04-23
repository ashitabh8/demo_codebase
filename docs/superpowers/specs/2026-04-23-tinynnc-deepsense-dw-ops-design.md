# Tiny-NN-in-C: Op support for DeepSenseDWClean

Status: design (approved verbally 2026-04-23)
Owner: misra8
Scope: extend `Tiny-NN-in-C` so that `src2/models/DeepSenseDWClean.py` can be lowered, code-generated, compiled, and verified against PyTorch end-to-end.

## Motivation

The DeepSense depthwise-separable backbone is the next target model for the Tiny-NN-in-C compiler. Today the compiler only handles 2D convolutional stacks (Conv2d / BatchNorm2d / Linear / 4D permute). DeepSense adds:

- a **temporal 1D depthwise-separable conv stack** (Conv1d depthwise + Conv1d pointwise + BatchNorm1d), and
- two **rank-mixing tensor manipulations** between the freq and temporal stacks (3D permute, 4D→3D reshape, 3D mean over the last dim).

Without these, the model cannot be compiled. The user requirement is to add the missing surface with **minimal duplication** and to keep BatchNorm/ReLU/permute/mean in float — only conv and matmul are perf-critical and worth quantizing later.

## Gap analysis (`DeepSenseDWClean.py` vs `nn_ops_float.h`)

Already supported (no work):
- `nn.Conv2d` standard and depthwise (`conv2d_nhwc`, `depthwise_conv2d_nhwc`)
- `nn.BatchNorm2d` (`batchnorm2d_nhwc`)
- `nn.ReLU` (`relu`)
- `nn.Linear` (`dense`)
- 4D `tensor.permute` (`permute_4d`)
- `tensor.mean(dim=[2,3])` over 4D (`mean_hwc`)
- `nn.Dropout`/`nn.Dropout2d` — inference no-op, nothing to add

Missing or partial:

| # | Op (PyTorch) | Status | Resolution |
|---|---|---|---|
| 1 | `nn.Conv1d` (standard, k=1 pointwise) | missing | wrap `conv2d_nhwc` with `H=1` |
| 2 | `nn.Conv1d` depthwise (`groups=C`) | missing | wrap `depthwise_conv2d_nhwc` with `H=1` |
| 3 | `nn.BatchNorm1d` | missing | wrap `batchnorm2d_nhwc` with `h=1` |
| 4 | `tensor.permute(0, 2, 1)` (3D) | missing | new `permute_3d` C kernel |
| 5 | `tensor.reshape(B, I, C*S)` (4D→3D, contiguous tail-merge) | partial | layout-only; codegen `method_reshape` already exists, validate it under the chosen NLC layout |
| 6 | `tensor.mean(dim=-1)` over 3D `[B,T,I]` | partial | reuse `mean_last_dim` via codegen shape adapter (flatten leading dims) |

## Design

### Layout convention

- 4D tensors: **NHWC** (existing).
- 3D tensors (output of spectrum projection, input/output of temporal stack): **NLC** = `[B, L, C]`. This is the natural 1D analogue of NHWC and makes the 1D-as-2D-with-`H=1` wrapping trivial.

PyTorch's `[B, C, L]` for the temporal stack is converted at the boundary by the existing reshape/permute lowering — same approach the compiler already uses for the NCHW→NHWC boundary.

### New / extended C kernels (in `nn_ops_float.h`)

1. **`permute_3d`** — generic 3D permute. Drops in next to `permute_4d`, ~30 lines, same pattern.
   ```c
   static inline void permute_3d(
       const float* in,
       int d0, int d1, int d2,
       int p0, int p1, int p2,
       float* out);
   ```
2. **`mean_last_dim`** — no kernel change. Comment update only: clarify that callers reducing an N-D tensor over the last dim should pass `rows = product(leading_dims)`, `cols = last_dim`. The kernel is already shape-agnostic.

No new conv or batchnorm kernels. Conv1d/BatchNorm1d are pure codegen wrappers that emit calls to the existing 2D kernels with `H=1`.

### Compiler-side changes

**Frontend / lowering (`pytorch_to_c/lowering/lower.py`):**
- Register `nn.Conv1d` → IR op `conv1d` (carry `groups`, `kernel_size`, `stride`, `padding`, `bias`).
- Register `nn.BatchNorm1d` → IR op `batchnorm1d` (same param names as `batchnorm` 2D).
- 3D `permute` and 3D `mean(dim=-1)` already arrive as `method_permute` / `method_mean`; the rank dispatch lives in codegen.

**Codegen (`pytorch_to_c/codegen/c_printer.py`):**
- `_generate_conv1d`: emits `conv2d_nhwc(...)` or `depthwise_conv2d_nhwc(...)` with `in_h=1`, `k_h=1`, `stride_h=1`, `pad_h=0`. Weight reshape happens at param-emit time and depends on the path:
  - **standard Conv1d**: PyTorch weight `[out_ch, in_ch, k]` → emit as HWIO `[1, k, in_ch, out_ch]` (the layout `conv2d_nhwc` expects).
  - **depthwise Conv1d** (`groups == in_ch`, `out_ch == in_ch`): PyTorch weight `[C, 1, k]` → emit as HWC `[1, k, C]` (the layout `depthwise_conv2d_nhwc` expects).
- `_generate_batchnorm1d`: emits `batchnorm2d_nhwc(..., h=1, w=L, c=C, ...)`.
- `_generate_permute`: dispatch on input rank — rank 4 → `permute_4d`, rank 3 → `permute_3d`. Reject rank ≠ 3,4 with a clear error.
- `_generate_mean`: when `dim=-1` (or `dim==rank-1`) on rank ≥ 2, emit `mean_last_dim(in, prod(leading_dims), last_dim, out)`. Existing 4D `mean(dim=[2,3])` path is unchanged.

### Buffer sizing

`c_printer.py` currently sizes activation slots from `output_shape`. The new ops don't change shape arithmetic — they preserve or trivially reduce dims that are already on the IR node. No new shape inference required.

### Quantization (out of scope for this spec)

The user wants conv and matmul to be the *only* ops that ever go to lower precision. This spec adds **float-only** Conv1d/BatchNorm1d codegen. INT8 paths for Conv1d will be a follow-up that can also be wrappers (`conv2d_nhwc_int8` with `H=1`). BN/ReLU/permute/mean stay float regardless of model dtype.

## Testing

For each new op, mirror the existing test pattern under `Tiny-NN-in-C/test/`:

1. **Unit test per kernel** (`permute_3d`): random inputs, compare to a Python reference (`numpy.transpose`) byte-exact. The user requested no kernel change for `mean_last_dim`, so its existing test already covers it.
2. **Codegen smoke tests**: a tiny PyTorch module containing each new op (Conv1d depthwise, Conv1d pointwise, BatchNorm1d, 3D permute, 3D mean(dim=-1)) — verify with the existing `tools.verify_model` harness against PyTorch within float tolerance.
3. **End-to-end**: compile the full `DeepSenseDWCleanBackbone` with a small config and run `verify_model` for ≥ 50 random samples. This is the acceptance test for the whole spec.

## Out of scope

- INT8 / W8A8 paths for the new ops (follow-up).
- Any RNN / recurrent support (explicitly excluded by user).
- Changes to NHWC layout convention.
- Optimizing the 1D paths beyond the H=1 wrapper (wrappers may be 1–2× slower than dedicated NLC kernels; acceptable for v1).

## Files touched

- `Tiny-NN-in-C/src/c_ops/nn_ops_float.h` — add `permute_3d`, comment update on `mean_last_dim`.
- `Tiny-NN-in-C/src/pytorch_to_c/lowering/lower.py` — register Conv1d, BatchNorm1d.
- `Tiny-NN-in-C/src/pytorch_to_c/codegen/c_printer.py` — `_generate_conv1d`, `_generate_batchnorm1d`, rank dispatch in `_generate_permute` and `_generate_mean`.
- `Tiny-NN-in-C/test/` — unit + codegen + end-to-end tests.
