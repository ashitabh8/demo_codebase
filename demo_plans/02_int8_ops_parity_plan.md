# INT8 Ops Parity Plan

## Scope
Close parity gaps between float and int8 operation coverage needed by generated models, with emphasis on reduction ops and dtype-safe view transforms.

## Owner
Person A

## Dependencies
- `Tiny-NN-in-C/src/c_ops/nn_ops_float.h`
- `Tiny-NN-in-C/src/c_ops/nn_ops_int8.h`
- `Tiny-NN-in-C/src/pytorch_to_c/codegen/c_printer.py`
- C tests under `Tiny-NN-in-C/test`

## Implementation Tasks
1. Add required int8 reduction/pooling ops:
   - `mean_hwc_int8`
   - `mean_last_dim_int8`
   - `global_average_pool_2d_int8`
   - `adaptive_avg_pool_2d_1x1_int8`
2. Implement int8 `flatten_int8` helper.
3. Update codegen to route quantized reduction nodes to int8 ops.
4. Remove float-size assumptions in view/squeeze/unsqueeze fallback paths:
   - replace `sizeof(float)` with dtype-aware element size.
5. Add unit tests in `test_c_ops_int8.c` for new int8 kernels.
6. Confirm generated int8 inference remains classification-consistent against reference outputs.

## Validation Commands
- `make -C Tiny-NN-in-C test_int8_ops`
- `python src2/gen_code/run_validation.py --mode quick`
- `python src2/gen_code/compare_outputs.py --c_output_path src2/gen_code/c_outputs.txt --reference_dir src2/gen_code/test_data`

## Exit Criteria
- New int8 ops compile and pass unit tests.
- Quantized graph paths use int8 helpers where expected.
- No dtype-related memcpy bugs in quantized view transforms.
- Validation outputs pass agreed classification checks.

## Handoff Artifacts
- Updated int8 ops and codegen files.
- Test updates and validation logs.
