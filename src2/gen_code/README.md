# DeepSense C Validation

This directory contains scripts to:

1. compile `DeepSenseDWSimpleBackbone` from a PyTorch checkpoint into C,
2. run generated C inference on reusable sample inputs,
3. compare C logits against reference logits on 50 samples,
4. write a CSV report with prediction agreement (`CORRECT`/`INCORRECT`).

## Files

- `compile_deepsense.py`: checkpoint -> generated C (`generated/`).
- `export_test_data.py`: generates reusable samples and PyTorch references.
- `test_main.c`: runs `model_forward()` for all samples.
- `compare_outputs.py`: compares C vs reference and writes CSV report.
- `run_validation.py`: orchestration script for two workflows below.

## Output Contract

`test_main.c` prints one line per sample:

`sample_<idx> <logit_0> <logit_1> ... <logit_9>`

`expected_outputs.txt` uses the same format, so C-only checks can run without PyTorch.

## Workflow A: C-only Quick Test

Use this when generated C files and reusable references already exist.

```bash
cd src2/gen_code
python run_validation.py quick-c-check
# or
make quick
```

What it does:

1. Builds `test_inference` from `generated/model.c`.
2. Runs C inference on saved `test_inputs.h` samples -> `c_outputs.txt`.
3. Compares against `expected_outputs.txt`.
4. Writes `comparison_report.csv`.

## Workflow B: Full Checkpoint Validation (Compile + 50 Samples)

Use this when you want end-to-end validation from a checkpoint.

```bash
cd src2/gen_code
python run_validation.py full-checkpoint-check \
  --checkpoint_path /home/misra8/demo_codebase/src2/experiments/20260402_005734_only_audio_deepsense_dw_simple_tiny/models/best_model.pth \
  --num_samples 50
# or
make full CHECKPOINT=/home/misra8/demo_codebase/src2/experiments/20260402_005734_only_audio_deepsense_dw_simple_tiny/models/best_model.pth NUM_SAMPLES=50
```

What it does:

1. Compiles checkpoint to C (`generated/`).
2. Exports 50 deterministic samples and references:
   - `test_data/test_input_<N>.txt`
   - `test_data/pytorch_output_<N>.txt`
   - `test_inputs.h`
   - `expected_outputs.txt`
3. Builds and runs C inference -> `c_outputs.txt`.
4. Compares outputs and writes `comparison_report.csv`.

## CSV Report

`comparison_report.csv` has one row per sample with:

- `sample_id`
- `pytorch_pred` (argmax of reference logits)
- `c_pred` (argmax of C logits)
- `classification_match` (`CORRECT` or `INCORRECT`)
- `max_abs_err`
- per-class logits: `pt_logit_0..9`, `c_logit_0..9`

## Notes

- Sample generation is deterministic with `--seed` (default `1234`), so tests are reproducible.
- Quick test can run without providing a checkpoint, as long as `generated/`, `test_inputs.h`, and `expected_outputs.txt` already exist.

## Make Commands

```bash
cd src2/gen_code
make help
make quick
make full CHECKPOINT=/path/to/best_model.pth NUM_SAMPLES=50
```

Useful step-by-step targets:

- `make compile`
- `make export`
- `make build`
- `make run`
- `make compare-dir`
- `make compare-file`
