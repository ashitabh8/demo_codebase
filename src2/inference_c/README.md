# ACIDS 1-channel mel inference preprocessing (C)

Standalone C99 log-mel preprocessing for `student_audio_deepsense_dw_large_mel_1ch`.
Matches the Python test path (`augmentation_mode="no"`, no normalization, no downsampling).

## Pipeline scope

```
Input (upstream)                    This module                    Model
─────────────────                   ───────────                    ─────
PCM @ 16 kHz, 1 s                   1600 float32 @ 1.6 kHz          [1, 7, 80] float32
       │                            (mic ch0, already decimated)
       └── decimate (not in C) ──►  split → rfft → mel → log ──►  DeepSense DW
```

Upstream is responsible for:

- 16 kHz → 1.6 kHz decimation (see `data_augmenter/audio_downsample.py`)
- Selecting mic channel 0 (see `data_augmenter/audio_channel_select.py`)

This module does **not** apply minmax normalization (finetune/test skip it; see `train_test/test.py`).

## Input / output

| Field | Value |
|-------|-------|
| Input | `float input[1600]` — 1 s @ 1600 Hz, single channel |
| Segment split | Trim to 1596 samples → 7 × 228 samples/segment (last 4 input samples ignored) |
| Per-segment FFT | `torch.fft.rfft(x, n=160)` — uses first 160 samples of each 228-sample segment |
| Mel | `n_fft=160`, `mel_bins=80`, `fmin=20`, `fmax=800`, `sample_rate=1600` |
| Output CHW | `float out[560]` — layout `[C=1][H=7][W=80]` (PyTorch NCHW) |
| Output HWC | `float out[560]` — layout `[H=7][W=80][C=1]` (same flat order when C=1) |

## Files

| File | Purpose |
|------|---------|
| `acids_mel_preprocess.h` | Public API |
| `acids_mel_preprocess.c` | Segment split + RFFT + mel + log |
| `acids_mel_tables.h` | Mel filterbank `[80][81]` (auto-generated) |
| `acids_rfft_tables.h` | RFFT basis matrices from `torch.fft.rfft` (auto-generated) |
| `export_mel_tables.py` | Regenerate both headers after YAML mel param changes |
| `test_mel_parity.py` | Assert C output matches `MelPreprocessor` within 1e-4 |

## API

```c
#include "acids_mel_preprocess.h"

float pcm_1600[ACIDS_INPUT_SAMPLES];   /* fill from your 1.6 kHz decimator */
float model_input[ACIDS_OUTPUT_CHW_SIZE];

acids_mel_preprocess_chw(pcm_1600, model_input);
/* model_input is [1][7][80] flattened: out[seg * 80 + mel] */
```

## Build

Static library:

```bash
cd src2/inference_c
gcc -c -O2 -std=c99 acids_mel_preprocess.c -o acids_mel_preprocess.o
ar rcs libacids_mel_preprocess.a acids_mel_preprocess.o
```

Shared library (used by parity test):

```bash
gcc -shared -fPIC -O2 -std=c99 acids_mel_preprocess.c -o libacids_mel_preprocess.so -lm
```

## Regenerate coefficient tables

After changing mel params in `data/ACIDS.yaml`:

```bash
python3 export_mel_tables.py
```

## Parity test

```bash
python3 test_mel_parity.py
```

Compares C output against `data_augmenter.mel_preprocess.MelPreprocessor` on random, zero, and ramp inputs.

## Validate against Python export

Use `train_test/export_acids_test_txt.py` to export post-augmenter mel tensors, then compare with C output on the same 1600 Hz input buffer:

```bash
cd src2/train_test
python3 export_acids_test_txt.py \
  --experiment_name finetune_audio_deepsense_dw_large_mel_1ch_ch0_5class_supcon_unfreeze \
  --yaml_path ../data/ACIDS.yaml \
  --num_samples 10 \
  --gpu -1
```

Exported `c_nhwc/*.txt` files use HWC flatten order (identical to CHW when `C=1`).
