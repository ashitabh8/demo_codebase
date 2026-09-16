# ACIDS Spectral MLP (Welch, 1600 → 83 → W8)

Self-contained C inference packet. **No Python required.**

## Build

```bash
make
```

## Run

```bash
./spectral_infer                  # all packaged samples/
./spectral_infer samples/foo.bin  # one or more raw float32[1600] files
```

Human-readable txt → bin:

```bash
./spectral_txt_to_bin samples_txt/foo.txt /tmp/foo.bin
./spectral_infer /tmp/foo.bin
```

## Input

- **`.bin`**: exactly 1600 little-endian `float32` values (6400 bytes), no header.
- **`.txt`**: `name`, `label`, `RAW_AUDIO_STREAM`, `1600`, then 1600 floats.

## Pipeline

```text
audio[1600] @ 1600 Hz
  → Welch RFFT (n_fft=160, hop=80) over full second → 83 raw features
  → StandardScaler
  → W8 MLP (83→128→128→10)
  → argmax class
```

Classes: `background`, `1` … `9`.
