# Plan: Log-Mel Preprocessing + Normalization Fix + Arch Improvements

## Context

Current pipeline processes audio as: raw time-domain → FFT → real+imag interleaved → DeepSense DW conv stack. User asked about (1) whether normalization is hurting accuracy, (2) what NAS literature says about conv layer design, and (3) whether to switch to Mel preprocessing.

Architecture under discussion: `student_audio_deepsense_dw` (Parkland.yaml L192–208).

---

## Finding 1: Normalization is Suboptimal (Medium Impact)

**What it does:** `normalize.py` computes **one global scalar** mean + std by flattening the entire tensor (`reshape(-1)`). For audio shaped `[B, 2, 10, 1600]`, this means a single number represents the "center" of a 32,000-element space where low-frequency bins have 10× more energy than high-frequency bins.

**Why it hurts:** After FFT, bin 0 (DC offset) has completely different magnitude from bin 800 (high freq). Collapsing them to one mean distorts the relative spectral structure the network needs to discriminate vehicles. It doesn't destroy learning but throws away free information.

**What Parkland uses:** `method: minmax` to `[0, 1]` globally — same issue.

**Fix:** Normalize per-frequency-bin (axis = last dim). Compute `mean[freq_bin]` and `std[freq_bin]` over the training set for each of the 1600 bins. This preserves relative spectral shape while centering.

**Estimated impact:** +1–3% val accuracy. Not huge, but essentially free.

---

## Finding 2: NAS Trends for Conv Layers

The user's memory of "thick on edges, thin in the middle" maps to two real concepts:

### Kernel Size Decay (what the current arch already does)
DeepSense DW already follows the empirically-confirmed pattern: **large kernels early → small kernels late** `[20, 10, 5, 3]`. This is correct. NAS (EfficientNet, MNASNet, FBNet) consistently validates this for spectrograms:
- **Early layers**: wide frequency kernel captures global spectral context (e.g., engine rumble spanning 200–800Hz = many bins)
- **Late layers**: small kernel refines local features on already-compressed representations

### Inverted Bottleneck (the more impactful finding)
What NAS actually most consistently finds is the **inverted bottleneck** (MobileNetV2 → EfficientNet → ConvNeXt all converge on this):
```
narrow in → 1×1 expand wide → 3×3 DW-sep (wide) → 1×1 squeeze → narrow out
```
Compare to standard bottleneck (ResNet): `wide → narrow → narrow → wide`.

The current temporal stack (`256 → 256 → 256 → 256`) is flat. Replacing it with inverted bottleneck blocks (`256 → 512 DW → 256`) would improve expressivity with modest param increase.

**For spectrograms specifically:** NAS also finds that asymmetric kernels (`[1, k]` in freq, `[k, 1]` in time) are better than square — the current architecture already uses `[1, k]` kernels which is correct.

---

## Finding 3: Switch to Log-Mel — Yes, Recommended

**Current pipeline:** FFT (complex) → real+imag → 2 channels × 1600 bins → conv stack compresses to 13 bins.

**Why Mel is better for this task:**

| | Raw FFT (current) | Log-Mel |
|---|---|---|
| Input channels | 2 (real + imag) | 1 (magnitude) |
| Freq bins | 1600 | 64–128 |
| Perceptual alignment | No | Yes (vehicle acoustics are log-scale) |
| Phase info | Included (noise for this task) | Discarded |
| Spectrum projection params | 6,656 → 256 = **1.7M** | ~1,024 → 256 = **262K** (w/ 128 bins, 2 layers) |

Key points:
1. **Phase is noise here.** Vehicle classification depends on spectral energy, not phase coherence. Real+imag channels add parameters to learn to ignore phase.
2. **Mel matches vehicle acoustics.** Engine harmonics, transmission noise, tire noise — all strongest in log-frequency space (20Hz–4kHz).
3. **Massive param reduction in spectrum projection.** With 64 Mel bins compressed to ~8 after 2 freq conv layers: `512×8 = 4096 → 256 = 1M params`. With the channel reduction trick below, can get to ~260K.

**Spectrum projection bottleneck fix (independent improvement):**
Current: flatten `[512, 13]` → Linear(6656, 256) = **1.7M params** (67% of model!).
Better: 1×1 conv to reduce channels first: `512→128` (pointwise), then flatten `128×13=1664` → Linear(1664, 256) = **425K params** (4× reduction, same architecture, minor accuracy cost).

---

## Proposed Changes

### Change 1: Per-frequency-bin Normalization (Quick win)
- **File:** `src2/train_test/normalize.py`
- Change stats computation from `reshape(-1)` to computing mean/std over the batch+channels+segments axes, keeping the freq-bin axis intact
- Result: normalizer has shape `[freq_bins]` not scalar

### Change 2: Log-Mel Preprocessing
- **New file:** `src2/data_augmenter/mel_preprocess.py` (or add to `Augmenter.py`)
- Insert between time-domain augmentation and the existing `fft_preprocess()` call - there should be a switch , add option to training part of config which preprocessing we should be using FFT or Mel run accordingly
- Implement melscale manually do not use library FFT -> Mel -> log
- Config additions to YAML: `mel_bins: 80`, `mel_fmin: 20`, `mel_fmax: 8000`

### Change 3: Redesign Freq Conv Stack for 80 Mel Bins
- **File:** `src2/models/DeepSenseDepthwise.py` + `Parkland.yaml`
- Current kernels `[20,10,5,3]` / strides `[10,5,2,1]` designed for 1600 bins → **collapse 80 bins in 1 step**
- New config: `kernel_sizes_freq: [[1,5],[1,3],[1,3]]`, `strides_freq: [[1,2],[1,2],[1,1]]`
  - 80 → 38 → 18 → 18 (3 layers, preserves meaningful freq resolution)
  - Input channels: 1 (magnitude only, not 2)
- Optional: `channels_freq: [64, 128, 256]` instead of `[128, 256, 512, 512]` to further reduce size

### Change 4: Inverted Bottleneck in Temporal Stack (Optional, accuracy boost)
- **File:** `src2/models/DeepSenseDepthwise.py`
- Replace flat `256→256` DWConv1d layers with inverted bottleneck: `256 → [expand 2×] → DW → [squeeze] → 256`
- Add to `DSTemporalDWLayer`: `expansion_ratio` param (default 2)
- `+~65K params` per layer but better expressivity

### Change 5: Mel Test Suite
- **New file:** `src2/tests/test_mel_preprocess.py`
- Tests run with `python -m pytest src2/tests/test_mel_preprocess.py -v`

**What to test:**

1. **Filterbank matrix shape**: `build_mel_filterbank(n_fft, n_mel, fmin, fmax, sample_rate)` returns shape `[n_mel, n_fft//2+1]`, all values ≥ 0
2. **Frequency monotonicity**: Mel center frequencies are strictly increasing
3. **Filter coverage**: Every FFT bin between fmin and fmax is covered by at least one filter (no dead bins)
4. **Shape correctness**: Input `[B, C, segments, n_fft]` time-domain → Output `[B, C, segments, n_mel]`
5. **vs torchaudio reference**: Our manual `FFT → mel_filterbank @ magnitude² → log` should match `torchaudio.transforms.MelSpectrogram` output to within 1e-3 absolute tolerance (using same n_fft, n_mel, fmin, fmax, sample_rate, power=2, norm=None)
6. **Log stability**: No `-inf` or `nan` in output — floor at small epsilon before log
7. **Hz↔Mel round-trip**: `mel_to_hz(hz_to_mel(f)) ≈ f` to within 1e-4 for range 20–8000 Hz
8. **DC and Nyquist bins**: Filterbank correctly handles bin 0 (DC) and bin n_fft//2 (Nyquist) - DC should have near-zero weight (fmin > 0)
9. **Energy ordering**: A pure tone at a low frequency should produce higher energy in low Mel bins than high Mel bins
10. **Batch consistency**: Processing a batch of identical inputs gives identical outputs (no batch-level leakage)

---

## Implementation Order

1. **Log-Mel preprocessing** (Change 2) — new `MelPreprocessor` class in `src2/data_augmenter/mel_preprocess.py`
2. **Mel test suite** (Change 5) — write and pass all 10 tests before wiring into training
3. **Augmenter wiring** — add `preprocess_mode: "mel"/"fft"` switch in Augmenter + YAML
4. **Freq conv redesign** (Change 3) — update Parkland.yaml model config for 80-bin input
5. **Normalization fix** (Change 1) — update `normalize.py` stat computation
6. **Inverted bottleneck** (Change 4) — optional, after above are validated

## Verification

After Changes 1–4, run 1-epoch sanity check:
```bash
cd src2/train_test
python train.py -experiment_name mel_test -yaml_path ../data/Parkland.yaml -gpu 0
```
Check `train_log_claude.jsonl` for:
- `header.dataset.in_channels` should be 1 (magnitude only, not 2)
- `epoch.val_acc` trend — should not regress vs FFT baseline
- `epoch.per_class_recall` — check if previously hard classes improve with Mel

## Critical Files
- `src2/data_augmenter/mel_preprocess.py` — new file: manual Mel filterbank implementation
- `src2/tests/test_mel_preprocess.py` — new file: correctness tests
- `src2/data_augmenter/Augmenter.py` — add `preprocess_mode` switch
- `src2/train_test/normalize.py` — fix stat computation axis
- `src2/models/DeepSenseDepthwise.py` — inverted bottleneck (Change 4 only)
- `src2/data/Parkland.yaml` — model config (kernel_sizes_freq, channels_freq, mel params)
