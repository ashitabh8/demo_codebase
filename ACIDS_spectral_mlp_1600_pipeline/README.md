# Spectral MLP Pipeline (Welch 83-dim)

Self-contained train → W8 quantize → **C inference** package for ACIDS vehicle classification.

- **Input:** 1 s ch0 continuous audio stream from ACIDS `.pt`
- **Frontend:** Welch mean periodogram → **83** features
- **Model:** MLP `83 → 128 → 128 → C`, StandardScaler, weight-only **int8**
- **Inference:** entirely in C (`deploy/spectral_infer`)

`C` is the number of classes. It defaults to `max(label) + 1` over train and val (10 for the published ACIDS vehicle task). Override with `--num_classes` / `--class_names` when your dataset needs a different head.

Needs `gcc` / `make`. **Tiny-NN-in-C is vendored** in `Tiny-NN-in-C/` (used only for W8 export).

## Latest full-ACIDS results (10 classes)

| Split | N | Accuracy | Macro F1 |
|-------|---|----------|----------|
| Train | 21631 | 98.1% | 0.963 |
| Val | 2190 | 90.7% | 0.778 |
| **Test (C infer)** | **2190** | **91.0%** | **0.780** |

These numbers are from the original 10-class run on `random_partition` where **val == test** (not a true hold-out). Prefer the default **fixed** partition in `paths.yaml` for disjoint train/val/test and re-run to get honest test metrics. Test scoring calls `./spectral_infer` on every test `.bin` (Welch + scaler + W8 MLP in C).

## Choosing 10 vs 15 classes (or any `C`)

Do **not** hand-edit `deploy/acids_class_names.h` or the W8 weights to flip class count.
Retrain (or re-export from a checkpoint trained with that `C`) so the MLP head,
`acids_class_names.h`, and C logit buffer all match.

Label ids are integers:

- `.txt` streams: line 2 of each file
- ACIDS `.pt`: `sample["label"]["vehicle_type"]`

They must already be contiguous ids `0 .. C-1`. The pipeline does not remap labels.

| Goal | What to pass |
|------|----------------|
| **10 classes** (published ACIDS vehicle) | nothing — data max label is 9 → `C=10` |
| **15 classes** | `--num_classes 15` (labels must be `0..14`) |
| Custom names | `--class_names my_classes.txt` (one name per line, length = `C`) |

### Default (read from data)

Omit the flag. Training sets `C = max(train labels, val labels) + 1`. The published ACIDS vehicle set stays at 10 with no extra arguments.

```bash
python3 train_from_1600.py --data_root data --output_dir artifacts
python3 run_pipeline.py --data_root data --epochs 20 --gpu -1
python3 run_acids_pipeline.py --paths_yaml paths.yaml --epochs 50 --gpu 0
```

### Force 15 classes (example)

```bash
# labels in data/ or exported_acids must be 0..14
python3 run_pipeline.py --data_root data --num_classes 15 --epochs 20 --gpu -1
python3 run_acids_pipeline.py --paths_yaml paths.yaml --num_classes 15 --epochs 50 --gpu 0
```

Or call train alone:

```bash
python3 train_from_1600.py --export_root exported_acids --num_classes 15
python3 train_from_1600.py --data_root data --num_classes 15
```

Every label must be in `0 .. N-1`. A higher label raises an error.

### Optional class names

Create a text file with one name per line (length must equal `C`):

```text
background
truck
car
...
```

```bash
python3 train_from_1600.py --data_root data --num_classes 15 --class_names my_classes.txt
python3 run_pipeline.py --data_root data --num_classes 15 --class_names my_classes.txt
python3 run_acids_pipeline.py --paths_yaml paths.yaml --num_classes 15 --class_names my_classes.txt
```

If you omit `--class_names`:

- `C == 10` keeps the ACIDS defaults (`background`, `1` … `9`)
- otherwise classes are named `0` … `C-1`

Names are stored in `artifacts/spectral_mlp.pth` as `class_names` / `num_classes`.

### After training: do not hand-edit C artifacts

Do not edit `deploy/model.c`, `deploy/weights.h`, or `deploy/acids_class_names.h` by hand. The next W8 export rewrites them from the checkpoint:

```bash
python3 export_w8_spectral_mlp.py \
  --ckpt_path artifacts/spectral_mlp.pth \
  --output_dir deploy
make -C deploy
```

`export_w8_spectral_mlp.py` writes `acids_class_names.h` (`ACIDS_NUM_CLASSES` and the name table) so C inference's logit buffer matches the trained head. Packaging and C test scoring also read class names from the checkpoint.

## Put your data here (default pathway)

Drop `RAW_AUDIO_STREAM` `.txt` clips into:

```text
data/train/*.txt
data/val/*.txt
data/test/*.txt    # held-out; used for final report + C scoring
```

| Split | Role |
|-------|------|
| **train** | Fit scaler + MLP |
| **val** | Checkpoint selection (best val accuracy) |
| **test** | Held-out metrics + C `spectral_infer` scoring |

```bash
cd ACIDS_spectral_mlp_1600_pipeline
python3 run_pipeline.py --data_root data --epochs 20 --gpu -1
```

That trains on `train`, selects on `val`, scores `test` (if present), writes W8 into `deploy/`, and builds `spectral_infer`.  
If `data/test/` is empty, it warns and skips held-out C scoring.

### Optional: full ACIDS `.pt` indices

Index lists (disjoint fixed partition) live under `indices/` and are referenced by `paths.yaml`:

```bash
python3 run_acids_pipeline.py --paths_yaml paths.yaml --epochs 50 --gpu 0
```

Quick smoke:

```bash
python3 run_acids_pipeline.py --max_export_samples 200 --epochs 10 --gpu -1
```

Export refuses overlapping splits unless you pass `--allow_overlapping_splits`.

### Steps (`run_acids_pipeline.py`)

1. **`export_acids_indices.py`** — ACIDS `.pt` → ch0 continuous stream → `exported_acids/{train,val,test}_X.npy` + `test_bin/*.bin` (rejects overlapping splits)
2. **`train_from_1600.py`** — Welch 83-dim → StandardScaler → MLP on train/val; reports test if exported
3. **`export_w8_spectral_mlp.py`** — Tiny-NN W8 → `deploy/model.*`, `weights.h`, `spectral_scaler.h`, `acids_class_names.h`
4. **`make -C deploy`** — builds `spectral_infer` (Welch C + kissfft + scaler + W8)
5. **`run_c_test_infer.py`** — C inference on all **test** bins → `artifacts/c_test_report.json`

## Run C inference on samples

```bash
cd deploy
make

# packaged smoke samples
./spectral_infer

# any raw stream .bin
./spectral_infer /path/to/clip.bin

# from samples_txt-style .txt
./spectral_txt_to_bin clip.txt /tmp/clip.bin
./spectral_infer /tmp/clip.bin
```

Each inference prints `pred_id`, `pred_name`, and **`confidence`** (softmax probability of the predicted class, in `[0, 1]`). That uses `ACIDS_NUM_CLASSES` from `acids_class_names.h`, so the same binary works for a 10-class or 15-class packet after you retrain/export for that head.

`.txt` layout:

```text
sample_name
<label_id 0..C-1>    # integer class id; ACIDS vehicle defaults: 0=background, 1..9
RAW_AUDIO_STREAM
1600
<1600 floats>
```

## Layout

```text
data/train/*.txt          # your RAW_AUDIO_STREAM clips (default pathway)
data/val/*.txt
data/test/*.txt           # held-out split
indices/*_index.txt       # optional ACIDS .pt path lists
paths.yaml                # points at indices/ for run_acids_pipeline.py
export_acids_indices.py
train_from_1600.py
export_w8_spectral_mlp.py
package_deploy_samples.py
run_c_test_infer.py
run_pipeline.py           # data/train|val|test txt → W8 → C test score
run_acids_pipeline.py     # indices/*.txt .pt lists → train → W8 → C test score

welch_features.py
load_1600_txt.py
Tiny-NN-in-C/             # vendored W8 compiler (export only)
exported_acids/           # after export (gitignored)
artifacts/                # checkpoint + reports (*.pth gitignored)
deploy/                   # COMPLETE C inference packet (handoff)
```

## Handoff

Zip/copy **`deploy/`** alone. Recipients need only `gcc`/`make` — no Python, no Tiny-NN, no ACIDS `.pt` data.

Scaler stays in C (`spectral_scaler.h`), not folded into layer weights. Class count and names in the packet come from training via `acids_class_names.h`.
