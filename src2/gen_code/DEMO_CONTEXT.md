# Demo Context

## Goal
Run compressed-model inference on Arduino GIGA R1 using pre-exported mel-audio samples from Parkland, stream predictions to PC, and visualize rolling accuracy for Arduino vs Raspberry Pi in a browser dashboard.

## Pipeline Overview
1. Export demo samples from Parkland (`Polaris`, `Warhog`, `Truck`) as mel features.
2. Generate firmware bundle (`model.*`, ops headers, `demo_samples.h`) and flash Arduino.
3. Stream predictions from Arduino (and optionally Raspberry Pi) using `sample_<id> ...` protocol.
4. Browser dashboard ingests stream/replay and displays rolling accuracy bars (window=20 by default).

## Key Files
- Exporter: `src2/gen_code/export_demo_samples.py`
- Export validator: `src2/gen_code/validate_demo_export.py`
- Browser dashboard: `src2/gen_code/demo_ui.py`
- Arduino firmware: `src2/gen_code/arduino_demo/arduino_demo.ino`
- Arduino bundle prep: `src2/gen_code/arduino_demo/prepare_arduino_bundle.py`
- Healthcheck: `src2/gen_code/demo_healthcheck.py`
- Mock/replay helpers:
  - `src2/gen_code/create_mock_session_log.py`
  - `src2/gen_code/generate_mock_streams.py`
  - `src2/gen_code/replay_streams.py`

## Protocol Contract
Prediction lines should follow:

`sample_<id> src=<arduino|rpi> pred=<int> target=<int> inf_ms=<float> logits=<comma_vals_optional>`

The dashboard uses `sample_id`, `src`, and `pred` (plus ground truth from `demo_labels.csv`) to compute rolling accuracy.

## Commands

### 1) Export 50 samples
```bash
python src2/gen_code/export_demo_samples.py \
  --yaml_path src2/data/Parkland.yaml \
  --output_dir src2/gen_code/demo_data \
  --num_samples 50 \
  --split test \
  --export_header
```

### 2) Validate export
```bash
python src2/gen_code/validate_demo_export.py --data_dir src2/gen_code/demo_data
```

### 3) Prepare Arduino bundle
```bash
python src2/gen_code/arduino_demo/prepare_arduino_bundle.py
```

### 4) Browser dashboard (replay mode, 1s updates)
```bash
python src2/gen_code/create_mock_session_log.py \
  --labels_csv src2/gen_code/demo_data/demo_labels.csv \
  --output_csv src2/gen_code/demo_data/session_log.csv

python src2/gen_code/demo_ui.py \
  --labels_csv src2/gen_code/demo_data/demo_labels.csv \
  --replay_csv src2/gen_code/demo_data/session_log.csv \
  --replay_delay_s 1.0 \
  --host 0.0.0.0 \
  --port 8050
```

If remote, tunnel from local machine:
```bash
ssh -L 8050:localhost:8050 <user>@<server>
```
Then open `http://localhost:8050`.

### 5) Browser dashboard (live streams)
```bash
python src2/gen_code/demo_ui.py \
  --labels_csv src2/gen_code/demo_data/demo_labels.csv \
  --arduino_port /dev/ttyACM0 \
  --rpi_source serial:/dev/ttyUSB0 \
  --host 0.0.0.0 \
  --port 8050
```

### 6) Healthcheck
```bash
python src2/gen_code/demo_healthcheck.py \
  --labels_csv src2/gen_code/demo_data/demo_labels.csv \
  --samples_csv src2/gen_code/demo_data/demo_samples.csv \
  --require_web
```

## Notes
- `--replay_delay_s` and `--file_delay_s` default to `1.0` for visual debugging.
- For faster playback, set `--replay_delay_s 0.05`.
- If using live serial, delay flags do not affect serial input speed.
