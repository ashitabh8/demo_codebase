# Arduino Demo Runbook

## 1) Prepare Data and Firmware Inputs
1. Export samples and labels:
   - `python src2/gen_code/export_demo_samples.py --yaml_path src2/data/Parkland.yaml --output_dir src2/gen_code/demo_data --num_samples 90 --split test --export_header`
2. Validate export:
   - `python src2/gen_code/validate_demo_export.py --data_dir src2/gen_code/demo_data`
3. Bundle Arduino files:
   - `python src2/gen_code/arduino_demo/prepare_arduino_bundle.py`

## 2) Flash and Verify Arduino
1. Open `src2/gen_code/arduino_demo/arduino_demo.ino` in Arduino IDE.
2. Ensure `MODEL_OUTPUT_SIZE` matches generated model output classes.
3. Flash Arduino GIGA R1.
4. Open serial monitor (`115200`) and verify:
   - `ready src=arduino commands=START,STOP,RESET,STATUS`

## 3) Start UI (Live)
1. Install web UI dependencies:
   - `pip install flask pyserial`
2. Start UI:
   - `python src2/gen_code/demo_ui.py --labels_csv src2/gen_code/demo_data/demo_labels.csv --arduino_port /dev/ttyACM0 --rpi_source serial:/dev/ttyUSB0 --host 0.0.0.0 --port 8050`
3. Open browser:
   - `http://<server-ip>:8050` (or use SSH tunnel: `ssh -L 8050:localhost:8050 <user>@<server>`)
3. Send `START` on Arduino serial input.

## 4) Health Checks
- Run:
  - `python src2/gen_code/demo_healthcheck.py --labels_csv src2/gen_code/demo_data/demo_labels.csv --samples_csv src2/gen_code/demo_data/demo_samples.csv`
- Expected:
  - labels/samples row counts are non-zero and matching.
  - dependency status printed.

## 5) Fallback (Replay Mode)
If live stream fails, replay from session log:
1. Create (or use) session log:
   - `python src2/gen_code/create_mock_session_log.py --labels_csv src2/gen_code/demo_data/demo_labels.csv --output_csv src2/gen_code/demo_data/session_log.csv`
2. Run UI in replay mode:
   - `python src2/gen_code/demo_ui.py --labels_csv src2/gen_code/demo_data/demo_labels.csv --replay_csv src2/gen_code/demo_data/session_log.csv --host 0.0.0.0 --port 8050`

## 6) Demo-Day Quick Command Set
- `STATUS` -> check running/cursor.
- `START` -> begin inference stream.
- `STOP` -> pause stream.
- `RESET` -> reset to sample 0.
