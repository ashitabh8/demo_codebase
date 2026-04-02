# UI and Streaming Plan

## Scope
Create a browser-based Python dashboard that ingests Arduino and Raspberry Pi predictions, joins with ground truth labels, and displays dynamic rolling accuracy over the last 20 samples.

## Owner
Person B

## Dependencies
- `flask`, `pyserial`
- `src2/gen_code/compare_outputs.py` parsing patterns
- `src2/gen_code/demo_data/demo_labels.csv`

## Implementation Tasks
1. Add host app `src2/gen_code/demo_ui.py`.
2. Define unified line protocol:
   - `sample_<id> src=<arduino|rpi> pred=<int> logits=<comma_sep_optional> inf_ms=<float>`
3. Implement adapters:
   - Serial reader for Arduino
   - Serial/socket/file reader for Raspberry Pi
4. Implement in-memory state keyed by `sample_id` with GT lookup.
5. Add rolling windows (`N=20`) and compute:
   - Arduino accuracy vs GT
   - Raspberry Pi accuracy vs GT
6. Build UI:
   - Top-center rolling bars
   - Recent sample table/status text
7. Add CSV session logger for replay/fallback.

## Validation Commands
- `python src2/gen_code/demo_ui.py --help`
- `python src2/gen_code/demo_ui.py --labels_csv src2/gen_code/demo_data/demo_labels.csv --arduino_port /dev/ttyACM0 --rpi_source file:src2/gen_code/demo_data/rpi_mock_stream.txt`
- `python src2/gen_code/replay_streams.py --ui_input src2/gen_code/demo_data/session_log.csv`

## Exit Criteria
- UI runs locally and updates live without freezes.
- Rolling metrics update correctly on simulated and live streams.
- Missing/out-of-order sample IDs are handled gracefully.
- Session log can replay a full demo sequence.

## Handoff Artifacts
- UI app and stream adapters.
- Protocol spec and launch command examples.
