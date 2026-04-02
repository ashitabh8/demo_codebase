# Integration and Demo Runbook

## Scope
Define end-to-end integration checks and a reliable live-demo sequence with fallback paths.

## Owner
Both (Person A + Person B)

## Dependencies
- Data export outputs in `src2/gen_code/demo_data`
- Arduino firmware emitting stable protocol lines
- UI app `src2/gen_code/demo_ui.py`

## Integration Tasks
1. Confirm shared class map and sample IDs across all producers.
2. Run dry integration with replayed logs before live hardware.
3. Run Arduino live stream into UI and verify rolling metrics.
4. Run Raspberry Pi stream in parallel and verify synchronization.
5. Validate no sustained parser errors or UI stalls.
6. Record one full reference session log for fallback.

## Demo Day Startup Sequence
1. Connect Arduino GIGA R1 and verify serial port.
2. Start Raspberry Pi prediction stream.
3. Launch UI with labels CSV and both stream sources.
4. Send `START` command to Arduino.
5. Monitor rolling bars and recent sample panel.
6. If stream fails, switch UI to replay mode using saved session log.

## Validation Commands
- `python src2/gen_code/demo_healthcheck.py --labels_csv src2/gen_code/demo_data/demo_labels.csv`
- `python src2/gen_code/demo_ui.py --labels_csv src2/gen_code/demo_data/demo_labels.csv --arduino_port /dev/ttyACM0 --rpi_source serial:/dev/ttyUSB0`
- `python src2/gen_code/demo_ui.py --labels_csv src2/gen_code/demo_data/demo_labels.csv --replay_csv src2/gen_code/demo_data/session_log_reference.csv`

## Exit Criteria
- Live run stable for at least 5 minutes.
- Rolling accuracy bars for Arduino and Raspberry Pi update continuously.
- Sample ID alignment is correct across all three streams (Arduino, RPi, GT).
- Fallback replay path is verified and documented.

## Handoff Artifacts
- Final launch commands.
- Troubleshooting checklist.
- Reference session log for backup demo.
