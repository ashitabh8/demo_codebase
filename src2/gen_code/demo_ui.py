#!/usr/bin/env python3
"""Browser-based live dashboard for Arduino vs Raspberry Pi rolling accuracy."""

from __future__ import annotations

import argparse
import csv
import queue
import re
import socket
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from flask import Flask, jsonify, render_template_string
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Flask is required. Install with `pip install flask`.") from exc

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None


LINE_RE = re.compile(r"^sample_(\d+)\s+(.*)$")

_REPLAY_CSV_COLUMNS = ("sample_id", "source", "pred", "target", "inf_ms")

HTML_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Arduino vs Raspberry Pi Demo</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }
      .row { display: flex; gap: 24px; }
      .card { background: #111827; border-radius: 10px; padding: 16px; flex: 1; border: 1px solid #374151; }
      .label { font-size: 14px; margin-bottom: 10px; color: #93c5fd; }
      .bar-wrap { background: #1f2937; height: 36px; border-radius: 8px; overflow: hidden; }
      .bar { height: 100%; width: 0%; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; color: white; font-weight: bold; transition: width 200ms ease; }
      #arduino-bar { background: linear-gradient(90deg, #2563eb, #38bdf8); }
      #rpi-bar { background: linear-gradient(90deg, #16a34a, #4ade80); }
      .muted { color: #94a3b8; font-size: 13px; margin-top: 10px; }
      pre { background: #020617; border: 1px solid #334155; border-radius: 8px; padding: 12px; max-height: 340px; overflow-y: auto; white-space: pre-wrap; }
    </style>
  </head>
  <body>
    <h2>Live Rolling Accuracy (last <span id="wsize">20</span> samples)</h2>
    <div class="row">
      <div class="card">
        <div class="label">Arduino</div>
        <div class="bar-wrap"><div id="arduino-bar" class="bar">0%</div></div>
        <div id="arduino-meta" class="muted">0/0</div>
      </div>
      <div class="card">
        <div class="label">Raspberry Pi</div>
        <div class="bar-wrap"><div id="rpi-bar" class="bar">0%</div></div>
        <div id="rpi-meta" class="muted">0/0</div>
      </div>
    </div>
    <h3>Recent Events</h3>
    <pre id="events"></pre>
    <script>
      async function poll() {
        const res = await fetch('/api/state');
        const s = await res.json();
        document.getElementById('wsize').textContent = s.window_size;
        const aPct = Math.round(s.arduino.accuracy * 100);
        const rPct = Math.round(s.rpi.accuracy * 100);
        const aBar = document.getElementById('arduino-bar');
        const rBar = document.getElementById('rpi-bar');
        aBar.style.width = aPct + '%'; aBar.textContent = aPct + '%';
        rBar.style.width = rPct + '%'; rBar.textContent = rPct + '%';
        document.getElementById('arduino-meta').textContent = `${s.arduino.correct}/${s.arduino.total}`;
        document.getElementById('rpi-meta').textContent = `${s.rpi.correct}/${s.rpi.total}`;
        document.getElementById('events').textContent = s.recent_events.join('\\n');
      }
      setInterval(poll, 300);
      poll();
    </script>
  </body>
</html>
"""


@dataclass
class PredictionEvent:
    sample_id: int
    source: str
    pred: Optional[int]
    target: Optional[int]
    inf_ms: Optional[float]
    raw: str


def parse_prediction_line(line: str) -> Optional[PredictionEvent]:
    line = line.strip()
    m = LINE_RE.match(line)
    if not m:
        return None
    sample_id = int(m.group(1))
    payload = m.group(2).strip()
    fields = {}
    for token in payload.split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        fields[k] = v
    source = fields.get("src", "unknown")
    pred = int(fields["pred"]) if "pred" in fields else None
    target = int(fields["target"]) if "target" in fields else None
    inf_ms = float(fields["inf_ms"]) if "inf_ms" in fields else None
    return PredictionEvent(sample_id, source, pred, target, inf_ms, line)


def load_labels_csv(labels_csv: Path) -> dict[int, int]:
    labels = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels[int(row["sample_id"])] = int(row["target"])
    return labels


class DemoState:
    def __init__(self, labels: dict[int, int], window_size: int, session_csv: Optional[Path]) -> None:
        self.labels = labels
        self.window_size = window_size
        self.rolling_arduino = deque(maxlen=window_size)
        self.rolling_rpi = deque(maxlen=window_size)
        self.recent_events = deque(maxlen=40)
        self.events_missing_label = 0
        self.lock = threading.Lock()
        self.session_writer = None
        self.session_handle = None
        if session_csv is not None:
            session_csv.parent.mkdir(parents=True, exist_ok=True)
            self.session_handle = session_csv.open("w", encoding="utf-8", newline="")
            self.session_writer = csv.writer(self.session_handle)
            self.session_writer.writerow(
                ["timestamp", "sample_id", "source", "pred", "target", "correct", "inf_ms", "raw"]
            )

    def close(self) -> None:
        if self.session_handle is not None:
            self.session_handle.close()

    def consume(self, ev: PredictionEvent) -> None:
        if ev.pred is None:
            return
        if ev.sample_id not in self.labels:
            with self.lock:
                self.events_missing_label += 1
            return
        gt = self.labels[ev.sample_id]
        src = ev.source.lower()
        correct = int(ev.pred == gt)
        with self.lock:
            if src == "arduino":
                self.rolling_arduino.append(correct)
            elif src in ("rpi", "raspberrypi", "raspberry_pi"):
                self.rolling_rpi.append(correct)
            event_line = (
                f"sample_{ev.sample_id} src={ev.source} pred={ev.pred} gt={gt} "
                f"correct={correct} inf_ms={ev.inf_ms if ev.inf_ms is not None else 'na'}"
            )
            self.recent_events.appendleft(event_line)
            if self.session_writer is not None:
                self.session_writer.writerow(
                    [time.time(), ev.sample_id, ev.source, ev.pred, gt, correct, ev.inf_ms, ev.raw]
                )
                self.session_handle.flush()

    @staticmethod
    def _acc(values: deque[int]) -> float:
        return float(sum(values)) / float(len(values)) if values else 0.0

    def snapshot(self) -> dict:
        with self.lock:
            a_total = len(self.rolling_arduino)
            r_total = len(self.rolling_rpi)
            return {
                "window_size": self.window_size,
                "arduino": {
                    "accuracy": self._acc(self.rolling_arduino),
                    "correct": int(sum(self.rolling_arduino)),
                    "total": a_total,
                },
                "rpi": {
                    "accuracy": self._acc(self.rolling_rpi),
                    "correct": int(sum(self.rolling_rpi)),
                    "total": r_total,
                },
                "recent_events": list(self.recent_events),
                "events_missing_label": self.events_missing_label,
            }


def _reader_serial(port: str, baud: int, event_q: queue.Queue, stop_evt: threading.Event) -> None:
    if serial is None:
        raise RuntimeError("pyserial is required for serial input. Install with `pip install pyserial`.")
    with serial.Serial(port, baudrate=baud, timeout=0.2) as ser:
        while not stop_evt.is_set():
            raw = ser.readline().decode(errors="ignore").strip()
            if not raw:
                continue
            ev = parse_prediction_line(raw)
            if ev is not None:
                event_q.put(ev)


def _reader_file(
    path: Path,
    source_name: str,
    event_q: queue.Queue,
    stop_evt: threading.Event,
    delay_s: float,
) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if stop_evt.is_set():
                return
            line = raw.strip()
            if not line:
                continue
            ev = parse_prediction_line(line)
            if ev is not None:
                if ev.source == "unknown":
                    ev.source = source_name
                event_q.put(ev)
            if delay_s > 0:
                time.sleep(delay_s)


def _reader_socket(host: str, port: int, source_name: str, event_q: queue.Queue, stop_evt: threading.Event) -> None:
    with socket.create_connection((host, port), timeout=5.0) as conn:
        conn.settimeout(0.5)
        buf = ""
        while not stop_evt.is_set():
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                break
            buf += chunk.decode(errors="ignore")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                ev = parse_prediction_line(line)
                if ev is not None:
                    if ev.source == "unknown":
                        ev.source = source_name
                    event_q.put(ev)


def _start_source_reader(
    spec: str,
    default_source_name: str,
    event_q: queue.Queue,
    stop_evt: threading.Event,
    baud: int,
    file_delay_s: float,
) -> threading.Thread:
    if spec.startswith("serial:"):
        port = spec.split(":", 1)[1]
        target = _reader_serial
        args = (port, baud, event_q, stop_evt)
    elif spec.startswith("file:"):
        path = Path(spec.split(":", 1)[1])
        target = _reader_file
        args = (path, default_source_name, event_q, stop_evt, file_delay_s)
    elif spec.startswith("socket:"):
        hp = spec.split(":", 1)[1]
        host, port_s = hp.rsplit(":", 1)
        target = _reader_socket
        args = (host, int(port_s), default_source_name, event_q, stop_evt)
    else:
        raise ValueError(
            f"Invalid source spec '{spec}'. Expected serial:/dev/tty*, file:/path, or socket:host:port"
        )
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t


def _start_replay(
    replay_csv: Path,
    event_q: queue.Queue,
    stop_evt: threading.Event,
    replay_delay_s: float,
) -> threading.Thread:
    def _runner() -> None:
        try:
            with replay_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames:
                    reader.fieldnames = [h.strip() for h in reader.fieldnames]
                names = set(reader.fieldnames or [])
                missing = [c for c in _REPLAY_CSV_COLUMNS if c not in names]
                if missing:
                    print(
                        f"demo_ui replay: missing columns {missing} in {replay_csv.resolve()}. "
                        f"Found headers: {reader.fieldnames}",
                        file=sys.stderr,
                    )
                    return
                n = 0
                for row in reader:
                    if stop_evt.is_set():
                        return
                    ev = PredictionEvent(
                        sample_id=int(row["sample_id"]),
                        source=row["source"],
                        pred=int(row["pred"]),
                        target=int(row["target"]),
                        inf_ms=float(row["inf_ms"]) if row["inf_ms"] not in ("", "na", "None") else None,
                        raw=row["raw"] if "raw" in row and row["raw"] is not None else "",
                    )
                    event_q.put(ev)
                    n += 1
                    if replay_delay_s > 0:
                        time.sleep(replay_delay_s)
            print(f"demo_ui replay: pushed {n} events from {replay_csv.resolve()}", file=sys.stderr)
        except Exception:
            print(
                f"demo_ui replay: error while reading {replay_csv.resolve()} (cwd={Path.cwd()})",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return t


def _drain_events_forever(event_q: queue.Queue, state: DemoState, stop_evt: threading.Event) -> None:
    while not stop_evt.is_set():
        try:
            ev = event_q.get(timeout=0.3)
        except queue.Empty:
            continue
        if isinstance(ev, PredictionEvent):
            state.consume(ev)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser rolling-accuracy UI for Arduino and Raspberry Pi.")
    parser.add_argument("--labels_csv", type=Path, required=True)
    parser.add_argument("--arduino_source", type=str, default=None, help="serial:/dev/ttyACM0 or file:/path")
    parser.add_argument("--rpi_source", type=str, default=None, help="serial:/dev/ttyUSB0 or file:/path or socket:host:port")
    parser.add_argument("--arduino_port", type=str, default=None, help="Shortcut for --arduino_source serial:<port>")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--session_csv", type=Path, default=Path("src2/gen_code/demo_data/session_log.csv"))
    parser.add_argument("--replay_csv", type=Path, default=None, help="Replay a previous session log")
    parser.add_argument(
        "--replay_delay_s",
        type=float,
        default=1.0,
        help="Delay between replayed rows (seconds).",
    )
    parser.add_argument(
        "--file_delay_s",
        type=float,
        default=1.0,
        help="Delay between lines when reading file:* sources (seconds).",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Listen port (default 8765; use another if Address already in use).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    labels = load_labels_csv(args.labels_csv)
    print(f"demo_ui: loaded {len(labels)} rows from {args.labels_csv.resolve()} (cwd={Path.cwd()})", file=sys.stderr)
    event_q: queue.Queue = queue.Queue()
    stop_evt = threading.Event()
    state = DemoState(labels, args.window_size, args.session_csv if args.replay_csv is None else None)

    readers: list[threading.Thread] = []
    if args.replay_csv is not None:
        rp = args.replay_csv.resolve()
        if not args.replay_csv.is_file():
            print(f"demo_ui: replay_csv is not a file: {rp} (cwd={Path.cwd()})", file=sys.stderr)
        print(
            f"demo_ui: replay mode, labels={args.labels_csv.resolve()} replay={rp} delay_s={args.replay_delay_s}",
            file=sys.stderr,
        )
        readers.append(_start_replay(args.replay_csv, event_q, stop_evt, args.replay_delay_s))
    else:
        arduino_spec = args.arduino_source
        if arduino_spec is None and args.arduino_port is not None:
            arduino_spec = f"serial:{args.arduino_port}"
        if arduino_spec is not None:
            readers.append(
                _start_source_reader(
                    arduino_spec, "arduino", event_q, stop_evt, args.baud, args.file_delay_s
                )
            )
        if args.rpi_source is not None:
            readers.append(
                _start_source_reader(
                    args.rpi_source, "rpi", event_q, stop_evt, args.baud, args.file_delay_s
                )
            )

    aggregator = threading.Thread(target=_drain_events_forever, args=(event_q, state, stop_evt), daemon=True)
    aggregator.start()

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/api/state")
    def api_state():
        return jsonify(state.snapshot())

    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    finally:
        stop_evt.set()
        state.close()
        for t in readers:
            t.join(timeout=1.0)


if __name__ == "__main__":
    main()
