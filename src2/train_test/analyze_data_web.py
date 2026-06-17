"""
Web UI for cross-day analyze_data.py.

Run from repo root or train_test:
    python analyze_data_web.py --host 0.0.0.0 --port 8765

Requires: Flask (see environment_eugene.yml pip section).
"""

import argparse
import re
import sys
import uuid
from pathlib import Path

# src2 on path (same pattern as test.py)
src2_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src2_path))

from flask import Flask, abort, redirect, render_template_string, request, send_from_directory, url_for

from train_test.analyze_data import indices_from_split_dir, load_merged_indices, run_crossday_analysis

RUNS_DIR = Path(__file__).resolve().parent / "analyze_web_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

_JOB_ID_RE = re.compile(r"^[a-f0-9\-]{36}$")
_SAFE_NAME_RE = re.compile(r"^[0-9a-zA-Z_.\-]+\.(png|txt)$")


def _parse_paths_multiline(text):
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            lines.append(s)
    return lines


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Cross-day data analysis</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.35rem; }
    fieldset { margin: 1rem 0; border: 1px solid #ccc; padding: 1rem; }
    label { display: block; margin: 0.5rem 0 0.2rem; font-weight: 600; }
    input[type=text], textarea { width: 100%; box-sizing: border-box; font-family: monospace; font-size: 0.9rem; }
    textarea { min-height: 5rem; }
    .row { display: flex; gap: 1rem; flex-wrap: wrap; }
    .row > div { flex: 1 1 200px; }
    button { margin-top: 1rem; padding: 0.5rem 1.2rem; cursor: pointer; }
    .deps { background: #f5f5f5; padding: 0.75rem; font-size: 0.85rem; margin-top: 1.5rem; }
    .err { color: #b00020; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>Cross-day domain shift analysis</h1>
  <p>Each day uses <strong>train + val + test</strong> index files merged (deduplicated), same as the CLI split-dir mode.</p>

  <form method="post" action="{{ url_for('run_analysis') }}">
    <fieldset>
      <legend>Mode</legend>
      <label><input type="radio" name="mode" value="split_dir" {{ 'checked' if mode == 'split_dir' else '' }}/> Split directories (train_index.txt, val_index.txt, test_index.txt in each folder)</label>
      <label><input type="radio" name="mode" value="indices" {{ 'checked' if mode == 'indices' else '' }}/> Custom: paste full paths to index files (one per line per day)</label>
    </fieldset>

    <fieldset id="split_block">
      <legend>Split directories</legend>
      <label for="day1_split_dir">Day 1 split directory</label>
      <input type="text" id="day1_split_dir" name="day1_split_dir" value="{{ day1_split_dir }}"/>
      <label for="day2_split_dir">Day 2 split directory</label>
      <input type="text" id="day2_split_dir" name="day2_split_dir" value="{{ day2_split_dir }}"/>
    </fieldset>

    <fieldset id="idx_block">
      <legend>Index file paths (one per line)</legend>
      <label for="day1_indices_text">Day 1</label>
      <textarea id="day1_indices_text" name="day1_indices_text">{{ day1_indices_text }}</textarea>
      <label for="day2_indices_text">Day 2</label>
      <textarea id="day2_indices_text" name="day2_indices_text">{{ day2_indices_text }}</textarea>
    </fieldset>

    <fieldset>
      <legend>Analysis parameters</legend>
      <div class="row">
        <div><label for="max_per_class">max_per_class</label><input type="text" name="max_per_class" value="{{ max_per_class }}"/></div>
        <div><label for="sr">sr</label><input type="text" name="sr" value="{{ sr }}"/></div>
        <div><label for="n_mels">n_mels</label><input type="text" name="n_mels" value="{{ n_mels }}"/></div>
        <div><label for="mic_idx">mic_idx</label><input type="text" name="mic_idx" value="{{ mic_idx }}"/></div>
        <div><label for="seed">seed</label><input type="text" name="seed" value="{{ seed }}"/></div>
      </div>
      <label><input type="checkbox" name="no_umap" value="1" {{ 'checked' if no_umap else '' }}/> Skip UMAP (faster)</label>
    </fieldset>

    <button type="submit">Run analysis</button>
  </form>

  {% if error %}<p class="err">{{ error }}</p>{% endif %}

  <div class="deps">
    <strong>Dependencies</strong> (conda env): librosa, soundfile, umap-learn, Flask — listed in <code>environment_eugene.yml</code> pip section.
    Without librosa, spectral plots are skipped; without umap-learn, UMAP panel is skipped.
  </div>

  <script>
    function syncMode() {
      const m = document.querySelector('input[name="mode"]:checked').value;
      document.getElementById('split_block').style.display = m === 'split_dir' ? 'block' : 'none';
      document.getElementById('idx_block').style.display = m === 'indices' ? 'block' : 'none';
    }
    document.querySelectorAll('input[name="mode"]').forEach(function (el) { el.addEventListener('change', syncMode); });
    syncMode();
  </script>
</body>
</html>
"""

RESULT_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Analysis {{ job_id }}</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; }
    img { max-width: 100%; height: auto; border: 1px solid #ddd; margin: 0.5rem 0; }
    pre { background: #f5f5f5; padding: 1rem; overflow: auto; font-size: 0.85rem; }
  </style>
</head>
<body>
  <h1>Run {{ job_id }}</h1>
  <p><a href="{{ url_for('index') }}">New run</a></p>
  <h2>Summary</h2>
  <pre>{{ summary_text }}</pre>
  <h2>Figures</h2>
  {% for name in images %}
  <h3>{{ name }}</h3>
  <img src="{{ url_for('result_file', job_id=job_id, filename=name) }}" alt="{{ name }}"/>
  {% endfor %}
</body>
</html>
"""


@app.route("/")
def index():
    defaults = {
        "mode": "split_dir",
        "day1_split_dir": "/data/misra8/GracesQuarters/index_files/2024-08-06-GQ-split-multiclass",
        "day2_split_dir": "/data/misra8/GracesQuarters/index_files/2024-08-07-GQ-split-multiclass",
        "day1_indices_text": "",
        "day2_indices_text": "",
        "max_per_class": "120",
        "sr": "16000",
        "n_mels": "64",
        "mic_idx": "0",
        "seed": "42",
        "no_umap": False,
        "error": None,
    }
    return render_template_string(PAGE, **defaults)


@app.route("/run", methods=["POST"])
def run_analysis():
    mode = request.form.get("mode", "split_dir")
    try:
        max_per_class = int(request.form.get("max_per_class", "120"))
        sr = int(request.form.get("sr", "16000"))
        n_mels = int(request.form.get("n_mels", "64"))
        mic_idx = int(request.form.get("mic_idx", "0"))
        seed = int(request.form.get("seed", "42"))
    except ValueError:
        return _form_error("Integer fields must be valid numbers.", request.form)

    no_umap = request.form.get("no_umap") == "1"

    if mode == "split_dir":
        d1s = request.form.get("day1_split_dir", "").strip()
        d2s = request.form.get("day2_split_dir", "").strip()
        if not d1s or not d2s:
            return _form_error("Both split directories are required.", request.form)
        try:
            day1_paths = indices_from_split_dir(d1s)
            day2_paths = indices_from_split_dir(d2s)
        except FileNotFoundError as e:
            return _form_error(str(e), request.form)
    else:
        p1 = _parse_paths_multiline(request.form.get("day1_indices_text", ""))
        p2 = _parse_paths_multiline(request.form.get("day2_indices_text", ""))
        if not p1 or not p2:
            return _form_error("Paste at least one index path per line for each day.", request.form)
        try:
            load_merged_indices(p1)
            load_merged_indices(p2)
        except FileNotFoundError as e:
            return _form_error(str(e), request.form)
        day1_paths = p1
        day2_paths = p2

    job_id = str(uuid.uuid4())
    out_dir = RUNS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run_crossday_analysis(
        day1_paths,
        day2_paths,
        max_per_class=max_per_class,
        sr=sr,
        n_mels=n_mels,
        mic_idx=mic_idx,
        out_dir=str(out_dir),
        no_umap=no_umap,
        seed=seed,
    )
    return redirect(url_for("results", job_id=job_id))


def _form_error(msg, form):
    ctx = {
        "mode": form.get("mode", "split_dir"),
        "day1_split_dir": form.get("day1_split_dir", ""),
        "day2_split_dir": form.get("day2_split_dir", ""),
        "day1_indices_text": form.get("day1_indices_text", ""),
        "day2_indices_text": form.get("day2_indices_text", ""),
        "max_per_class": form.get("max_per_class", "120"),
        "sr": form.get("sr", "16000"),
        "n_mels": form.get("n_mels", "64"),
        "mic_idx": form.get("mic_idx", "0"),
        "seed": form.get("seed", "42"),
        "no_umap": form.get("no_umap") == "1",
        "error": msg,
    }
    return render_template_string(PAGE, **ctx), 400


@app.route("/results/<job_id>")
def results(job_id):
    if not _JOB_ID_RE.match(job_id):
        abort(404)
    base = RUNS_DIR / job_id
    if not base.is_dir():
        abort(404)
    summary_path = base / "00_summary.txt"
    summary_text = summary_path.read_text() if summary_path.exists() else "(no summary)"
    images = sorted(
        p.name
        for p in base.iterdir()
        if p.suffix.lower() == ".png" and p.is_file()
    )
    return render_template_string(RESULT_PAGE, job_id=job_id, summary_text=summary_text, images=images)


@app.route("/files/<job_id>/<filename>")
def result_file(job_id, filename):
    if not _JOB_ID_RE.match(job_id) or not _SAFE_NAME_RE.match(filename):
        abort(404)
    base = RUNS_DIR / job_id
    if not base.is_dir():
        abort(404)
    return send_from_directory(base, filename, as_attachment=False)


def main_web():
    parser = argparse.ArgumentParser(description="Web UI for analyze_data.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=False)


if __name__ == "__main__":
    main_web()
