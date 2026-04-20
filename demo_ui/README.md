# Device Benchmark Dashboard

## How to Run

python3 -m venv .venv                                             
source .venv/bin/activate
python3 -m pip install fastapi "uvicorn[standard]" websockets

Then run
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

---

Step 2: Open the dashboard

Go to:
http://localhost:8000

---

## Controls

Start Run:
- Begins the simulation

Reset:
- Stops the current run
- Clears all data

---
