from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()
clients: list[WebSocket] = []
race_task: asyncio.Task[Any] | None = None

BATCH_DATA = [
    {"batch": 0, "n": 32, "accuracy": 31.25, "forward_ms": 75.65},
    {"batch": 1, "n": 32, "accuracy": 25.00, "forward_ms": 1.36},
    {"batch": 2, "n": 32, "accuracy": 53.12, "forward_ms": 1.08},
    {"batch": 3, "n": 32, "accuracy": 40.62, "forward_ms": 1.08},
    {"batch": 4, "n": 32, "accuracy": 56.25, "forward_ms": 1.11},
    {"batch": 5, "n": 32, "accuracy": 37.50, "forward_ms": 1.90},
    {"batch": 6, "n": 32, "accuracy": 37.50, "forward_ms": 1.85},
    {"batch": 7, "n": 32, "accuracy": 46.88, "forward_ms": 1.66},
    {"batch": 8, "n": 32, "accuracy": 31.25, "forward_ms": 2.17},
    {"batch": 9, "n": 32, "accuracy": 56.25, "forward_ms": 1.75},
    {"batch": 10, "n": 32, "accuracy": 53.12, "forward_ms": 1.98},
    {"batch": 11, "n": 32, "accuracy": 31.25, "forward_ms": 1.15},
    {"batch": 12, "n": 32, "accuracy": 43.75, "forward_ms": 1.08},
    {"batch": 13, "n": 32, "accuracy": 40.62, "forward_ms": 2.20},
    {"batch": 14, "n": 32, "accuracy": 37.50, "forward_ms": 2.12},
    {"batch": 15, "n": 32, "accuracy": 53.12, "forward_ms": 1.85},
    {"batch": 16, "n": 32, "accuracy": 50.00, "forward_ms": 1.89},
    {"batch": 17, "n": 32, "accuracy": 40.62, "forward_ms": 2.36},
    {"batch": 18, "n": 32, "accuracy": 53.12, "forward_ms": 1.94},
    {"batch": 19, "n": 32, "accuracy": 31.25, "forward_ms": 1.15},
    {"batch": 20, "n": 32, "accuracy": 53.12, "forward_ms": 1.39},
    {"batch": 21, "n": 32, "accuracy": 53.12, "forward_ms": 1.68},
    {"batch": 22, "n": 32, "accuracy": 43.75, "forward_ms": 1.82},
    {"batch": 23, "n": 32, "accuracy": 46.88, "forward_ms": 1.94},
    {"batch": 24, "n": 32, "accuracy": 37.50, "forward_ms": 1.50},
    {"batch": 25, "n": 32, "accuracy": 37.50, "forward_ms": 1.96},
    {"batch": 26, "n": 32, "accuracy": 37.50, "forward_ms": 2.13},
    {"batch": 27, "n": 32, "accuracy": 62.50, "forward_ms": 1.62},
    {"batch": 28, "n": 32, "accuracy": 34.38, "forward_ms": 1.22},
    {"batch": 29, "n": 32, "accuracy": 34.38, "forward_ms": 1.94},
    {"batch": 30, "n": 32, "accuracy": 40.62, "forward_ms": 1.67},
    {"batch": 31, "n": 32, "accuracy": 31.25, "forward_ms": 1.25},
    {"batch": 32, "n": 32, "accuracy": 43.75, "forward_ms": 1.76},
    {"batch": 33, "n": 32, "accuracy": 56.25, "forward_ms": 2.33},
    {"batch": 34, "n": 32, "accuracy": 37.50, "forward_ms": 1.97},
    {"batch": 35, "n": 32, "accuracy": 34.38, "forward_ms": 1.12},
    {"batch": 36, "n": 32, "accuracy": 43.75, "forward_ms": 1.61},
    {"batch": 37, "n": 32, "accuracy": 46.88, "forward_ms": 1.67},
    {"batch": 38, "n": 32, "accuracy": 34.38, "forward_ms": 1.26},
    {"batch": 39, "n": 32, "accuracy": 43.75, "forward_ms": 1.07},
    {"batch": 40, "n": 32, "accuracy": 50.00, "forward_ms": 1.30},
    {"batch": 41, "n": 32, "accuracy": 37.50, "forward_ms": 1.49},
    {"batch": 42, "n": 32, "accuracy": 37.50, "forward_ms": 1.91},
    {"batch": 43, "n": 32, "accuracy": 37.50, "forward_ms": 1.21},
    {"batch": 44, "n": 32, "accuracy": 43.75, "forward_ms": 1.79},
    {"batch": 45, "n": 32, "accuracy": 34.38, "forward_ms": 1.07},
    {"batch": 46, "n": 21, "accuracy": 23.81, "forward_ms": 4.45},
]

TOTAL_BATCHES = len(BATCH_DATA)
TOTAL_SAMPLES = sum(item["n"] for item in BATCH_DATA)
ARDUINO_VISUAL_EXTRA_MS = 2000.0
RPI_VISUAL_EXTRA_MS = 1000.0


def fresh_device_state(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "batches": 0,
        "processed_samples": 0,
        "correct": 0,
        "current_accuracy": 0.0,
        "current_latency_ms": 0.0,
        "history": [],
        "done": False,
        "finish_time": None,
    }


def fresh_state(race_id: int) -> dict[str, Any]:
    return {
        "race_id": race_id,
        "arduino": fresh_device_state("arduino"),
        "rpi": fresh_device_state("rpi"),
        "started": False,
        "over": False,
        "winner": None,
        "start_time": None,
    }


race = fresh_state(0)


def elapsed_since_start() -> float:
    if race["start_time"] is None:
        return 0.0
    return round(time.time() - race["start_time"], 2)


def clamp_percent(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def device_snapshot(device: dict[str, Any]) -> dict[str, Any]:
    return {
        "batches": device["batches"],
        "processed_samples": device["processed_samples"],
        "correct": device["correct"],
        "accuracy": round(device["current_accuracy"], 2),
        "current_latency_ms": round(device["current_latency_ms"], 2),
        "done": device["done"],
        "finish_time": device["finish_time"],
        "history": device["history"],
    }


def snapshot() -> dict[str, Any]:
    return {
        "type": "state",
        "started": race["started"],
        "over": race["over"],
        "winner": race["winner"],
        "total_batches": TOTAL_BATCHES,
        "total_samples": TOTAL_SAMPLES,
        "elapsed": elapsed_since_start(),
        "racers": {
            "arduino": device_snapshot(race["arduino"]),
            "rpi": device_snapshot(race["rpi"]),
        },
    }


async def broadcast(payload: dict[str, Any]) -> None:
    dead: list[WebSocket] = []
    msg = json.dumps(payload)

    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)

    for ws in dead:
        if ws in clients:
            clients.remove(ws)


async def run_racer(name: str, race_id: int, rng: random.Random | None = None) -> None:
    device = race[name]

    for item in BATCH_DATA:
        if race["race_id"] != race_id:
            return

        raw_accuracy = item["accuracy"]
        raw_latency_ms = item["forward_ms"]

        if name == "arduino":
            display_accuracy = raw_accuracy
            visual_delay_ms = raw_latency_ms + ARDUINO_VISUAL_EXTRA_MS
        else:
            assert rng is not None
            display_accuracy = clamp_percent(raw_accuracy + 5.0 + rng.uniform(-2.0, 2.0))
            visual_delay_ms = raw_latency_ms + RPI_VISUAL_EXTRA_MS

        await asyncio.sleep(visual_delay_ms / 1000.0)

        if race["race_id"] != race_id:
            return

        batch_correct = round(item["n"] * display_accuracy / 100.0)

        device["batches"] += 1
        device["processed_samples"] += item["n"]
        device["correct"] += batch_correct
        device["current_accuracy"] = display_accuracy
        device["current_latency_ms"] = raw_latency_ms

        point = {
            "x": device["batches"],
            "y": display_accuracy,
            "batch": item["batch"],
            "n": item["n"],
            "correct": batch_correct,
            "latency_ms": round(raw_latency_ms, 2),
            "timestamp": round(time.time(), 3),
        }
        device["history"].append(point)

        if device["batches"] >= TOTAL_BATCHES and not device["done"]:
            device["done"] = True
            if race["start_time"] is not None:
                device["finish_time"] = round(time.time() - race["start_time"], 2)

        await broadcast(snapshot())


def finalize_race() -> None:
    ard_finish = race["arduino"]["finish_time"]
    rpi_finish = race["rpi"]["finish_time"]

    if ard_finish is None or rpi_finish is None:
        return

    if ard_finish < rpi_finish:
        race["winner"] = "arduino"
    elif rpi_finish < ard_finish:
        race["winner"] = "rpi"
    else:
        race["winner"] = "tie"

    race["over"] = True


async def start_race(race_id: int) -> None:
    try:
        race["started"] = True
        race["over"] = False
        race["winner"] = None
        race["start_time"] = time.time()

        await broadcast(snapshot())

        rpi_rng = random.Random(1000 + race_id)

        await asyncio.gather(
            run_racer("arduino", race_id),
            run_racer("rpi", race_id, rpi_rng),
        )

        if race["race_id"] != race_id:
            return

        finalize_race()
        await broadcast(snapshot())
    except asyncio.CancelledError:
        pass


@app.get("/")
async def index() -> HTMLResponse:
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    global race
    global race_task

    await ws.accept()
    clients.append(ws)
    await ws.send_text(json.dumps(snapshot()))

    try:
        async for raw in ws.iter_text():
            msg = json.loads(raw)
            action = msg.get("action")

            if action == "start" and not race["started"]:
                race_task = asyncio.create_task(start_race(race["race_id"]))

            elif action == "reset":
                if race_task is not None and not race_task.done():
                    race_task.cancel()
                race = fresh_state(race["race_id"] + 1)
                await broadcast(snapshot())

    except WebSocketDisconnect:
        pass
    finally:
        if ws in clients:
            clients.remove(ws)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
