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

TOTAL_SAMPLES = 100

# RPi about 50 percent faster, both with lower accuracy in the 70s to 80s
SIM_CONFIG = {
    "arduino": {
        "base_ms": 82.0,
        "jitter_ms": 12.0,
        "correct_prob": 0.78,
    },
    "rpi": {
        "base_ms": 55.0,
        "jitter_ms": 8.0,
        "correct_prob": 0.84,
    },
}


def fresh_device_state(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "samples": 0,
        "correct": 0,
        "latencies": [],
        "history": [],
        "done": False,
        "finish_time": None,
    }


def fresh_state() -> dict[str, Any]:
    return {
        "arduino": fresh_device_state("arduino"),
        "rpi": fresh_device_state("rpi"),
        "started": False,
        "over": False,
        "winner": None,
        "start_time": None,
    }


race = fresh_state()


def avg_latency_ms(latencies: list[float]) -> float:
    if not latencies:
        return 0.0
    return round(sum(latencies) / len(latencies) * 1000.0, 1)


def current_accuracy(samples: int, correct: int) -> float:
    if samples == 0:
        return 0.0
    return round((correct / samples) * 100.0, 2)


def elapsed_since_start() -> float:
    if race["start_time"] is None:
        return 0.0
    return round(time.time() - race["start_time"], 2)


def device_snapshot(device: dict[str, Any]) -> dict[str, Any]:
    return {
        "samples": device["samples"],
        "correct": device["correct"],
        "accuracy": current_accuracy(device["samples"], device["correct"]),
        "avg_latency_ms": avg_latency_ms(device["latencies"]),
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


async def run_racer(name: str) -> None:
    cfg = SIM_CONFIG[name]
    device = race[name]

    for _ in range(TOTAL_SAMPLES):
        delay_ms = max(5.0, cfg["base_ms"] + random.gauss(0.0, cfg["jitter_ms"]))
        t0 = time.time()
        await asyncio.sleep(delay_ms / 1000.0)
        latency_s = time.time() - t0

        is_correct = random.random() < cfg["correct_prob"]

        device["samples"] += 1
        if is_correct:
            device["correct"] += 1

        device["latencies"].append(latency_s)

        point = {
            "x": device["samples"],
            "y": current_accuracy(device["samples"], device["correct"]),
            "correct": bool(is_correct),
            "latency_ms": round(latency_s * 1000.0, 1),
            "timestamp": round(time.time(), 3),
        }
        device["history"].append(point)

        if device["samples"] >= TOTAL_SAMPLES and not device["done"]:
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


async def start_race() -> None:
    race["started"] = True
    race["over"] = False
    race["winner"] = None
    race["start_time"] = time.time()

    await broadcast(snapshot())

    await asyncio.gather(
        run_racer("arduino"),
        run_racer("rpi"),
    )

    finalize_race()
    await broadcast(snapshot())


@app.get("/")
async def index() -> HTMLResponse:
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    global race

    await ws.accept()
    clients.append(ws)
    await ws.send_text(json.dumps(snapshot()))

    try:
        async for raw in ws.iter_text():
            msg = json.loads(raw)
            action = msg.get("action")

            if action == "start" and not race["started"]:
                print("START RECEIVED")
                asyncio.create_task(start_race())

            elif action == "reset":
                print("RESET RECEIVED")
                race = fresh_state()
                await broadcast(snapshot())

    except WebSocketDisconnect:
        pass
    finally:
        if ws in clients:
            clients.remove(ws)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)