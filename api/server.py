#!/usr/bin/env python3
"""
server.py — FastAPI backend for Smart Signal dashboard.

Endpoints:
  GET  /                     → serve dashboard
  GET  /api/status           → training status, Q-table stats, best params
  GET  /api/results          → evaluation results JSON
  GET  /api/training-log     → full training log for charts
  POST /api/detect           → upload image, get YOLO density
  GET  /api/policy/{state}   → get agent's action for a given density state
  POST /api/train/start      → kick off background training run
  GET  /api/train/progress   → streaming training progress (SSE)

Run:
  uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
"""

import asyncio
import json
import logging
import pathlib
import sys
import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Signal API",
    description="Adaptive Traffic Signal Control using Q-Learning + YOLOv8",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static dashboard
DASHBOARD_DIR = ROOT / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

RESULTS_DIR    = ROOT / "results"
Q_TABLES_DIR   = ROOT / "results" / "q_tables"

GREEN_PHASE_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# ---------------------------------------------------------------------------
# In-memory training state (shared with background task)
# ---------------------------------------------------------------------------
training_state = {
    "running":   False,
    "episode":   0,
    "total":     0,
    "reward":    0.0,
    "wait":      0.0,
    "epsilon":   0.0,
    "td_error":  0.0,
    "log":       [],
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index = DASHBOARD_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>Smart Signal API running</h1>"
                        "<p>Dashboard not found at dashboard/index.html</p>")


@app.get("/api/status")
async def get_status():
    """Return current system status."""
    q_path = Q_TABLES_DIR / "q_learning_best.npy"
    has_model = q_path.exists()

    best_params = {}
    for agent in ["q_learning", "sarsa"]:
        bp = RESULTS_DIR / f"best_params_{agent}.json"
        if bp.exists():
            best_params[agent] = json.loads(bp.read_text())

    status = {
        "model_trained":   has_model,
        "training_running": training_state["running"],
        "training_episode": training_state["episode"],
        "best_params":     best_params,
    }

    if has_model:
        try:
            Q = np.load(str(q_path))
            status["q_table_stats"] = {
                "shape":       list(Q.shape),
                "nonzero_pct": round(float(np.mean(Q != 0) * 100), 1),
                "max_q":       round(float(np.max(Q)), 3),
                "mean_q":      round(float(np.mean(Q)), 3),
            }
        except Exception:
            pass

    return JSONResponse(status)


@app.get("/api/results")
async def get_results():
    """Return evaluation results (Fixed / SARSA / Q-learning comparison)."""
    rpath = RESULTS_DIR / "evaluation_results.json"
    if not rpath.exists():
        raise HTTPException(404, detail="No evaluation results. Run python train.py first.")
    return JSONResponse(json.loads(rpath.read_text()))


@app.get("/api/training-log/{agent_name}")
async def get_training_log(agent_name: str):
    """Return training log for a specific agent (q_learning or sarsa)."""
    log_path = RESULTS_DIR / f"training_log_{agent_name}.json"
    if not log_path.exists():
        raise HTTPException(404, detail=f"No log for {agent_name}")
    return JSONResponse(json.loads(log_path.read_text()))


@app.get("/api/policy/{state}")
async def get_policy(state: int):
    """Return the trained agent's action for a given traffic density state."""
    if not 1 <= state <= 100:
        raise HTTPException(400, detail="State must be in [1, 100]")

    q_path = Q_TABLES_DIR / "q_learning_best.npy"
    if not q_path.exists():
        raise HTTPException(404, detail="Model not trained yet")

    Q          = np.load(str(q_path))
    action_idx = int(np.argmax(Q[state]))
    duration   = GREEN_PHASE_OPTIONS[action_idx]
    value      = float(np.max(Q[state]))

    return JSONResponse({
        "state":    state,
        "action":   action_idx,
        "duration": duration,
        "value":    round(value, 4),
        "q_values": [round(float(q), 4) for q in Q[state]],
        "all_actions": GREEN_PHASE_OPTIONS,
    })


@app.post("/api/detect")
async def detect_vehicles(file: UploadFile = File(...)):
    """
    Run YOLOv8 on uploaded image.
    Returns vehicle counts, weighted density, and RL state.
    """
    try:
        import cv2
        data    = await file.read()
        arr     = np.frombuffer(data, np.uint8)
        img     = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        from detection.yolo_detector import YOLODetector
        detector = YOLODetector()
        result   = detector.detect(img)

        # Encode annotated image as JPEG for response
        _, buf = cv2.imencode(".jpg", result["annotated"])
        img_b64 = __import__("base64").b64encode(buf.tobytes()).decode()

        return JSONResponse({
            "counts":         result["counts"],
            "raw_density":    result["raw"],
            "density":        result["density"],
            "num_detections": result["num_detections"],
            "annotated_b64":  img_b64,
        })
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/api/train/start")
async def start_training(background_tasks: BackgroundTasks,
                         episodes: int = 500,
                         tune: bool = False,
                         no_sumo: bool = True):
    """Start a background training run."""
    if training_state["running"]:
        return JSONResponse({"status": "already_running"})

    background_tasks.add_task(_run_training, episodes, tune, no_sumo)
    return JSONResponse({"status": "started", "episodes": episodes})


@app.get("/api/train/progress")
async def stream_progress():
    """Server-Sent Events stream of training progress."""
    async def event_generator():
        last_ep = 0
        while training_state["running"] or training_state["episode"] > last_ep:
            ep = training_state["episode"]
            if ep != last_ep:
                data = json.dumps({
                    "episode": ep,
                    "total":   training_state["total"],
                    "reward":  training_state["reward"],
                    "wait":    training_state["wait"],
                    "epsilon": training_state["epsilon"],
                })
                yield f"data: {data}\n\n"
                last_ep = ep
            await asyncio.sleep(0.5)
        yield 'data: {"done": true}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/plot/{name}")
async def serve_plot(name: str):
    """Serve a generated plot image."""
    allowed = ["training_curves", "comparison_bar", "q_table_heatmap",
               "epsilon_decay", "td_error", "yolo_demo"]
    if name not in allowed:
        raise HTTPException(400, detail="Unknown plot name")
    path = RESULTS_DIR / "plots" / f"{name}.png"
    if name == "yolo_demo":
        path = RESULTS_DIR / "yolo_demo.png"
    if not path.exists():
        raise HTTPException(404, detail=f"Plot '{name}' not found. Run training first.")
    return FileResponse(str(path), media_type="image/png")


# ---------------------------------------------------------------------------
# Background training task
# ---------------------------------------------------------------------------

async def _run_training(episodes: int, tune: bool, no_sumo: bool):
    """Background task wrapper for training."""
    import subprocess, sys
    training_state["running"] = True
    training_state["total"]   = episodes
    training_state["episode"] = 0

    cmd = [sys.executable, str(ROOT / "train.py"),
           "--episodes", str(episodes)]
    if tune:
        cmd.append("--tune")
    if no_sumo:
        cmd.append("--no-sumo")

    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    async for line in proc.stdout:
        text = line.decode().strip()
        # Parse episode line: "  Ep  42/500 | reward= +3.0 | wait= 18.2s ..."
        if "Ep " in text and "reward=" in text:
            try:
                parts = text.split("|")
                ep_str = parts[0].split("Ep")[1].strip().split("/")[0]
                training_state["episode"] = int(ep_str.strip())
                for part in parts[1:]:
                    if "reward=" in part:
                        training_state["reward"] = float(part.split("=")[1].strip())
                    if "wait=" in part:
                        training_state["wait"] = float(
                            part.split("=")[1].strip().replace("s",""))
                    if "ε=" in part or "epsilon=" in part:
                        training_state["epsilon"] = float(part.split("=")[1].strip())
            except Exception:
                pass

    await proc.wait()
    training_state["running"] = False
