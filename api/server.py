#!/usr/bin/env python3
"""
server.py — FastAPI backend for Smart Signal dashboard.

Endpoints:
  GET  /                        → serve dashboard
  GET  /api/status              → training status, Q-table stats, best params
  GET  /api/results             → evaluation results JSON
  GET  /api/training-log        → full training log for charts
  POST /api/detect              → upload image, get YOLO density + RL action + Q-table context
  POST /api/simulate-outcome    → simulate what happens when an action is applied at a state
  POST /api/feedback            → provide before/after images to update Q-table online (returns diff)
  GET  /api/policy/{state}      → get agent's action for a given density state
  GET  /api/q-table-snapshot    → full Q-table as JSON (for live heatmap)
  POST /api/train/start         → kick off background training run
  GET  /api/train/progress      → streaming training progress (SSE)
  GET  /api/decision-history    → session-level log of all decisions

Run:
  uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
"""

import asyncio
import json
import logging
import pathlib
import sys
import io
import time
import numpy as np
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Signal API",
    description="Adaptive Traffic Signal Control using Q-Learning + YOLOv8",
    version="3.0.0",
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

RESULTS_DIR  = ROOT / "results"
Q_TABLES_DIR = ROOT / "results" / "q_tables"

GREEN_PHASE_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# ---------------------------------------------------------------------------
# In-memory training state (shared with background task)
# ---------------------------------------------------------------------------
training_state = {
    "running":  False,
    "episode":  0,
    "total":    0,
    "reward":   0.0,
    "wait":     0.0,
    "epsilon":  0.0,
    "td_error": 0.0,
    "log":      [],
}

# ---------------------------------------------------------------------------
# In-memory session decision history
# ---------------------------------------------------------------------------
decision_history: List[dict] = []

# ---------------------------------------------------------------------------
# Live Q-learning agent instance for online learning
# ---------------------------------------------------------------------------
_live_agent = None
_agent_lock = asyncio.Lock()


def _get_or_load_agent():
    """Load (or return cached) the Q-learning agent for online inference + updates."""
    global _live_agent
    if _live_agent is not None:
        return _live_agent

    from agents.q_learning_agent import QLearningAgent
    agent = QLearningAgent(alpha=0.05, gamma=0.9, epsilon=0.0, epsilon_min=0.0)

    q_path = Q_TABLES_DIR / "q_learning_best.npy"
    if q_path.exists():
        try:
            agent.load(str(Q_TABLES_DIR / "q_learning_best"))
            agent.epsilon = 0.0   # pure greedy at inference
            logger.info("[Agent] Loaded pre-trained Q-table from disk")
        except Exception as e:
            logger.warning(f"[Agent] Could not load Q-table: {e} — starting fresh")
    else:
        logger.info("[Agent] No pre-trained Q-table found — using zero-initialized table")

    _live_agent = agent
    return _live_agent


def _get_q_table_context(agent, state: int, radius: int = 5) -> dict:
    """
    Extract Q-table context around a given state for visualization.
    Returns Q-values for states [state-radius ... state+radius] and metadata.
    """
    lo = max(1, state - radius)
    hi = min(100, state + radius)
    states_range = list(range(lo, hi + 1))

    rows = {}
    for s in states_range:
        q_row = [round(float(v), 4) for v in agent.Q[s]]
        rows[str(s)] = q_row

    # Best action per state in context
    best_actions = {}
    for s in states_range:
        best_a = int(np.argmax(agent.Q[s]))
        best_actions[str(s)] = {
            "action_idx": best_a,
            "duration": GREEN_PHASE_OPTIONS[best_a],
            "q_value": round(float(np.max(agent.Q[s])), 4),
        }

    return {
        "state_range": [lo, hi],
        "current_state": state,
        "q_rows": rows,
        "best_actions": best_actions,
        "actions": GREEN_PHASE_OPTIONS,
    }


def _simulate_next_state(density: int, action_idx: int) -> dict:
    """
    Simulate what happens after applying an action at a given traffic density.
    Returns dict with next_density, reward, reward_breakdown, success.
    """
    rng = np.random.default_rng(int(time.time() * 1000) % (2**32))
    green_duration = GREEN_PHASE_OPTIONS[action_idx]

    density_start = density

    cleared   = int(rng.poisson(green_duration * 0.4))
    arrivals  = int(rng.poisson(green_duration * 0.3))
    mid_density = max(1, density_start - cleared // 2 + int(rng.poisson(3)))
    next_density = max(1, min(100,
        density_start - cleared + arrivals + int(rng.normal(0, 3))
    ))

    reward = 0.0
    reward_breakdown = []

    midpoint_bonus = False
    if density_start > 0 and mid_density < density_start / 3:
        reward += 2.0
        midpoint_bonus = True
        reward_breakdown.append({"component": "Fast clearance bonus", "value": +2.0})

    if next_density < density_start:
        reward += 1.0
        reward_breakdown.append({"component": "Density decreased", "value": +1.0})
    else:
        reward -= 1.0
        reward_breakdown.append({"component": "Density increased/unchanged", "value": -1.0})

    success = next_density < density_start
    density_change = next_density - density_start
    wait_reduction_pct = max(0.0, round((density_start - next_density) / max(density_start, 1) * 100, 1))

    return {
        "density_before":     density_start,
        "density_after":      next_density,
        "density_change":     density_change,
        "midpoint_density":   mid_density,
        "midpoint_bonus":     midpoint_bonus,
        "reward":             round(reward, 2),
        "reward_breakdown":   reward_breakdown,
        "success":            success,
        "wait_reduction_pct": wait_reduction_pct,
        "green_duration":     green_duration,
        "vehicles_cleared":   cleared,
        "vehicles_arrived":   arrivals,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index = DASHBOARD_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Smart Signal API running</h1>"
                        "<p>Dashboard not found at dashboard/index.html</p>")


@app.get("/api/status")
async def get_status():
    """Return current system status."""
    q_path = Q_TABLES_DIR / "q_learning_best.npy"
    has_model = q_path.exists()

    best_params = {}
    bp = RESULTS_DIR / "best_params_q_learning.json"
    if bp.exists():
        best_params["q_learning"] = json.loads(bp.read_text())

    # Online learning stats
    agent = _get_or_load_agent()
    online_updates = agent.total_updates if agent else 0

    status = {
        "model_trained":     has_model,
        "training_running":  training_state["running"],
        "training_episode":  training_state["episode"],
        "best_params":       best_params,
        "online_updates":    online_updates,
        "decisions_made":    len(decision_history),
        "agent_config": agent.get_config() if agent else {},
    }

    if has_model:
        try:
            Q = np.load(str(q_path))
            status["q_table_stats"] = {
                "shape":       list(Q.shape),
                "nonzero_pct": round(float(np.mean(Q != 0) * 100), 1),
                "max_q":       round(float(np.max(Q)), 3),
                "min_q":       round(float(np.min(Q)), 3),
                "mean_q":      round(float(np.mean(Q)), 3),
                "std_q":       round(float(np.std(Q)), 3),
            }
        except Exception:
            pass

    return JSONResponse(status)


@app.get("/api/results")
async def get_results():
    """Return evaluation results (Fixed / Q-learning comparison)."""
    rpath = RESULTS_DIR / "evaluation_results.json"
    if not rpath.exists():
        raise HTTPException(404, detail="No evaluation results. Run python train.py first.")
    return JSONResponse(json.loads(rpath.read_text()))


@app.get("/api/training-log/{agent_name}")
async def get_training_log(agent_name: str):
    """Return training log for a specific agent."""
    log_path = RESULTS_DIR / f"training_log_{agent_name}.json"
    if not log_path.exists():
        raise HTTPException(404, detail=f"No log for {agent_name}")
    return JSONResponse(json.loads(log_path.read_text()))


@app.get("/api/policy/{state}")
async def get_policy(state: int):
    """Return the trained agent's action for a given traffic density state."""
    if not 1 <= state <= 100:
        raise HTTPException(400, detail="State must be in [1, 100]")

    agent = _get_or_load_agent()
    action_idx = int(np.argmax(agent.Q[state]))
    duration   = GREEN_PHASE_OPTIONS[action_idx]
    value      = float(np.max(agent.Q[state]))

    return JSONResponse({
        "state":       state,
        "action":      action_idx,
        "duration":    duration,
        "value":       round(value, 4),
        "q_values":    [round(float(q), 4) for q in agent.Q[state]],
        "all_actions": GREEN_PHASE_OPTIONS,
    })


@app.get("/api/q-table-snapshot")
async def get_q_table_snapshot():
    """
    Return the full Q-table as a 2D list for live heatmap rendering.
    States 1-100, 10 actions each.
    """
    agent = _get_or_load_agent()
    # Q-table is (101, 10), we return states 1-100
    q_data = agent.Q[1:101, :].tolist()  # 100 x 10

    # Also return the greedy policy (best action per state)
    policy = []
    for s in range(1, 101):
        best_a = int(np.argmax(agent.Q[s]))
        policy.append({
            "state": s,
            "action_idx": best_a,
            "duration": GREEN_PHASE_OPTIONS[best_a],
            "q_value": round(float(np.max(agent.Q[s])), 4),
        })

    return JSONResponse({
        "q_table": q_data,
        "num_states": 100,
        "num_actions": 10,
        "actions": GREEN_PHASE_OPTIONS,
        "policy": policy,
        "stats": {
            "nonzero_pct": round(float(np.mean(agent.Q[1:101] != 0) * 100), 1),
            "max_q": round(float(np.max(agent.Q[1:101])), 4),
            "min_q": round(float(np.min(agent.Q[1:101])), 4),
            "mean_q": round(float(np.mean(agent.Q[1:101])), 4),
        },
        "hyperparams": {
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "epsilon": agent.epsilon,
        },
    })


@app.post("/api/detect")
async def detect_vehicles(file: UploadFile = File(...)):
    """
    Run YOLOv8 on uploaded image.
    Returns vehicle counts, weighted density, RL state, recommended action,
    Q-table context around the detected state, and simulated outcome.
    """
    try:
        import cv2
        data = await file.read()
        arr  = np.frombuffer(data, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        from detection.yolo_detector import YOLODetector
        detector = YOLODetector()
        result   = detector.detect(img)

        density    = result["density"]
        state      = density

        # ── RL Agent Decision ────────────────────────────────────────────
        agent      = _get_or_load_agent()
        action_idx = agent.greedy_action(state)
        duration   = GREEN_PHASE_OPTIONS[action_idx]
        q_values   = [round(float(q), 4) for q in agent.Q[state]]
        q_value    = float(np.max(agent.Q[state]))

        # ── Q-Table Context (for mini-heatmap) ───────────────────────────
        q_context = _get_q_table_context(agent, state, radius=5)

        # ── Policy Explanation ────────────────────────────────────────────
        sorted_actions = sorted(enumerate(agent.Q[state]),
                                key=lambda x: x[1], reverse=True)
        explanation_parts = []
        explanation_parts.append(
            f"At state s={state} (density={density}/100), "
            f"the agent selects action a={action_idx} "
            f"(green phase={duration}s) because Q({state},{action_idx})="
            f"{q_value:.4f} is the maximum Q-value."
        )
        if sorted_actions[0][1] == sorted_actions[1][1]:
            explanation_parts.append(
                "Note: multiple actions have equal Q-values (tie-breaking by index)."
            )
        second_best = sorted_actions[1]
        explanation_parts.append(
            f"Second-best: action {second_best[0]} "
            f"({GREEN_PHASE_OPTIONS[second_best[0]]}s) with "
            f"Q={second_best[1]:.4f}."
        )

        # ── Simulate Outcome ─────────────────────────────────────────────
        outcome = _simulate_next_state(state, action_idx)

        # ── MDP Tuple ─────────────────────────────────────────────────────
        mdp_tuple = {
            "S": state,
            "A": action_idx,
            "A_label": f"{duration}s green",
            "R": outcome["reward"],
            "S_prime": outcome["density_after"],
            "done": False,
            "max_Q_S_prime": round(float(np.max(agent.Q[outcome["density_after"]])), 4),
        }

        # ── Save to Decision History ─────────────────────────────────────
        record = {
            "id":          len(decision_history) + 1,
            "timestamp":   datetime.now().isoformat(timespec="seconds"),
            "filename":    file.filename or "uploaded_image",
            "density":     density,
            "action_idx":  action_idx,
            "duration":    duration,
            "q_value":     round(q_value, 4),
            "outcome":     outcome,
            "counts":      result["counts"],
        }
        decision_history.append(record)

        # ── Encode annotated image ───────────────────────────────────────
        _, buf    = cv2.imencode(".jpg", result["annotated"])
        img_b64   = __import__("base64").b64encode(buf.tobytes()).decode()

        return JSONResponse({
            # Detection results
            "counts":         result["counts"],
            "raw_density":    result["raw"],
            "density":        density,
            "num_detections": result["num_detections"],
            "annotated_b64":  img_b64,
            # RL decision
            "rl_action": {
                "action_idx": action_idx,
                "duration":   duration,
                "q_value":    round(q_value, 4),
                "q_values":   q_values,
                "all_actions": GREEN_PHASE_OPTIONS,
                "explanation": " ".join(explanation_parts),
            },
            # Q-table context around detected state
            "q_table_context": q_context,
            # MDP transition tuple
            "mdp_tuple": mdp_tuple,
            # Simulated outcome
            "outcome": outcome,
            # History
            "decision_id": record["id"],
        })

    except Exception as e:
        logger.exception("Detection error")
        raise HTTPException(500, detail=str(e))


@app.post("/api/simulate-outcome")
async def simulate_outcome(state: int, action_idx: int):
    """Simulate what happens when a given action is applied at a given state."""
    if not 1 <= state <= 100:
        raise HTTPException(400, detail="State must be in [1, 100]")
    if not 0 <= action_idx <= 9:
        raise HTTPException(400, detail="action_idx must be in [0, 9]")

    outcome = _simulate_next_state(state, action_idx)
    return JSONResponse(outcome)


@app.post("/api/feedback")
async def provide_feedback(
    before_file: UploadFile = File(...),
    after_file:  UploadFile = File(...),
    action_idx:  int        = Form(...),
):
    """
    Online Q-table update from real before/after images.
    
    Returns the Q-table ROW BEFORE and AFTER the update so the frontend
    can show exactly how the agent learned from this transition.

    Workflow:
      1. Run YOLO on 'before' image → get state s
      2. Run YOLO on 'after' image  → get next_state s'
      3. Calculate reward from s → (action) → s'
      4. Snapshot Q-row BEFORE update
      5. Apply Q-learning update: Q(s,a) ← Q(s,a) + α·[R + γ·max Q(s',a') − Q(s,a)]
      6. Snapshot Q-row AFTER update
      7. Save updated Q-table to disk
      8. Return both snapshots + diff + Bellman equation with real numbers
    """
    if not 0 <= action_idx <= 9:
        raise HTTPException(400, detail="action_idx must be in [0, 9]")

    try:
        import cv2
        from detection.yolo_detector import YOLODetector
        detector = YOLODetector()

        # Detect state from before image
        before_data = await before_file.read()
        arr = np.frombuffer(before_data, np.uint8)
        before_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        before_result = detector.detect(before_img)
        state = before_result["density"]

        # Detect next_state from after image
        after_data = await after_file.read()
        arr = np.frombuffer(after_data, np.uint8)
        after_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        after_result = detector.detect(after_img)
        next_state = after_result["density"]

        # Compute reward (same logic as environment)
        reward = 0.0
        reward_breakdown = []
        if next_state < state:
            reward += 1.0
            reward_breakdown.append({"component": "Density decreased", "value": +1.0})
            if next_state < state / 3:
                reward += 2.0
                reward_breakdown.append({"component": "Fast clearance bonus", "value": +2.0})
        else:
            reward -= 1.0
            reward_breakdown.append({"component": "Density increased/unchanged", "value": -1.0})

        # Online Q-learning update WITH before/after snapshots
        async with _agent_lock:
            agent = _get_or_load_agent()

            # ── SNAPSHOT BEFORE ──────────────────────────────────────
            q_row_before = [round(float(v), 6) for v in agent.Q[state]]
            q_old = float(agent.Q[state, action_idx])
            max_q_next = float(np.max(agent.Q[next_state]))

            # ── APPLY Q-LEARNING UPDATE (Sutton & Barto Eq. 6.8) ────
            td_error = agent.update(
                state=state,
                action=action_idx,
                reward=reward,
                next_state=next_state,
                done=False,
            )

            # ── SNAPSHOT AFTER ───────────────────────────────────────
            q_row_after = [round(float(v), 6) for v in agent.Q[state]]
            q_new = float(agent.Q[state, action_idx])

            # ── COMPUTE DIFF ─────────────────────────────────────────
            q_diff = [round(q_row_after[i] - q_row_before[i], 6)
                      for i in range(len(q_row_before))]

            # ── BELLMAN EQUATION WITH REAL NUMBERS ───────────────────
            td_target = reward + agent.gamma * max_q_next
            bellman = {
                "equation": f"Q({state},{action_idx}) ← {q_old:.4f} + {agent.alpha} × [{reward} + {agent.gamma} × {max_q_next:.4f} − {q_old:.4f}]",
                "result": f"Q({state},{action_idx}) = {q_new:.4f}",
                "components": {
                    "Q_old": round(q_old, 6),
                    "alpha": agent.alpha,
                    "reward": reward,
                    "gamma": agent.gamma,
                    "max_Q_next": round(max_q_next, 6),
                    "td_target": round(td_target, 6),
                    "td_error": round(float(td_error), 6),
                    "Q_new": round(q_new, 6),
                    "delta_Q": round(q_new - q_old, 6),
                },
            }

            # Save updated Q-table
            Q_TABLES_DIR.mkdir(parents=True, exist_ok=True)
            agent.save(str(Q_TABLES_DIR / "q_learning_best"))

        duration = GREEN_PHASE_OPTIONS[action_idx]
        success  = next_state < state

        # Record in history
        record = {
            "id":         len(decision_history) + 1,
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "filename":   f"feedback ({before_file.filename} → {after_file.filename})",
            "density":    state,
            "action_idx": action_idx,
            "duration":   duration,
            "q_value":    round(q_new, 4),
            "outcome": {
                "density_before":     state,
                "density_after":      next_state,
                "density_change":     next_state - state,
                "reward":             round(reward, 2),
                "reward_breakdown":   reward_breakdown,
                "success":            success,
                "wait_reduction_pct": max(0.0, round((state - next_state) / max(state, 1) * 100, 1)),
                "green_duration":     duration,
            },
            "counts": before_result["counts"],
            "is_real_feedback": True,
        }
        decision_history.append(record)

        return JSONResponse({
            # State transition
            "state":         state,
            "next_state":    next_state,
            "action_idx":    action_idx,
            "duration":      duration,
            "reward":        round(reward, 2),
            "reward_breakdown": reward_breakdown,
            "success":       success,

            # Q-table before/after (THE KEY RL VISUALIZATION DATA)
            "q_row_before":  q_row_before,
            "q_row_after":   q_row_after,
            "q_diff":        q_diff,

            # Bellman equation with real numbers
            "bellman":       bellman,

            # Metadata
            "td_error":      round(float(td_error), 5),
            "total_updates": agent.total_updates,
            "message":       "Q-table updated from real traffic data!",

            # Detection details
            "before_counts": before_result["counts"],
            "after_counts":  after_result["counts"],
            "before_density": state,
            "after_density":  next_state,
        })

    except Exception as e:
        logger.exception("Feedback error")
        raise HTTPException(500, detail=str(e))


@app.get("/api/decision-history")
async def get_decision_history():
    """Return session-level log of all decisions."""
    return JSONResponse({
        "total":   len(decision_history),
        "history": list(reversed(decision_history)),
    })


@app.delete("/api/decision-history")
async def clear_decision_history():
    """Clear the session decision history."""
    decision_history.clear()
    return JSONResponse({"message": "Decision history cleared."})


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
    global _live_agent
    import subprocess
    training_state["running"] = True
    training_state["total"]   = episodes
    training_state["episode"] = 0

    cmd = [sys.executable, str(ROOT / "train.py"),
           "--episodes", str(episodes),
           "--agent", "q_learning"]  # Q-Learning only
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
                            part.split("=")[1].strip().replace("s", ""))
                    if "ε=" in part or "epsilon=" in part:
                        training_state["epsilon"] = float(part.split("=")[1].strip())
            except Exception:
                pass

    await proc.wait()
    training_state["running"] = False
    # Reload live agent after training completes
    _live_agent = None
    _get_or_load_agent()
