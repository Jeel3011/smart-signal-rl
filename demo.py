#!/usr/bin/env python3
"""
demo.py — Interactive demonstration of all Smart Signal components.

Demonstrates:
  1. YOLO detection on a traffic image (shows bounding boxes)
  2. Q-table policy visualisation (what action the trained agent takes for each state)
  3. Live episode replay using SUMO-GUI (if SUMO is available)
  4. Comparison summary table

Usage:
  python demo.py                    # full demo (needs SUMO)
  python demo.py --no-sumo          # show YOLO + Q-table only
  python demo.py --image path.png   # custom image for YOLO demo
"""

import argparse
import json
import logging
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("TkAgg" if sys.platform == "darwin" else "Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

GREEN_PHASE_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def demo_yolo(image_path: str | None):
    """Show YOLO detection on an image and print density."""
    print("\n" + "="*60)
    print("  MODULE 2: YOLOv8 Vehicle Detection Demo")
    print("="*60)

    # Use most recent SUMO screenshot or user-supplied image
    if image_path is None:
        shots = sorted(pathlib.Path("sumo_env/screenshots").glob("*.png"))
        if shots:
            image_path = str(shots[-1])
        else:
            print("  ⚠️  No screenshot found. Run with --gui once to capture frames.")
            print("     Skipping YOLO demo.\n")
            return

    from detection.yolo_detector import YOLODetector
    detector = YOLODetector()
    result   = detector.detect(image_path)

    print(f"  Image       : {image_path}")
    print(f"  Detections  :")
    for cls, cnt in result["counts"].items():
        if cnt > 0:
            print(f"    {cls:<10} : {cnt}")
    print(f"  Raw density : {result['raw']:.2f}")
    print(f"  RL State    : {result['density']} / 100   "
          f"{'🔴 HIGH' if result['density'] > 70 else '🟡 MED' if result['density'] > 35 else '🟢 LOW'}")

    # Save annotated image
    import cv2
    out = pathlib.Path("results/yolo_demo.png")
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), result["annotated"])
    print(f"  Annotated image saved → {out}\n")
    return result


def demo_q_table():
    """Visualise the trained Q-table policy."""
    print("="*60)
    print("  MODULE 3: Q-Learning Agent — Policy Visualisation")
    print("="*60)

    from agents.q_learning_agent import QLearningAgent

    q_path = pathlib.Path("results/q_tables/q_learning_best.npy")
    if not q_path.exists():
        print("  ⚠️  No trained Q-table found. Run python train.py first.\n")
        return None

    agent = QLearningAgent()
    agent.load("results/q_tables/q_learning_best")
    agent.epsilon = 0.0   # pure greedy for demo

    print(f"\n  Hyperparameters:")
    for k, v in agent.get_config().items():
        print(f"    {k:<20} = {v}")

    print(f"\n  Value Function V(s) = max_a Q(s, a)  [selected states]:")
    print(f"  {'State (density)':>17} | {'Best Action':>12} | {'V(s)':>8}")
    print(f"  {'─'*45}")
    for state in [10, 25, 50, 75, 90]:
        action_idx = agent.greedy_action(state)
        duration   = GREEN_PHASE_OPTIONS[action_idx]
        value      = agent.value_function(state)
        bar        = "█" * (duration // 5)
        print(f"  {state:>17d} | {duration:>10d}s | {value:>8.3f}  {bar}")

    stats = agent.q_table_stats()
    print(f"\n  Q-table stats: min={stats['min']:.3f}  max={stats['max']:.3f}  "
          f"mean={stats['mean']:.3f}  nonzero={stats['nonzero_pct']:.1f}%\n")
    return agent


def demo_results():
    """Print evaluation results comparison."""
    print("="*60)
    print("  EVALUATION RESULTS — Algorithm Comparison")
    print("="*60)

    rpath = pathlib.Path("results/evaluation_results.json")
    if not rpath.exists():
        print("  ⚠️  No results yet. Run python train.py first.\n")
        return

    results = json.loads(rpath.read_text())
    fixed   = results.get("Fixed Timer", {}).get("mean_wait_time", 1.0)

    print(f"\n  {'Algorithm':<20} {'Mean Wait (s)':>14} {'Improvement':>12}")
    print(f"  {'─'*50}")
    for name, m in results.items():
        wait = m["mean_wait_time"]
        std  = m.get("std_wait_time", 0)
        pct  = (1 - wait / fixed) * 100 if name != "Fixed Timer" else 0.0
        sign = f"↓ {pct:.1f}%" if pct > 0 else "  baseline"
        print(f"  {name:<20} {wait:>10.2f} ± {std:.2f}s {sign:>12}")
    print()


def demo_live_sumo(cfg: dict, agent):
    """Run one episode with SUMO-GUI for visual demonstration."""
    print("="*60)
    print("  LIVE DEMO: SUMO-GUI Traffic Simulation")
    print("="*60)
    print("  Launching SUMO-GUI... (close window to exit)\n")

    try:
        from sumo_env.environment import SumoEnvironment
    except Exception as e:
        print(f"  ⚠️  SUMO not available: {e}\n")
        return

    env   = SumoEnvironment(use_gui=True, episode_length=600)
    state = env.reset(episode=999)
    agent.epsilon = 0.0  # greedy

    step = 0
    while True:
        action = agent.greedy_action(state)
        duration = GREEN_PHASE_OPTIONS[action]
        print(f"  Step {step+1:3d} | density={state:3d} | agent chooses {duration}s green")

        state, reward, done, info = env.step(action)
        env.capture_screenshot(step)
        step += 1
        if done:
            break

    print(f"\n  Episode done. Mean wait time: {env.get_mean_wait_time():.2f}s")
    env.close()


def main():
    import yaml
    parser = argparse.ArgumentParser(description="Smart Signal Demo")
    parser.add_argument("--no-sumo", action="store_true")
    parser.add_argument("--image",   default=None)
    parser.add_argument("--config",  default="config/config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())

    print("\n" + "🚦 " * 20)
    print("   SMART SIGNAL — Adaptive Traffic Control Demo")
    print("   RL + YOLOv8 | Sutton & Barto Q-Learning")
    print("🚦 " * 20 + "\n")

    # 1. YOLO detection
    demo_yolo(args.image)

    # 2. Q-table policy
    agent = demo_q_table()

    # 3. Results table
    demo_results()

    # 4. Live SUMO demo (optional)
    if not args.no_sumo and agent is not None:
        resp = input("  Launch SUMO-GUI live demo? [y/N] ").strip().lower()
        if resp == "y":
            demo_live_sumo(cfg, agent)

    print("✅  Demo complete.")


if __name__ == "__main__":
    main()
