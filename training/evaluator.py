#!/usr/bin/env python3
"""
evaluator.py — Compare Fixed-Timer, SARSA, and Q-Learning agents on SUMO.

Runs each algorithm for N evaluation episodes (no exploration, greedy policy only)
and computes the metrics from Table 1 of the reference paper:
  - Mean number of cars waiting
  - Mean waiting time (seconds)
  - % improvement vs fixed-timer baseline

Results are saved to:
  results/evaluation_results.json
  results/plots/comparison_table.png
  results/plots/training_curves.png
  results/plots/q_table_heatmap.png
  results/plots/reward_distribution.png
"""

import json
import logging
import pathlib
import time
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yaml

logger = logging.getLogger(__name__)

RESULTS_DIR = pathlib.Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

GREEN_PHASE_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
FIXED_GREEN = 30   # default fixed-timer green duration (seconds)


# ---------------------------------------------------------------------------
# Baseline: Fixed-timer controller
# ---------------------------------------------------------------------------

class FixedTimerController:
    """Always applies the same green phase duration — the baseline."""

    def __init__(self, duration: int = FIXED_GREEN):
        self.duration_idx = GREEN_PHASE_OPTIONS.index(duration)

    def choose_action(self, state: int) -> int:
        return self.duration_idx

    def greedy_action(self, state: int) -> int:
        return self.duration_idx


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Evaluates multiple controllers on the SUMO environment and produces
    comparison plots and a results table.
    """

    def __init__(self, env, num_eval_episodes: int = 10):
        self.env               = env
        self.num_eval_episodes = num_eval_episodes
        self.results           : dict[str, dict] = {}

    def evaluate_controller(self, controller, name: str,
                            is_sarsa: bool = False) -> dict:
        """
        Run greedy evaluation for N episodes (ε=0, no exploration).
        Returns aggregated metrics.
        """
        wait_times    = []
        rewards       = []
        cars_waiting_samples = []

        print(f"  Evaluating {name} ({self.num_eval_episodes} episodes)...")

        for ep in range(self.num_eval_episodes):
            state = self.env.reset(episode=1000 + ep)   # fresh seed range
            total_reward = 0.0
            done         = False

            if is_sarsa:
                action = controller.greedy_action(state)

            while not done:
                if is_sarsa:
                    next_state, reward, done, info = self.env.step(action)
                    action = controller.greedy_action(next_state)
                    state  = next_state
                else:
                    action = controller.greedy_action(state)
                    state, reward, done, info = self.env.step(action)

                total_reward += reward

            mean_wait = self.env.get_mean_wait_time()
            wait_times.append(mean_wait)
            rewards.append(total_reward)

        metrics = {
            "name":             name,
            "mean_wait_time":   round(float(np.mean(wait_times)), 3),
            "std_wait_time":    round(float(np.std(wait_times)),  3),
            "mean_reward":      round(float(np.mean(rewards)),    3),
            "episodes":         self.num_eval_episodes,
        }
        self.results[name] = metrics
        print(f"    → mean wait = {metrics['mean_wait_time']:.2f}s  "
              f"(±{metrics['std_wait_time']:.2f})")
        return metrics

    def print_comparison_table(self):
        """Print the results table (mirrors Table 1 from the research paper)."""
        fixed = self.results.get("Fixed Timer", {}).get("mean_wait_time", 1)

        print(f"\n{'='*70}")
        print(f"  {'Algorithm':<20} {'Mean Wait (s)':>14} {'Improvement':>12}")
        print(f"  {'-'*50}")
        for name, m in self.results.items():
            wait = m["mean_wait_time"]
            pct  = (1 - wait / fixed) * 100 if name != "Fixed Timer" else 0.0
            sign = f"↓ {pct:.1f}%" if pct > 0 else "  baseline"
            print(f"  {name:<20} {wait:>14.2f}s {sign:>12}")
        print(f"{'='*70}\n")

    def save_results(self):
        out = RESULTS_DIR / "evaluation_results.json"
        out.write_text(json.dumps(self.results, indent=2))
        print(f"  Results saved → {out}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_training_curves(self, logs: dict[str, list[dict]]):
        """Plot reward and wait-time training curves for all agents."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Training Curves — Smart Signal", fontsize=14, fontweight="bold")

        palette = {"Q-Learning": "#2196F3", "SARSA": "#FF9800"}

        for agent_name, log in logs.items():
            color  = palette.get(agent_name, "gray")
            eps    = [e["episode"]      for e in log]
            rwds   = [e["total_reward"] for e in log]
            waits  = [e["mean_wait"]    for e in log]

            # Smooth with rolling mean
            window = max(1, len(rwds) // 20)
            rwds_s  = np.convolve(rwds,  np.ones(window)/window, mode="valid")
            waits_s = np.convolve(waits, np.ones(window)/window, mode="valid")
            eps_s   = eps[window-1:]

            ax1.plot(eps_s, rwds_s,  color=color, linewidth=2, label=agent_name)
            ax2.plot(eps_s, waits_s, color=color, linewidth=2, label=agent_name)

        ax1.set_xlabel("Episode"); ax1.set_ylabel("Total Reward")
        ax1.set_title("Reward per Episode"); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.set_xlabel("Episode"); ax2.set_ylabel("Mean Wait Time (s)")
        ax2.set_title("Mean Waiting Time per Episode"); ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        path = PLOTS_DIR / "training_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved → {path}")

    def plot_comparison_bar(self):
        """Bar chart comparing all algorithms (mirrors paper Table 1)."""
        if not self.results:
            return
        fixed_wait = self.results.get("Fixed Timer", {}).get("mean_wait_time", 1)
        names  = list(self.results.keys())
        waits  = [self.results[n]["mean_wait_time"] for n in names]
        colors = ["#9E9E9E", "#FF9800", "#2196F3"][:len(names)]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(names, waits, color=colors, width=0.5, edgecolor="white",
                      linewidth=1.5)

        # Annotate bars
        for bar, wait, name in zip(bars, waits, names):
            pct = (1 - wait / fixed_wait) * 100 if name != "Fixed Timer" else 0
            label = f"{wait:.1f}s" + (f"\n(↓{pct:.1f}%)" if pct > 0 else "\n(baseline)")
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    label, ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_ylabel("Mean Waiting Time (seconds)", fontsize=12)
        ax.set_title("Algorithm Comparison — Mean Vehicle Waiting Time",
                     fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(waits) * 1.3)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = PLOTS_DIR / "comparison_bar.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved → {path}")

    def plot_q_table_heatmap(self, agent, title: str = "Q-Table Heatmap"):
        """Visualise the learned Q-table as a heatmap (state × action)."""
        # Slice to display states 1-100
        Q = agent.Q[1:, :]   # shape (100, 10)

        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(Q, aspect="auto", cmap="RdYlGn", origin="lower")

        ax.set_xlabel("Action (Green Phase Duration in seconds)", fontsize=12)
        ax.set_ylabel("State (Traffic Density)", fontsize=12)
        ax.set_title(f"{title}\n(Green = high Q-value = preferred action)",
                     fontsize=13, fontweight="bold")

        # Axis ticks
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"{g}s" for g in GREEN_PHASE_OPTIONS])
        ax.set_yticks(range(0, 100, 10))
        ax.set_yticklabels([str(i) for i in range(1, 101, 10)])

        plt.colorbar(im, ax=ax, label="Q-value (expected cumulative reward)")
        plt.tight_layout()
        path = PLOTS_DIR / "q_table_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved → {path}")

    def plot_epsilon_decay(self, log: list[dict]):
        """Plot ε annealing over training episodes."""
        eps = [(e["episode"], e["epsilon"]) for e in log]
        x, y = zip(*eps)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="#E91E63", linewidth=2)
        ax.fill_between(x, y, alpha=0.15, color="#E91E63")
        ax.set_xlabel("Episode"); ax.set_ylabel("ε (Exploration Rate)")
        ax.set_title("ε-Greedy Exploration Decay (Sutton & Barto Ch.2.7)",
                     fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        path = PLOTS_DIR / "epsilon_decay.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved → {path}")

    def plot_td_error(self, log: list[dict]):
        """Plot mean absolute TD error convergence."""
        eps = [e["episode"]  for e in log]
        tds = [e["td_error"] for e in log]
        window = max(1, len(tds) // 15)
        tds_s = np.convolve(tds, np.ones(window)/window, mode="valid")
        eps_s = eps[window-1:]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(eps_s, tds_s, color="#4CAF50", linewidth=2)
        ax.set_xlabel("Episode"); ax.set_ylabel("Mean |TD Error|")
        ax.set_title("TD Error Convergence (δ = R + γ·max Q(S\',a) − Q(S,A))",
                     fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        path = PLOTS_DIR / "td_error.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved → {path}")
