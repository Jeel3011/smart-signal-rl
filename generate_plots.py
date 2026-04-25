#!/usr/bin/env python3
"""
generate_plots.py — Standalone script to regenerate all dashboard plots.
Does NOT require SUMO. Trains a quick Q-table from scratch using FastTrafficEnv,
then generates all 5 plots needed by the dashboard.
Run: python generate_plots.py
"""
import sys, json, pathlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

PLOTS_DIR   = ROOT / "results" / "plots"
RESULTS_DIR = ROOT / "results"
Q_DIR       = ROOT / "results" / "q_tables"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
Q_DIR.mkdir(parents=True, exist_ok=True)

GREEN_PHASE_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# ─────────────────────────────────────────────
# Step 1: Quick retrain to produce .npy files
# ─────────────────────────────────────────────
print("=" * 55)
print("  Step 1: Quick training to generate Q-tables (.npy)")
print("=" * 55)

from training.tuner import FastTrafficEnv
from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SARSAAgent

EPISODES = 300

def train_agent(agent, name):
    env = FastTrafficEnv(seed=42, episode_steps=60)
    log = []
    best_wait = float("inf")
    for ep in range(EPISODES):
        state = env.reset()
        if name == "sarsa":
            action = agent.choose_action(state)
        total_r, steps = 0.0, 0
        while True:
            if name == "sarsa":
                ns, r, done = env.step(action)
                na = agent.choose_action(ns)
                agent.update(state, action, r, ns, na, done)
                state, action = ns, na
            else:
                action = agent.choose_action(state)
                ns, r, done = env.step(action)
                agent.update(state, action, r, ns, done)
                state = ns
            total_r += r
            steps += 1
            if done:
                break
        wait = env.mean_wait_per_phase()
        agent.decay_epsilon()
        log.append({
            "episode": ep + 1,
            "total_reward": round(total_r, 3),
            "mean_wait": round(wait, 3),
            "epsilon": round(agent.epsilon, 5),
            "td_error": round(agent.mean_td_error(), 5),
            "steps": steps,
            "time_s": 0.0,
        })
        if wait < best_wait and wait > 0:
            best_wait = wait
        if (ep + 1) % 50 == 0:
            print(f"    {name.upper()} Ep {ep+1}/{EPISODES} | reward={total_r:+.1f} | "
                  f"wait={wait:.2f} | ε={agent.epsilon:.3f}")
    return log, best_wait

# Train Q-learning
ql_agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.7,
                          epsilon_decay=0.995, epsilon_min=0.05, seed=42)
print("\n  Training Q-Learning...")
ql_log, ql_best = train_agent(ql_agent, "q_learning")
ql_agent.save(str(Q_DIR / "q_learning_best"))
ql_agent.save(str(Q_DIR / "q_learning_final"))
print(f"  ✅  Q-Learning done. Best wait: {ql_best:.2f} | Q-table saved.")

# Train SARSA
sarsa_agent = SARSAAgent(alpha=0.1, gamma=0.9, epsilon=0.7,
                         epsilon_decay=0.995, epsilon_min=0.05, seed=42)
print("\n  Training SARSA...")
sarsa_log, sarsa_best = train_agent(sarsa_agent, "sarsa")
sarsa_agent.save(str(Q_DIR / "sarsa_best"))
sarsa_agent.save(str(Q_DIR / "sarsa_final"))
print(f"  ✅  SARSA done. Best wait: {sarsa_best:.2f} | Q-table saved.")

# Save training logs
(RESULTS_DIR / "training_log_q_learning.json").write_text(json.dumps(ql_log, indent=2))
(RESULTS_DIR / "training_log_sarsa.json").write_text(json.dumps(sarsa_log, indent=2))
print("\n  Training logs saved.")

# ─────────────────────────────────────────────
# Step 2: Generate all plots
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Step 2: Generating dashboard plots")
print("=" * 55)

# Load evaluation results
eval_results = json.loads((RESULTS_DIR / "evaluation_results.json").read_text())
logs = {"Q-Learning": ql_log, "SARSA": sarsa_log}

# --- Plot 1: Training Curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training Curves — Smart Signal", fontsize=14, fontweight="bold")
palette = {"Q-Learning": "#2196F3", "SARSA": "#FF9800"}
for name, log in logs.items():
    color = palette.get(name, "gray")
    eps   = [e["episode"]      for e in log]
    rwds  = [e["total_reward"] for e in log]
    waits = [e["mean_wait"]    for e in log]
    window = max(1, len(rwds) // 20)
    rwds_s  = np.convolve(rwds,  np.ones(window)/window, mode="valid")
    waits_s = np.convolve(waits, np.ones(window)/window, mode="valid")
    eps_s   = eps[window-1:]
    ax1.plot(eps_s, rwds_s,  color=color, linewidth=2, label=name)
    ax2.plot(eps_s, waits_s, color=color, linewidth=2, label=name)
ax1.set_xlabel("Episode"); ax1.set_ylabel("Total Reward"); ax1.set_title("Reward per Episode")
ax1.legend(); ax1.grid(alpha=0.3)
ax2.set_xlabel("Episode"); ax2.set_ylabel("Mean Wait Time (s)"); ax2.set_title("Mean Waiting Time")
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
p = PLOTS_DIR / "training_curves.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"  ✅  {p.name}")

# --- Plot 2: Comparison Bar ---
fixed_wait = eval_results.get("Fixed Timer", {}).get("mean_wait_time", 1)
names  = list(eval_results.keys())
waits  = [eval_results[n]["mean_wait_time"] for n in names]
colors = ["#9E9E9E", "#FF9800", "#2196F3"][:len(names)]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(names, waits, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
for bar, wait, name in zip(bars, waits, names):
    pct = (1 - wait / fixed_wait) * 100 if name != "Fixed Timer" else 0
    label = f"{wait:.1f}s" + (f"\n(↓{pct:.1f}%)" if pct > 0 else "\n(baseline)")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            label, ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Mean Waiting Time (seconds)", fontsize=12)
ax.set_title("Algorithm Comparison — Mean Vehicle Waiting Time", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(waits) * 1.3); ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
p = PLOTS_DIR / "comparison_bar.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"  ✅  {p.name}")

# --- Plot 3: Q-Table Heatmap ---
Q = ql_agent.Q[1:, :]  # shape (100, 10)
fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(Q, aspect="auto", cmap="RdYlGn", origin="lower")
ax.set_xlabel("Action (Green Phase Duration in seconds)", fontsize=12)
ax.set_ylabel("State (Traffic Density)", fontsize=12)
ax.set_title("Q-Learning — Learned Signal Policy\n(Green = high Q-value = preferred action)",
             fontsize=13, fontweight="bold")
ax.set_xticks(range(10))
ax.set_xticklabels([f"{g}s" for g in GREEN_PHASE_OPTIONS])
ax.set_yticks(range(0, 100, 10))
ax.set_yticklabels([str(i) for i in range(1, 101, 10)])
plt.colorbar(im, ax=ax, label="Q-value (expected cumulative reward)")
plt.tight_layout()
p = PLOTS_DIR / "q_table_heatmap.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"  ✅  {p.name}")

# --- Plot 4: Epsilon Decay ---
eps_vals = [(e["episode"], e["epsilon"]) for e in ql_log]
x, y = zip(*eps_vals)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, color="#E91E63", linewidth=2)
ax.fill_between(x, y, alpha=0.15, color="#E91E63")
ax.set_xlabel("Episode"); ax.set_ylabel("ε (Exploration Rate)")
ax.set_title("ε-Greedy Exploration Decay (Sutton & Barto Ch.2.7)", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
p = PLOTS_DIR / "epsilon_decay.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"  ✅  {p.name}")

# --- Plot 5: TD Error ---
tds = [e["td_error"] for e in ql_log]
window = max(1, len(tds) // 15)
tds_s = np.convolve(tds, np.ones(window)/window, mode="valid")
eps_s = [e["episode"] for e in ql_log][window-1:]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(eps_s, tds_s, color="#4CAF50", linewidth=2)
ax.set_xlabel("Episode"); ax.set_ylabel("Mean |TD Error|")
ax.set_title("TD Error Convergence (δ = R + γ·max Q(S',a) − Q(S,A))", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
p = PLOTS_DIR / "td_error.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"  ✅  {p.name}")

print("\n" + "=" * 55)
print("  All plots generated! Files in results/plots/")
print("=" * 55)
for f in sorted(PLOTS_DIR.glob("*.png")):
    print(f"    • {f.name}  ({f.stat().st_size // 1024} KB)")
print("\n  Now go to http://127.0.0.1:8000 and refresh the dashboard!")
