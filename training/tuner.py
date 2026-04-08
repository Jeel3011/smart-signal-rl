#!/usr/bin/env python3
"""
tuner.py — Grid-search hyperparameter tuning for Q-learning and SARSA agents.

Searches over:
  α (alpha)        : learning rate     — {0.01, 0.05, 0.1, 0.2, 0.5}
  γ (gamma)        : discount factor   — {0.1, 0.5, 0.7, 0.9, 0.99}
  ε (epsilon)      : initial explore   — {0.3, 0.5, 0.7, 0.9}
  ε_decay          : decay rate        — {0.995, 0.999}

Total: 5 × 5 × 4 × 2 = 200 configurations.
Each config is trained for `tuning_episodes` episodes,
then evaluated on the mean wait time of the last `eval_episodes` episodes.
Best config is saved to results/best_params.json.

Reference: Sutton & Barto Ch.2 — ε-greedy parameter sensitivity analysis.
"""

import json
import logging
import pathlib
import itertools
import time
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight SimPy-free simulation for fast tuning (no SUMO overhead)
# We use a simple stochastic traffic model during tuning, then re-train
# the best agent on the full SUMO environment.
# ---------------------------------------------------------------------------

class FastTrafficEnv:
    """
    Lightweight stochastic traffic environment for fast hyperparameter tuning.
    Avoids SUMO startup overhead during the 200-config grid search.

    Traffic dynamics:
      - Vehicles arrive according to a Poisson process
      - Green phase duration reduces density proportionally
      - Noise is added to simulate real-world uncertainty
    """

    def __init__(self, seed: int = 42, episode_steps: int = 60):
        self.rng           = np.random.default_rng(seed)
        self.episode_steps = episode_steps  # number of phase decisions per episode
        self.reset()

    def reset(self) -> int:
        self.density     = int(self.rng.integers(10, 60))
        self.step_count  = 0
        self.total_wait  = 0.0
        self.phases_done = 0
        return self.density

    def step(self, action_idx: int) -> tuple[int, float, bool]:
        """Apply a green-phase action, return (next_state, reward, done)."""
        GREEN_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        green_duration = GREEN_OPTIONS[action_idx]

        density_start = self.density

        # Vehicles cleared proportional to green duration + noise
        cleared = self.rng.poisson(green_duration * 0.4)
        # New vehicles arriving during the phase
        arrivals = self.rng.poisson(green_duration * 0.3)

        mid_density = max(0, density_start - cleared // 2 + self.rng.poisson(3))

        self.density = max(1, min(100,
            density_start - cleared + arrivals +
            int(self.rng.normal(0, 3))
        ))

        # Reward (same as SUMO environment)
        reward = 0.0
        if density_start > 0 and mid_density < density_start / 3:
            reward += 2.0
        if self.density < density_start:
            reward += 1.0
        else:
            reward -= 1.0

        # Approximate waiting time contribution
        self.total_wait += density_start * 1.5 / max(1, green_duration / 10)
        self.step_count  += 1

        done = self.step_count >= self.episode_steps
        if done:
            self.phases_done += 1
        return self.density, reward, done

    def mean_wait_per_phase(self) -> float:
        return self.total_wait / max(1, self.step_count)


def run_episode_qlearning(env: FastTrafficEnv, agent,
                          train: bool = True) -> tuple[float, float]:
    """Run one episode of Q-learning on the fast env. Returns (mean_wait, total_reward)."""
    state = env.reset()
    total_reward = 0.0

    while True:
        action = agent.choose_action(state) if train else agent.greedy_action(state)
        next_state, reward, done = env.step(action)
        if train:
            agent.update(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        if done:
            break

    return env.mean_wait_per_phase(), total_reward


def run_episode_sarsa(env: FastTrafficEnv, agent,
                      train: bool = True) -> tuple[float, float]:
    """Run one episode of SARSA on the fast env."""
    state  = env.reset()
    action = agent.choose_action(state)
    total_reward = 0.0

    while True:
        next_state, reward, done = env.step(action)
        next_action = agent.choose_action(next_state) if train else agent.greedy_action(next_state)
        if train:
            agent.update(state, action, reward, next_state, next_action, done)
        total_reward += reward
        state  = next_state
        action = next_action
        if done:
            break

    return env.mean_wait_per_phase(), total_reward


def tune(agent_type: str = "q_learning",
         config_path: str = "config/config.yaml",
         verbose: bool = True) -> dict:
    """
    Grid-search hyperparameter tuning.

    Returns the best hyperparameter dict.
    """
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

    cfg = yaml.safe_load(pathlib.Path(config_path).read_text())
    ht  = cfg["hyperparameter_tuning"]

    alphas          = ht["alpha"]
    gammas          = ht["gamma"]
    epsilons        = ht["epsilon"]
    epsilon_decays  = ht["epsilon_decay"]
    tuning_episodes = ht["tuning_episodes"]
    eval_episodes   = ht["eval_episodes"]

    param_grid = list(itertools.product(alphas, gammas, epsilons, epsilon_decays))
    total_configs = len(param_grid)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Hyperparameter Tuning — {agent_type.upper()}")
        print(f"  Total configurations : {total_configs}")
        print(f"  Episodes per config  : {tuning_episodes}")
        print(f"  Eval last N episodes : {eval_episodes}")
        print(f"{'='*60}\n")

    best_score  = float("inf")
    best_params = {}
    results     = []

    start_time = time.time()

    for i, (alpha, gamma, epsilon, eps_decay) in enumerate(param_grid):
        if agent_type == "q_learning":
            from agents.q_learning_agent import QLearningAgent
            agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon,
                                   epsilon_decay=eps_decay, seed=42)
            run_ep = run_episode_qlearning
        else:
            from agents.sarsa_agent import SARSAAgent
            agent = SARSAAgent(alpha=alpha, gamma=gamma, epsilon=epsilon,
                               epsilon_decay=eps_decay, seed=42)
            run_ep = run_episode_sarsa

        env = FastTrafficEnv(seed=42)
        wait_history = []

        for ep in range(tuning_episodes):
            wait, _ = run_ep(env, agent, train=True)
            wait_history.append(wait)
            agent.decay_epsilon()

        # Score = mean wait time over last eval_episodes (lower is better)
        score = float(np.mean(wait_history[-eval_episodes:]))

        results.append({
            "alpha": alpha, "gamma": gamma, "epsilon": epsilon,
            "epsilon_decay": eps_decay, "score": score
        })

        if score < best_score:
            best_score  = score
            best_params = {
                "alpha": alpha, "gamma": gamma, "epsilon": epsilon,
                "epsilon_decay": eps_decay, "score": score
            }

        if verbose and (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            remaining = elapsed / (i + 1) * (total_configs - i - 1)
            print(f"  [{i+1:3d}/{total_configs}]  "
                  f"α={alpha:.3f}  γ={gamma:.2f}  ε={epsilon:.1f}  "
                  f"ε_decay={eps_decay:.3f}  →  score={score:.3f}  "
                  f"(best={best_score:.3f})  ETA: {remaining:.0f}s")

    elapsed = time.time() - start_time

    # Save results
    out_dir = pathlib.Path("results")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"tuning_{agent_type}_all.json").write_text(
        json.dumps(sorted(results, key=lambda x: x["score"]), indent=2)
    )
    (out_dir / f"best_params_{agent_type}.json").write_text(
        json.dumps(best_params, indent=2)
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Tuning complete in {elapsed:.1f}s")
        print(f"  Best parameters ({agent_type}):")
        for k, v in best_params.items():
            print(f"    {k:20s} = {v}")
        print(f"{'='*60}\n")

    return best_params


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["q_learning", "sarsa"], default="q_learning")
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args()
    tune(agent_type=args.agent, config_path=args.config, verbose=True)
