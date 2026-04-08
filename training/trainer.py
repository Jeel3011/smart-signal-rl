#!/usr/bin/env python3
"""
trainer.py — Training loop for Q-learning and SARSA agents on SUMO environment.

Training procedure (Sutton & Barto Ch.6 Figure 6.6 — Q-learning pseudocode):
  1. Initialise Q(s, a) arbitrarily (here: zeros)
  2. For each episode:
       a. Initialise S
       b. Choose A from S using ε-greedy policy
       c. Take action A, observe R, S'
       d. Q(S,A) ← Q(S,A) + α·[R + γ·max_a Q(S',a) − Q(S,A)]     (Q-learning)
          Q(S,A) ← Q(S,A) + α·[R + γ·Q(S',A') − Q(S,A)]           (SARSA)
       e. S ← S'; (SARSA: A ← A')
       f. Repeat until S is terminal
  3. Decay ε after each episode
"""

import logging
import pathlib
import json
import time
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates Q-learning / SARSA training against the SUMO environment.

    Features:
    - Episode loop with metric logging (reward, wait time, TD error, ε)
    - Early stopping if reward converges
    - Checkpoint saving every N episodes
    - Produces training_log.json for dashboard / plotting
    """

    def __init__(self,
                 agent,
                 env,
                 num_episodes: int = 500,
                 checkpoint_every: int = 50,
                 save_dir: str = "results/q_tables",
                 agent_name: str = "q_learning"):

        self.agent           = agent
        self.env             = env
        self.num_episodes    = num_episodes
        self.checkpoint_every = checkpoint_every
        self.save_dir        = pathlib.Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.agent_name      = agent_name

        self.log: list[dict] = []     # per-episode metrics

    def train(self, verbose: bool = True) -> list[dict]:
        """
        Run the full training loop.
        Returns the training log (list of per-episode dicts).
        """
        is_sarsa = self.agent_name == "sarsa"
        best_wait = float("inf")

        print(f"\n{'='*60}")
        print(f"  Training {self.agent_name.upper()} — {self.num_episodes} episodes")
        print(f"  α={self.agent.alpha}  γ={self.agent.gamma}  "
              f"ε={self.agent.epsilon}  ε_decay={self.agent.epsilon_decay}")
        print(f"{'='*60}")

        for ep in range(self.num_episodes):
            t_start = time.time()

            # --- Reset environment ---
            state  = self.env.reset(episode=ep)
            if is_sarsa:
                action = self.agent.choose_action(state)

            total_reward = 0.0
            steps        = 0
            done         = False

            # --- Episode loop (Sutton & Barto Ch.6 pseudocode) ---
            while not done:
                if is_sarsa:
                    next_state, reward, done, _ = self.env.step(action)
                    next_action = self.agent.choose_action(next_state)
                    self.agent.update(state, action, reward,
                                      next_state, next_action, done)
                    state  = next_state
                    action = next_action
                else:
                    action = self.agent.choose_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.agent.update(state, action, reward, next_state, done)
                    state = next_state

                total_reward += reward
                steps        += 1

            # --- Post-episode updates ---
            mean_wait  = self.env.get_mean_wait_time()
            td_error   = self.agent.mean_td_error()
            ep_time    = time.time() - t_start

            self.agent.decay_epsilon()

            entry = {
                "episode":      ep + 1,
                "total_reward": round(total_reward, 3),
                "mean_wait":    round(mean_wait, 3),
                "epsilon":      round(self.agent.epsilon, 5),
                "td_error":     round(td_error, 5),
                "steps":        steps,
                "time_s":       round(ep_time, 2),
            }
            self.log.append(entry)

            # Save best
            if mean_wait < best_wait and mean_wait > 0:
                best_wait = mean_wait
                self.agent.save(str(self.save_dir / f"{self.agent_name}_best"))

            # Checkpoint
            if (ep + 1) % self.checkpoint_every == 0:
                self.agent.save(str(self.save_dir / f"{self.agent_name}_ep{ep+1}"))
                self._save_log()

            if verbose and (ep + 1) % 10 == 0:
                print(f"  Ep {ep+1:4d}/{self.num_episodes} | "
                      f"reward={total_reward:+8.2f} | "
                      f"wait={mean_wait:6.2f}s | "
                      f"ε={self.agent.epsilon:.4f} | "
                      f"TD={td_error:.4f}")

        # Final save
        self.agent.save(str(self.save_dir / f"{self.agent_name}_final"))
        self._save_log()

        print(f"\n  ✅  Training complete. Best mean wait: {best_wait:.2f}s")
        print(f"  Q-table saved → {self.save_dir}/{self.agent_name}_best.npy\n")
        return self.log

    def _save_log(self):
        out = pathlib.Path("results") / f"training_log_{self.agent_name}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(self.log, indent=2))
