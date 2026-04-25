#!/usr/bin/env python3
"""
q_learning_agent.py — Tabular Q-Learning agent.

Implements the Q-learning (off-policy TD control) update rule from:
  Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An Introduction.
  Chapter 6, Section 6.5, Equation 6.8:

      Q(S,A) ← Q(S,A) + α · [R + γ · max_a Q(S', a) − Q(S,A)]

where:
  α     = learning rate         (how fast knowledge is updated)
  γ     = discount factor       (weight given to future rewards)
  ε     = exploration rate      (probability of random action)
  S, S' = current, next state
  A     = action taken
  R     = reward received

Q-learning is OFF-POLICY: it learns V*(s) = max_a Q(s,a) regardless of
the exploration policy used (ε-greedy), making it more sample-efficient
than on-policy methods like SARSA.

State space : 100 discrete density bins  (S ∈ {1, …, 100})
Action space: 10 green-phase durations   (A ∈ {10, 20, …, 100} seconds)
Q-table size: 100 × 10 = 1,000 entries
"""

import numpy as np
import json
import pathlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default hyper-params (overridden by tuner)
DEFAULT_ALPHA         = 0.1
DEFAULT_GAMMA         = 0.9
DEFAULT_EPSILON       = 0.7
DEFAULT_EPSILON_DECAY = 0.995
DEFAULT_EPSILON_MIN   = 0.05

NUM_STATES  = 100
NUM_ACTIONS = 10
GREEN_PHASE_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


class QLearningAgent:
    """
    Tabular Q-Learning agent implementing Sutton & Barto Ch.6 Section 6.5.

    The Q-table maps each (state, action) pair to an expected cumulative reward.
    The agent uses ε-greedy action selection (Sutton & Barto Ch.2 Section 2.4):
      - With probability ε   → explore: random action
      - With probability 1-ε → exploit: arg max_a Q(s, a)

    After each transition (s, a, r, s'), the Q-table is updated:
      Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s', a') - Q(s,a)]
    This is the TD error (temporal-difference error, Ch.6.1), scaled by α.
    """

    def __init__(self,
                 num_states: int = NUM_STATES,
                 num_actions: int = NUM_ACTIONS,
                 alpha: float = DEFAULT_ALPHA,
                 gamma: float = DEFAULT_GAMMA,
                 epsilon: float = DEFAULT_EPSILON,
                 epsilon_decay: float = DEFAULT_EPSILON_DECAY,
                 epsilon_min: float = DEFAULT_EPSILON_MIN,
                 seed: int = 42):

        self.num_states   = num_states
        self.num_actions  = num_actions
        self.alpha        = alpha
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min  = epsilon_min
        self.rng          = np.random.default_rng(seed)

        # Q-table initialised to zeros (optimistic init can be used with small +ve)
        # Sutton & Barto Ch.2.6: initialising to 0 is fine for ε-greedy
        self.Q = np.zeros((num_states + 1, num_actions), dtype=np.float64)

        # Tracking
        self.episode = 0
        self.total_updates = 0
        self._td_errors: list[float] = []           # for diagnostics

    # ------------------------------------------------------------------
    # Action selection (Sutton & Barto Ch.2 Section 2.4 — ε-greedy)
    # ------------------------------------------------------------------

    def choose_action(self, state: int) -> int:
        """
        ε-greedy policy:
          - explore with probability ε  → uniform random action
          - exploit with probability 1-ε → greedy arg max Q(s, ·)

        Returns action INDEX (0–9), not the duration in seconds.
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))   # explore
        return int(np.argmax(self.Q[state]))                      # exploit

    def greedy_action(self, state: int) -> int:
        """Pure greedy (no exploration) — used for evaluation."""
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Q-learning update (Sutton & Barto Ch.6 Eq. 6.8)
    # ------------------------------------------------------------------

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool = False) -> float:
        """
        Apply the Q-learning update rule:

            Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s', a') − Q(s,a)]

        If done (terminal step): target = r  (no future reward)

        Returns the TD error (useful for diagnostics and convergence check).
        """
        q_current = self.Q[state, action]

        if done:
            # Terminal: Bellman target = r (Sutton & Barto Ch.3 Eq. 3.8)
            td_target = reward
        else:
            # Q-learning: off-policy → best possible next action
            # V(s') = max_a Q(s', a)   [Sutton & Barto Ch.6 Eq. 6.8]
            td_target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = td_target - q_current

        # Update: Q(s,a) ← Q(s,a) + α · δ
        self.Q[state, action] += self.alpha * td_error

        self.total_updates += 1
        self._td_errors.append(abs(td_error))
        return td_error

    # ------------------------------------------------------------------
    # Exploration decay (called once per episode)
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """
        Anneal ε after each episode (Sutton & Barto Ch.2 Section 2.7).
        ε decreases exponentially towards ε_min, reducing exploration over time.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1

    # ------------------------------------------------------------------
    # Value function (Sutton & Barto Ch.3 Eq. 3.8)
    # ------------------------------------------------------------------

    def value_function(self, state: int) -> float:
        """
        V(s) = max_a Q(s, a)
        The best expected cumulative reward from state s under the greedy policy.
        """
        return float(np.max(self.Q[state]))

    def best_action_duration(self, state: int) -> int:
        """Return the green-phase duration (seconds) for the greedy action."""
        return GREEN_PHASE_OPTIONS[self.greedy_action(state)]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def mean_td_error(self) -> float:
        """Mean absolute TD error over recent updates (convergence indicator)."""
        if not self._td_errors:
            return 0.0
        return float(np.mean(self._td_errors[-1000:]))

    def q_table_stats(self) -> dict:
        """Summary statistics of the Q-table."""
        return {
            "min":  float(np.min(self.Q)),
            "max":  float(np.max(self.Q)),
            "mean": float(np.mean(self.Q)),
            "std":  float(np.std(self.Q)),
            "nonzero_pct": float(np.mean(self.Q != 0) * 100),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save Q-table and hyperparameters to disk."""
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p.with_suffix(".npy")), self.Q)
        meta = {
            "alpha":         self.alpha,
            "gamma":         self.gamma,
            "epsilon":       self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min":   self.epsilon_min,
            "episode":       self.episode,
            "total_updates": self.total_updates,
        }
        p.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        logger.info(f"[QLearning] Q-table saved → {p.with_suffix('.npy')}")

    def load(self, path: str):
        """Load Q-table and hyperparameters from disk."""
        p = pathlib.Path(path)
        loaded_Q = np.load(str(p.with_suffix(".npy")))
        expected_shape = (self.num_states + 1, self.num_actions)
        if loaded_Q.shape != expected_shape:
            logger.warning(
                f"[QLearning] Q-table shape mismatch: loaded {loaded_Q.shape}, "
                f"expected {expected_shape}. Re-initializing to zeros."
            )
            self.Q = np.zeros(expected_shape, dtype=np.float64)
        else:
            self.Q = loaded_Q
        meta = json.loads(p.with_suffix(".json").read_text())
        self.alpha         = meta["alpha"]
        self.gamma         = meta["gamma"]
        self.epsilon       = meta["epsilon"]
        self.epsilon_decay = meta["epsilon_decay"]
        self.epsilon_min   = meta["epsilon_min"]
        self.episode       = meta.get("episode", 0)
        self.total_updates = meta.get("total_updates", 0)
        logger.info(f"[QLearning] Q-table loaded ← {p.with_suffix('.npy')}")

    def get_config(self) -> dict:
        return {
            "agent":         "Q-Learning",
            "alpha":         self.alpha,
            "gamma":         self.gamma,
            "epsilon":       self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min":   self.epsilon_min,
            "num_states":    self.num_states,
            "num_actions":   self.num_actions,
        }
