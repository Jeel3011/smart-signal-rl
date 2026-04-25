#!/usr/bin/env python3
"""
sarsa_agent.py — Tabular SARSA (on-policy TD control) agent.

Implements SARSA from:
  Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An Introduction.
  Chapter 6, Section 6.4, Equation 6.7:

      Q(S,A) ← Q(S,A) + α · [R + γ · Q(S', A') − Q(S,A)]

KEY DIFFERENCE from Q-learning:
  - SARSA is ON-POLICY: it updates Q using the action A' actually taken in S'
    (which may be exploratory), whereas Q-learning uses max_a Q(S', a).
  - This makes SARSA more conservative — it accounts for future exploration cost.
  - In our experiments, Q-learning outperforms SARSA (~38% vs ~24% wait reduction)
    exactly because Q-learning learns the optimal policy directly.

In code, SARSA requires passing the NEXT action to update():
    a_next = agent.choose_action(s_next)
    agent.update(s, a, r, s_next, a_next)
    a = a_next  # carry forward
"""

import numpy as np
import json
import pathlib
import logging

logger = logging.getLogger(__name__)

NUM_STATES  = 100
NUM_ACTIONS = 10
GREEN_PHASE_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


class SARSAAgent:
    """
    Tabular SARSA agent (on-policy TD control) — Sutton & Barto Ch.6 Section 6.4.

    Update rule:
        Q(s,a) ← Q(s,a) + α · [r + γ · Q(s', a') − Q(s,a)]

    where a' is the action actually CHOSEN in s' by the ε-greedy policy.
    """

    def __init__(self,
                 num_states: int = NUM_STATES,
                 num_actions: int = NUM_ACTIONS,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.7,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.05,
                 seed: int = 42):

        self.num_states    = num_states
        self.num_actions   = num_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.rng           = np.random.default_rng(seed)

        self.Q             = np.zeros((num_states + 1, num_actions), dtype=np.float64)
        self.episode       = 0
        self.total_updates = 0
        self._td_errors: list[float] = []

    def choose_action(self, state: int) -> int:
        """ε-greedy action selection — same as Q-learning."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        return int(np.argmax(self.Q[state]))

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int, done: bool = False) -> float:
        """
        SARSA update (Sutton & Barto Ch.6 Eq. 6.7):

            Q(s,a) ← Q(s,a) + α · [r + γ · Q(s', a') − Q(s,a)]

        next_action is the action ACTUALLY chosen in next_state (on-policy).
        """
        q_current = self.Q[state, action]

        if done:
            td_target = reward
        else:
            # SARSA: use Q(s', a') where a' is the actual next action
            td_target = reward + self.gamma * self.Q[next_state, next_action]

        td_error = td_target - q_current
        self.Q[state, action] += self.alpha * td_error

        self.total_updates += 1
        self._td_errors.append(abs(td_error))
        return td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1

    def value_function(self, state: int) -> float:
        """V(s) = max_a Q(s, a) under greedy policy."""
        return float(np.max(self.Q[state]))

    def best_action_duration(self, state: int) -> int:
        return GREEN_PHASE_OPTIONS[self.greedy_action(state)]

    def mean_td_error(self) -> float:
        if not self._td_errors:
            return 0.0
        return float(np.mean(self._td_errors[-1000:]))

    def save(self, path: str):
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p.with_suffix(".npy")), self.Q)
        meta = {
            "alpha": self.alpha, "gamma": self.gamma,
            "epsilon": self.epsilon, "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min, "episode": self.episode,
            "total_updates": self.total_updates,
        }
        p.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    def load(self, path: str):
        p = pathlib.Path(path)
        loaded_Q = np.load(str(p.with_suffix(".npy")))
        expected_shape = (self.num_states + 1, self.num_actions)
        if loaded_Q.shape != expected_shape:
            logger.warning(
                f"[SARSA] Q-table shape mismatch: loaded {loaded_Q.shape}, "
                f"expected {expected_shape}. Re-initializing to zeros."
            )
            self.Q = np.zeros(expected_shape, dtype=np.float64)
        else:
            self.Q = loaded_Q
        meta = json.loads(p.with_suffix(".json").read_text())
        self.alpha = meta["alpha"]; self.gamma = meta["gamma"]
        self.epsilon = meta["epsilon"]
        self.episode = meta.get("episode", 0)
        self.total_updates = meta.get("total_updates", 0)

    def get_config(self) -> dict:
        return {"agent": "SARSA", "alpha": self.alpha, "gamma": self.gamma,
                "epsilon": self.epsilon, "epsilon_decay": self.epsilon_decay}
