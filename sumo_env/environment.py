#!/usr/bin/env python3
"""
environment.py — SUMO TraCI environment wrapper for Smart Signal.

Implements the MDP interface described in Sutton & Barto Ch. 3:
  - State  s  : traffic density scalar in [1, 100]
  - Action a  : green phase duration from {10, 20, ..., 100} seconds
  - Reward r  : +2 / +1 / -1 based on density change

Reference:
  Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An Introduction.
  Ch. 3 — Finite Markov Decision Processes
  Ch. 6 — Temporal-Difference Learning (Q-learning update rule)
"""

import os
import sys
import time
import pathlib
import subprocess
import logging
from typing import Tuple, Optional

import traci
import traci.constants as tc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = pathlib.Path(__file__).parent
NET_FILE   = BASE_DIR / "intersection.net.xml"
ROUTE_FILE = BASE_DIR / "traffic.rou.xml"
SUMO_CFG   = BASE_DIR / "intersection.sumocfg"
SCREENSHOT_DIR = BASE_DIR / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Action / state constants (must match config.yaml)
# ---------------------------------------------------------------------------
GREEN_PHASE_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # seconds
NUM_ACTIONS  = len(GREEN_PHASE_OPTIONS)
NUM_STATES   = 100   # density bins: 1 → 100
YELLOW_DURATION = 4  # seconds

# TLS phase indices (must match intersection.tll.xml)
NS_GREEN   = 0   # North-South green
NS_YELLOW  = 1
EW_GREEN   = 2   # East-West green
EW_YELLOW  = 3

# COCO class IDs for vehicle detection (used by density weighting)
CLASS_WEIGHTS = {
    0: 0.5,   # person
    2: 1.0,   # car
    5: 3.0,   # bus
    7: 2.0,   # truck
}

TLS_ID   = "center"
IN_LANES = ["north_in_0", "south_in_0", "east_in_0", "west_in_0"]


class SumoEnvironment:
    """
    Wraps a SUMO simulation as an MDP environment.

    Usage:
        env = SumoEnvironment(use_gui=False)
        state = env.reset(episode=0)
        done = False
        while not done:
            action_idx = agent.choose_action(state)
            state, reward, done, info = env.step(action_idx)
        env.close()
    """

    def __init__(self, use_gui: bool = False, episode_length: int = 3600,
                 step_length: float = 1.0, seed: int = 42,
                 vehicles_per_hour: int = 600):
        self.use_gui         = use_gui
        self.episode_length  = episode_length
        self.step_length     = step_length
        self.seed            = seed
        self.vehicles_per_hour = vehicles_per_hour

        self._sumo_proc: Optional[subprocess.Popen] = None
        self._sim_time   = 0
        self._phase_time = 0          # time spent in current green phase
        self._current_phase = NS_GREEN
        self._density_at_phase_start = 0
        self._density_midpoint_checked = False
        self._episode_num = 0

        # Cumulative episode metrics
        self.total_wait_time  = 0.0
        self.total_vehicles   = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, episode: int = 0) -> int:
        """Start a new SUMO episode, return initial state (density)."""
        self._episode_num = episode
        if traci.isLoaded():
            traci.close()
            time.sleep(0.2)

        # Regenerate routes with slight seed variation per episode
        self._regenerate_routes(episode)

        # Build SUMO command
        sumo_bin  = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd  = [
            sumo_bin,
            "-c", str(SUMO_CFG),
            "--step-length", str(self.step_length),
            "--random",      "false",
            "--seed",        str(self.seed + episode),
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--time-to-teleport", "-1",   # disable teleporting (keep cars queued)
        ]
        traci.start(sumo_cmd)

        # Reset counters
        self._sim_time   = 0
        self._phase_time = 0
        self._current_phase = NS_GREEN
        self._density_at_phase_start = 0
        self._density_midpoint_checked = False
        self.total_wait_time = 0.0
        self.total_vehicles  = 0

        # Set initial TLS phase
        traci.trafficlight.setPhase(TLS_ID, NS_GREEN)

        # Advance a few steps to populate vehicles
        for _ in range(10):
            traci.simulationStep()
            self._sim_time += self.step_length

        state = self._get_state()
        self._density_at_phase_start = state
        return state

    def step(self, action_idx: int) -> Tuple[int, float, bool, dict]:
        """
        Apply action (green duration), run simulation, return (s', r, done, info).

        Reward structure (Sutton & Barto Ch.3 MDP reward signal):
          +2 : density fell below ⅓ of phase-start by midpoint  (fast clearance)
          +1 : density decreased at end of green phase            (good clearance)
          -1 : density did NOT decrease                           (congestion)
        """
        green_duration = GREEN_PHASE_OPTIONS[action_idx]
        midpoint       = green_duration / 2
        reward         = 0.0
        midpoint_bonus_given = False

        density_start = self._get_state()
        self._density_at_phase_start = density_start

        # --- Run the green phase step-by-step ---
        elapsed = 0
        while elapsed < green_duration and self._sim_time < self.episode_length:
            traci.simulationStep()
            self._sim_time  += self.step_length
            elapsed         += self.step_length

            # Accumulate waiting time from all vehicles
            for veh_id in traci.vehicle.getIDList():
                self.total_wait_time += traci.vehicle.getWaitingTime(veh_id)
            self.total_vehicles = traci.simulation.getArrivedNumber()

            # Midpoint check (Sutton & Barto: intermediate reward signal)
            if not midpoint_bonus_given and elapsed >= midpoint:
                mid_density = self._get_state()
                if density_start > 0 and mid_density < density_start / 3:
                    reward += 2.0
                    midpoint_bonus_given = True

        # --- End-of-phase evaluation ---
        density_end = self._get_state()
        if density_end < density_start:
            reward += 1.0
        else:
            reward -= 1.0

        # --- Yellow transition phase ---
        yellow_phase = NS_YELLOW if self._current_phase == NS_GREEN else EW_YELLOW
        traci.trafficlight.setPhase(TLS_ID, yellow_phase)
        for _ in range(int(YELLOW_DURATION / self.step_length)):
            if self._sim_time >= self.episode_length:
                break
            traci.simulationStep()
            self._sim_time += self.step_length

        # --- Switch to next green phase ---
        self._current_phase = EW_GREEN if self._current_phase == NS_GREEN else NS_GREEN
        traci.trafficlight.setPhase(TLS_ID, self._current_phase)

        done  = self._sim_time >= self.episode_length
        state = self._get_state()

        info = {
            "sim_time":         self._sim_time,
            "density_start":    density_start,
            "density_end":      density_end,
            "green_duration":   green_duration,
            "total_wait_time":  self.total_wait_time,
        }
        return state, reward, done, info

    def get_mean_wait_time(self) -> float:
        """Mean waiting time per vehicle for this episode (seconds)."""
        if self.total_vehicles == 0:
            return 0.0
        return self.total_wait_time / max(self.total_vehicles, 1)

    def get_vehicle_counts(self, weighted: bool = True) -> dict:
        """
        Return current vehicle counts per lane.
        If weighted=True, applies class weights for density calculation.
        """
        counts = {lane: 0 for lane in IN_LANES}
        for lane in IN_LANES:
            try:
                counts[lane] = traci.lane.getLastStepVehicleNumber(lane)
            except traci.exceptions.TraCIException:
                counts[lane] = 0
        return counts

    def capture_screenshot(self, step: int) -> pathlib.Path:
        """Save a SUMO-GUI screenshot (only meaningful when use_gui=True)."""
        if not self.use_gui:
            return None
        path = SCREENSHOT_DIR / f"ep{self._episode_num:04d}_step{step:06d}.png"
        try:
            traci.gui.screenshot("View #0", str(path))
        except Exception:
            pass
        return path

    def close(self):
        """Cleanly close the SUMO connection."""
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> int:
        """
        Compute traffic density across all inbound lanes.
        Returns a scalar in [1, NUM_STATES] (clamped).

        Density = weighted sum of vehicles per class, normalised to 1-100.
        Since TraCI doesn't give class info per lane, we use vehicle counts
        and scale by lane capacity (50 vehicles per 500m lane ≈ jam density).
        """
        total_vehicles = 0
        for lane in IN_LANES:
            try:
                total_vehicles += traci.lane.getLastStepVehicleNumber(lane)
            except traci.exceptions.TraCIException:
                pass

        # Normalise: 0 vehicles → density 1, 50+ vehicles → density 100
        max_vehicles = 50
        density = int((total_vehicles / max_vehicles) * (NUM_STATES - 1)) + 1
        return max(1, min(NUM_STATES, density))

    def _regenerate_routes(self, episode: int):
        """Re-generate route file with episode-specific seed for variety."""
        import importlib.util, sys as _sys
        gen_path = BASE_DIR / "generate_routes.py"
        spec = importlib.util.spec_from_file_location("gen_routes", gen_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Vary density slightly per episode for robustness
        density = self.vehicles_per_hour + (episode % 5) * 60
        mod.generate(seed=self.seed + episode,
                     episode_length=self.episode_length,
                     vehicles_per_hour=density,
                     output=ROUTE_FILE)


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    env = SumoEnvironment(use_gui=False, episode_length=300)
    state = env.reset(episode=0)
    print(f"Initial state (density): {state}")
    for step in range(5):
        action = 3  # 40s green
        s2, r, done, info = env.step(action)
        print(f"  step={step+1} | density={s2:3d} | reward={r:+.1f} | done={done}")
        if done:
            break
    print(f"Mean wait time: {env.get_mean_wait_time():.2f}s")
    env.close()
