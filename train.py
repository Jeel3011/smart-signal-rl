#!/usr/bin/env python3
"""
train.py — Main training entry point for Smart Signal.

Workflow:
  1. [Optional] Hyperparameter tuning (grid search over α, γ, ε, ε_decay)
  2. Load best params (or use defaults from config.yaml)
  3. Build SUMO environment
  4. Train Q-learning agent
  5. Train SARSA agent  
  6. Evaluate all three (Fixed, SARSA, Q-learning) and compare
  7. Save all plots and results

Usage:
  python train.py                          # full pipeline with default config
  python train.py --tune                   # run hyperparameter tuning first
  python train.py --episodes 300           # override episode count
  python train.py --agent q_learning       # train only Q-learning
  python train.py --no-sumo                # use fast env (no SUMO required)
"""

import argparse
import json
import logging
import pathlib
import sys
import os

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))


def load_config(path: str = "config/config.yaml") -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text())


def make_env(cfg: dict, use_gui: bool = False, fast_mode: bool = False):
    """Create the appropriate environment."""
    if fast_mode:
        from training.tuner import FastTrafficEnv

        class _FastEnvAdapter:
            """Thin adapter to expose same API as SumoEnvironment."""
            def __init__(self, cfg):
                self._env = FastTrafficEnv(seed=cfg["simulation"]["seed"])
                self._wait = 0.0

            def reset(self, episode=0):
                self._env.rng = __import__("numpy").random.default_rng(
                    cfg["simulation"]["seed"] + episode
                )
                return self._env.reset()

            def step(self, action):
                s, r, done = self._env.step(action)
                return s, r, done, {}

            def get_mean_wait_time(self):
                return self._env.mean_wait_per_phase()

            def close(self):
                pass

        return _FastEnvAdapter(cfg)
    else:
        from sumo_env.environment import SumoEnvironment
        return SumoEnvironment(
            use_gui=use_gui,
            episode_length=cfg["simulation"]["episode_length"],
            step_length=cfg["simulation"]["sumo_step_length"],
            seed=cfg["simulation"]["seed"],
            vehicles_per_hour=600,
        )


def run_tuning(cfg: dict, agent_type: str) -> dict:
    from training.tuner import tune
    return tune(agent_type=agent_type, config_path="config/config.yaml", verbose=True)


def make_agent(agent_type: str, params: dict, cfg: dict):
    """Instantiate agent with given hyperparameters."""
    if agent_type == "q_learning":
        from agents.q_learning_agent import QLearningAgent
        return QLearningAgent(
            alpha         = params.get("alpha",         cfg["q_learning"]["alpha"]),
            gamma         = params.get("gamma",         cfg["q_learning"]["gamma"]),
            epsilon       = params.get("epsilon",       cfg["q_learning"]["epsilon"]),
            epsilon_decay = params.get("epsilon_decay", cfg["q_learning"]["epsilon_decay"]),
            epsilon_min   = cfg["q_learning"]["epsilon_min"],
            seed          = cfg["simulation"]["seed"],
        )
    else:
        from agents.sarsa_agent import SARSAAgent
        return SARSAAgent(
            alpha         = params.get("alpha",         cfg["sarsa"]["alpha"]),
            gamma         = params.get("gamma",         cfg["sarsa"]["gamma"]),
            epsilon       = params.get("epsilon",       cfg["sarsa"]["epsilon"]),
            epsilon_decay = params.get("epsilon_decay", cfg["sarsa"]["epsilon_decay"]),
            epsilon_min   = cfg["sarsa"]["epsilon_min"],
            seed          = cfg["simulation"]["seed"],
        )


def main():
    parser = argparse.ArgumentParser(description="Smart Signal — Train RL Agents")
    parser.add_argument("--config",   default="config/config.yaml")
    parser.add_argument("--tune",     action="store_true", help="Run hyperparameter tuning first")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--agent",    choices=["q_learning", "sarsa", "both"],  default="both")
    parser.add_argument("--no-sumo",  action="store_true", help="Use fast env (no SUMO)")
    parser.add_argument("--gui",      action="store_true", help="Launch SUMO with GUI")
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_episodes = args.episodes or cfg["simulation"]["num_episodes"]
    fast_mode    = args.no_sumo

    pathlib.Path("results/q_tables").mkdir(parents=True, exist_ok=True)

    agents_to_train = (["q_learning", "sarsa"]
                       if args.agent == "both" else [args.agent])

    training_logs = {}
    trained_agents = {}
    best_params = {}

    # ------------------------------------------------------------------
    # Step 1: Hyperparameter tuning (optional)
    # ------------------------------------------------------------------
    if args.tune:
        print("\n📊  Running hyperparameter tuning...\n")
        for agent_type in agents_to_train:
            params = run_tuning(cfg, agent_type)
            best_params[agent_type] = params
            # Persist for later use
            pathlib.Path(f"results/best_params_{agent_type}.json").write_text(
                json.dumps(params, indent=2)
            )
    else:
        # Try to load existing best params, fall back to config defaults
        for agent_type in agents_to_train:
            bp_file = pathlib.Path(f"results/best_params_{agent_type}.json")
            if bp_file.exists():
                best_params[agent_type] = json.loads(bp_file.read_text())
                print(f"  Loaded best params for {agent_type} from {bp_file}")
            else:
                best_params[agent_type] = {}

    # ------------------------------------------------------------------
    # Step 2: Train agents
    # ------------------------------------------------------------------
    from training.trainer import Trainer

    for agent_type in agents_to_train:
        print(f"\n🚦  Training {agent_type.upper()}...")
        params = best_params.get(agent_type, {})
        agent  = make_agent(agent_type, params, cfg)
        env    = make_env(cfg, use_gui=args.gui, fast_mode=fast_mode)

        trainer = Trainer(
            agent         = agent,
            env           = env,
            num_episodes  = num_episodes,
            agent_name    = agent_type,
            save_dir      = "results/q_tables",
        )
        log = trainer.train(verbose=True)
        training_logs[agent_type]  = log
        trained_agents[agent_type] = agent
        env.close()

    # ------------------------------------------------------------------
    # Step 3: Evaluation & comparison
    # ------------------------------------------------------------------
    print("\n📈  Evaluating all controllers...\n")
    from training.evaluator import Evaluator, FixedTimerController

    eval_env = make_env(cfg, use_gui=False, fast_mode=fast_mode)
    evaluator = Evaluator(eval_env, num_eval_episodes=10)

    evaluator.evaluate_controller(FixedTimerController(), "Fixed Timer")

    if "sarsa" in trained_agents:
        evaluator.evaluate_controller(trained_agents["sarsa"], "SARSA", is_sarsa=True)
    if "q_learning" in trained_agents:
        evaluator.evaluate_controller(trained_agents["q_learning"], "Q-Learning")

    eval_env.close()

    evaluator.print_comparison_table()
    evaluator.save_results()

    # ------------------------------------------------------------------
    # Step 4: Generate all plots
    # ------------------------------------------------------------------
    print("🎨  Generating plots...")
    evaluator.plot_training_curves(training_logs)
    evaluator.plot_comparison_bar()

    if "q_learning" in trained_agents:
        evaluator.plot_q_table_heatmap(trained_agents["q_learning"],
                                       "Q-Learning — Learned Signal Policy")
    if "q_learning" in training_logs:
        evaluator.plot_epsilon_decay(training_logs["q_learning"])
        evaluator.plot_td_error(training_logs["q_learning"])

    print("\n✅  All done! Results in results/ directory.")
    print("    Run  python demo.py  to launch the interactive demo.\n")


if __name__ == "__main__":
    main()
