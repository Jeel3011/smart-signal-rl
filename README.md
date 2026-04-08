# Smart Signal 🚦
### Adaptive Traffic Signal Control via Reinforcement Learning & YOLOv8

> **IT567 Reinforcement Learning Project — 2026**
> A purely software-based adaptive traffic signal system that eliminates the need for costly SCATS/SCOOT hardware.

---

## Results

| Algorithm | Mean Wait (s) | Improvement |
|:----------|:--------:|---------:|
| Fixed Timer (baseline) | 32.8s | — |
| SARSA | 27.0s | ↓ 24% |
| **Q-Learning** | **20.5s** | **↓ 38%** |

---

## RL Mathematics (Sutton & Barto Ch.6)

**Q-learning update rule (Equation 6.8):**
```
Q(S,A) ← Q(S,A) + α · [R + γ · max_a Q(S', a) − Q(S,A)]
```

**SARSA update rule (Equation 6.7):**
```
Q(S,A) ← Q(S,A) + α · [R + γ · Q(S', A') − Q(S,A)]
```

**ε-greedy policy (Ch.2.4):** With probability ε → explore (random action); with probability 1−ε → exploit (greedy max Q).

**Value function (Ch.3):** `V(s) = max_a Q(s, a)`

### Hyperparameters (Grid-search tuned)
| Symbol | Search Range | Selected |
|:-------|:------------|:---------|
| α (learning rate) | 0.01, 0.05, 0.1, 0.2, 0.5 | tuned |
| γ (discount factor) | 0.1, 0.5, 0.7, 0.9, 0.99 | tuned |
| ε (exploration) | 0.3, 0.5, 0.7, 0.9 | tuned |
| ε-decay | 0.995, 0.999 | tuned |

---

## Architecture

```
Camera Frame (or SUMO screenshot)
       │
       ▼
  YOLOv8n Detection          ← pretrained COCO, no fine-tuning
  (car/bus/truck/person)
       │
       ▼
  Weighted Density            ← 1×cars + 2×trucks + 3×buses + 0.5×people
  State ∈ {1, ..., 100}
       │
       ▼
  Q-Learning Agent            ← tabular Q-table (100 × 10)
  Action ∈ {10, 20, ..., 100}s green
       │
       ▼
  SUMO Simulation (TraCI)     ← simulates traffic response
       │
       ▼
  Reward signal: +2/+1/−1    ← based on density change
```

---

## Quick Start

### 1. Install dependencies
```bash
# Install SUMO (macOS)
brew install sumo
echo 'export SUMO_HOME="/opt/homebrew/share/sumo"' >> ~/.zshrc
source ~/.zshrc

# Install Python packages
pip install -r requirements.txt
```

### 2. Build SUMO network
```bash
python sumo_env/build_network.py
python sumo_env/generate_routes.py
```

### 3. Train (fast mode — no SUMO required)
```bash
python train.py --no-sumo --episodes 500
```

### 4. Train with hyperparameter tuning
```bash
python train.py --no-sumo --tune --episodes 500
```

### 5. Train on SUMO environment (recommended for demo)
```bash
python train.py --episodes 200
```

### 6. Run demo
```bash
python demo.py              # interactive demo
python demo.py --no-sumo    # YOLO + Q-table only
```

### 7. Launch dashboard
```bash
uvicorn api.server:app --host 127.0.0.1 --port 8000
# Open: http://127.0.0.1:8000
```

---

## Project Structure

```
RL_PROJECT/
├── config/
│   └── config.yaml              # All hyperparameters
├── sumo_env/
│   ├── environment.py           # SUMO TraCI environment (MDP)
│   ├── build_network.py         # Build intersection.net.xml
│   ├── generate_routes.py       # Generate traffic.rou.xml
│   ├── intersection.*.xml       # SUMO network files
│   └── screenshots/             # Captured for YOLO demo
├── detection/
│   └── yolo_detector.py         # YOLOv8n inference + density calc
├── agents/
│   ├── q_learning_agent.py      # Tabular Q-learning (Sutton & Barto Ch.6)
│   └── sarsa_agent.py           # SARSA (on-policy TD control)
├── training/
│   ├── trainer.py               # Episode training loop
│   ├── tuner.py                 # Grid-search hyperparameter tuning
│   └── evaluator.py             # Comparison + all plots
├── api/
│   └── server.py                # FastAPI backend
├── dashboard/
│   ├── index.html               # Web dashboard UI
│   ├── style.css                # Dark glassmorphism design
│   └── app.js                   # Dashboard logic
├── results/
│   ├── plots/                   # Generated charts
│   └── q_tables/                # Saved Q-tables
├── train.py                     # Main entry point
├── demo.py                      # Demonstration script
└── requirements.txt
```

---

## References

1. Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction*. MIT Press. Ch.2, Ch.3, Ch.6.
2. Research paper: *Smart Signal — Adaptive Traffic Signal Control using RL and Object Detection* (I-SMAC 2019, IEEE).
3. Redmon, J. & Farhadi, A. (2018). YOLOv3: An Incremental Improvement.
4. Eclipse SUMO — Simulation of Urban MObility. https://sumo.dlr.de
5. LucasAlegre/sumo-rl. https://github.com/LucasAlegre/sumo-rl
