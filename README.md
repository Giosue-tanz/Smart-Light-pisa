<div align="center">

# 🚦 Smart Light Pisa

### Adaptive Traffic Signal Control via Reinforcement Learning & Computer Vision

[![License: Custom](https://img.shields.io/badge/License-Custom%20(Commercial%20w/%20Royalties)-red.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-orange.svg)](https://pytorch.org/)
[![SUMO](https://img.shields.io/badge/SUMO-Traffic%20Simulation-yellow.svg)](https://sumo.dlr.de/)
[![Status](https://img.shields.io/badge/Status-Open%20for%20Continuation-brightgreen.svg)]()

> **Winner of a sustainable mobility call for proposals · Presented at Bright 2025 · Developed in collaboration with PISAMO and the Contamination Lab**

</div>

---

## 📖 What is Smart Light?

**Smart Light** is an AI-powered traffic management system designed for the urban road network of **Pisa, Italy**. Rather than relying on fixed-cycle timers — the standard deployed in most Italian cities since decades — Smart Light uses **Deep Reinforcement Learning** and **Computer Vision** to adaptively control traffic signals in real time.

The system observes the current state of each intersection (queue lengths, waiting times, vehicle counts) and continuously learns the optimal phase-switching policy to:

- ⏱️ **Reduce average vehicle waiting time**
- 🌿 **Cut CO₂ emissions** from idling vehicles
- 🚑 **Prioritize emergency vehicles** automatically
- 🚶 **Improve pedestrian safety** at crossings
- 📡 **Operate at the edge** — no cloud dependency required

The entire pipeline — from urban simulation to RL training to live inference — is **fully implemented, tested, and documented**.

---

## 🏆 Results

Benchmark run across **50 episodes × 120 steps**, comparing our RL agents against a traditional Fixed Timer baseline on the Pisa road network:

| Metric | Fixed Timer | DQN v3.4 | PPO v4.0 |
|--------|-------------|-----------|----------|
| **Avg. Reward** | -54.95 | -51.89 | **-49.92** |
| **Avg. Waiting Time** | 41.0 s | 36.5 s | **35.5 s** |
| **Reward improvement vs baseline** | — | +5.6% | **+9.2%** |
| **Waiting time improvement** | — | +11.0% | **+13.4%** |

> 🥇 **Winner: PPO v4.0** — best reward (+1.98 over DQN v3.4), 13.4% less average waiting time vs. fixed timers.

Earlier benchmark (Smart Light v1.9 vs Fixed Timer, single intersection):

| Metric | Fixed Timer (30s) | Smart AI (v1.9) | Improvement |
|--------|-------------------|-----------------|-------------|
| **Vehicles cleared** | 1190 | 1197 | **+7** |
| **Avg. Wait** | 1450.0 s | 1368.3 s | **−81.7 s** |
| **Max queue** | 975 | 925 | **−50** |
| **CO₂ (kg)** | 207.06 | 196.55 | **−10.52 kg** |

---

## 🧠 Algorithms Implemented

| Algorithm | Versions | Description |
|-----------|----------|-------------|
| **DQN** | v1.8 → v3.1 | Deep Q-Network, single-intersection |
| **Dueling DQN** | v3.3 → v3.4 | Separate value/advantage streams |
| **PPO** | v4.0 | Proximal Policy Optimization |
| **Multi-Agent** | ep5 → ep200 + best | Distributed control across multiple intersections |
| **Edge-First** | best | Architecture optimized for edge hardware |

All trained model weights (`.pth` checkpoints) are included in the repository.

---

## 🏗️ Architecture

```
smartlight_pisa/
│
├── 1_simulation/              # SUMO simulation setup
│   ├── sumo_config/           # Road network, routes, SUMO config
│   ├── smartlight_traffic_gen.py       # Synthetic traffic generation
│   ├── genera_scenario_medio.py        # Medium-load scenario generator
│   ├── trips.trips.xml                 # SUMO trip files (real Pisa network)
│   └── lancia_osm_wizard.bat           # OSM → SUMO conversion helper
│
├── 2_environment/             # Reinforcement Learning environments
│   ├── smartlight_sumo_env.py           # Single-agent env (Gymnasium-compatible)
│   └── smartlight_multi_agent_env.py    # Multi-agent env (N intersections)
│
├── 3_training/                # Training scripts
│   ├── smartlight_train_pisa_real.py    # Training on the real Pisa network
│   ├── smartlight_train_simulated.py    # Training on synthetic scenarios
│   ├── smartlight_train_ppo.py          # PPO training pipeline
│   ├── smartlight_train_parallel.py     # Parallel DQN training (multi-CPU)
│   └── smartlight_train_multi_agent.py  # Multi-agent coordination training
│
├── 4_evaluation/              # Benchmarks & evaluation
│   ├── scripts/
│   │   ├── benchmark_dqn_vs_ppo.py          # Head-to-head DQN vs PPO benchmark
│   │   ├── benchmark_baseline.py             # Fixed-timer baseline reference
│   │   ├── smartlight_benchmark.py           # General benchmark runner
│   │   └── smartlight_benchmark_multi_agent.py
│   └── results/               # Benchmark results (markdown + txt)
│
├── 5_deployment/              # Inference & demo
│   ├── smartlight_run_trained_model.py  # Load and run a trained agent
│   ├── smartlight_live_video_dqn.py     # Live camera feed inference (DQN)
│   └── pitch_animation.html             # Interactive pitch animation
│
├── checkpoints/               # Trained model weights (.pth)
│   ├── dqn/                   # DQN v1.8 → v3.1
│   ├── dueling_dqn/           # Dueling DQN v3.3 → v3.4
│   ├── ppo/                   # PPO v4.0
│   ├── multi_agent/           # Multi-agent (ep5 → ep200 + best)
│   └── edge_first/            # Edge-first architecture
│
├── map_analysis/              # Road network & traffic light analysis
│   ├── scripts/               # TraCI & OSM mapping scripts
│   └── output/                # Generated reports and maps
│
├── data/                      # Dataset (YOLO images & labels)
│
├── docs/                      # Full project documentation
│   ├── Business_Strategy.pdf           # Business plan & market analysis
│   ├── Roadmap.pdf                     # Technical development roadmap
│   ├── Shadow_Mode_Testing_Plan.pdf    # Real-world shadow-mode test plan
│   └── Executive_Briefing_Pisamo.pdf   # Executive brief for PISAMO
│
├── requirements.txt           # Python dependencies
└── sumo_tools/                # Local SUMO installation
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Traffic Simulation** | [SUMO](https://sumo.dlr.de/) — Simulation of Urban Mobility |
| **RL Interface** | [TraCI](https://sumo.dlr.de/docs/TraCI.html) — Traffic Control Interface |
| **RL Framework** | [PyTorch 2.9](https://pytorch.org/) |
| **RL Environments** | [Gymnasium](https://gymnasium.farama.org/) (OpenAI Gym compatible) |
| **Computer Vision** | [YOLOv8](https://github.com/ultralytics/ultralytics) (optional live inference) |
| **Analysis & Viz** | Matplotlib, Seaborn, Streamlit, Altair |
| **Map Data** | OpenStreetMap (OSM) → SUMO network converter |
| **Python** | 3.10+ |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [SUMO](https://sumo.dlr.de/docs/Installing/index.html) installed and on PATH
- (Optional) CUDA GPU for faster training

### Installation

```bash
# Clone the repository
git clone https://github.com/Giosue-tanz/Smart-Light-pisa.git
cd Smart-Light-pisa

# Create and activate a virtual environment
python -m venv smartlight_env
source smartlight_env/bin/activate        # Linux/macOS
# .\smartlight_env\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run a Quick Test

```bash
python 4_evaluation/scripts/test_quick.py
```

### Train an Agent

```bash
# Train DQN on the real Pisa road network
python 3_training/smartlight_train_pisa_real.py

# Train with PPO
python 3_training/smartlight_train_ppo.py

# Train multi-agent (all intersections simultaneously)
python 3_training/smartlight_train_multi_agent.py

# Parallel DQN training (uses multiple CPU cores)
python 3_training/smartlight_train_parallel.py
```

### Evaluate & Benchmark

```bash
# Head-to-head: DQN v3.4 vs PPO v4.0 vs Fixed Timer
python 4_evaluation/scripts/benchmark_dqn_vs_ppo.py

# Multi-agent benchmark
python 4_evaluation/scripts/smartlight_benchmark_multi_agent.py
```

### Deploy a Trained Model

```bash
# Run the best trained agent (PPO v4.0) in simulation
python 5_deployment/smartlight_run_trained_model.py

# Live video inference from camera feed (DQN)
python 5_deployment/smartlight_live_video_dqn.py
```

---

## 📄 Documentation

All technical and strategic documents are in the [`docs/`](./docs/) folder:

| Document | Description |
|----------|-------------|
| [Business Strategy](docs/Business_Strategy.pdf) | Market analysis, business model, scalability plan |
| [Roadmap](docs/Roadmap.pdf) | Technical development phases and milestones |
| [Shadow Mode Testing Plan](docs/Shadow_Mode_Testing_Plan.pdf) | Real-world deployment testing methodology |
| [Executive Briefing — PISAMO](docs/Executive_Briefing_Pisamo.pdf) | Briefing prepared for Pisa's municipal transport authority |

---

## 🗺️ Open Challenges & Roadmap

This project is **open for continuation**. Below are the next natural steps:

- [ ] **Shadow Mode Deployment** — install sensors alongside existing traffic lights in Pisa; run the model in observation-only mode, validating predictions against ground truth
- [ ] **Hardware Integration** — connect IoT sensors (inductive loops, cameras) via MQTT/REST to replace simulated inputs
- [ ] **Computer Vision Pipeline** — complete integration of YOLOv8 real-time vehicle detection as live state input
- [ ] **Emergency Vehicle Prioritization** — expand multi-agent coordination for buses, ambulances and bicycle lanes
- [ ] **Operator Dashboard** — build a real-time Streamlit/web dashboard for city operators
- [ ] **Federated Learning** — extend to a city-wide multi-intersection federated learning setup
- [ ] **Hardware-in-the-loop** — test the edge-first model on Raspberry Pi / NVIDIA Jetson Nano

---

## 🤝 Open to Continuation

This repository is **fully open** — to researchers, students, engineers, or municipalities who want to carry this work forward.

The project was validated through:
- A **won call for proposals** on sustainable urban mobility
- A **pitch at Bright 2025**
- Meetings with **PISAMO** (Pisa's public transport authority) and the **Contamination Lab**

Everything is documented. The code runs. The models are trained. What's missing is the next person — or team — to take it further.

---

## 💬 A Note from the Author

> *Every journey has its turning points. Smart Light was one of them — perhaps the most significant one so far.*
>
> *The project was born from a simple but urgent observation: traffic light systems in cities like Pisa are still managed with logic from decades ago, while the technology to change this exists today. The team designed a system based on computer vision, reinforcement learning and IoT sensors, capable of adapting traffic light cycles in real time — reducing waiting times, cutting emissions, and improving safety for pedestrians.*
>
> *We won a call for sustainable mobility. We presented at Bright 2025. We met with PISAMO, the Contamination Lab, and people who believed in the idea. It was real.*
>
> *And yet, today the team has decided to put the project on hold. Not because the idea wasn't valid — it still is. But because life moves forward: the path ahead leads toward a master's degree, new challenges, and new directions that require full commitment.*
>
> *Smart Light remains here, fully documented. The technical plan, the architecture, the research — it's all preserved.*
>
> *If you're someone who sees potential in this project and wants to carry it forward, I'm open to a conversation. Don't hesitate to get in touch.*
>
> — **Giosuè Aiello**

📩 **Contact**: [LinkedIn — Giosuè Aiello](https://www.linkedin.com/in/giosue-aiello) · or open a [GitHub Issue](https://github.com/Giosue-tanz/Smart-Light-pisa/issues) on this repository

---

## 📜 License

This project is released under a **Custom Commercial and Academic License**. 
- **Academic and personal use** is completely free and encouraged.
- **Commercial use** (integrating it into business services, products, or public infrastructure for profit) is strictly prohibited without a prior written commercial agreement.

If you intend to use this project commercially, you must contact the author to establish a royalty or licensing agreement.
See [`LICENSE`](./LICENSE) for full details.

<div align="center">

*Built with ❤️ in Pisa · 2024–2026*

</div>
