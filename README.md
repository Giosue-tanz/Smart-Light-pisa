# 🚦 SmartLight Pisa

**Ottimizzazione adattiva dei semafori nella rete stradale di Pisa tramite Reinforcement Learning e simulazione SUMO.**

---

## 📁 Struttura del Progetto

```
smartlight_pisa/
│
├── 1_simulation/              # Setup simulazione SUMO
│   ├── sumo_config/           # Rete stradale, rotte, configurazione SUMO
│   ├── traffic_gen.py         # Generazione traffico sintetico
│   ├── genera_scenario_medio.py
│   ├── trips.trips.xml        # File viaggi SUMO
│   └── lancia_osm_wizard.bat  # Script conversione OSM → SUMO
│
├── 2_environment/             # Ambienti di Reinforcement Learning
│   ├── smartlight_sumo_env.py           # Env single-agent (Gymnasium)
│   └── smartlight_multi_agent_env.py    # Env multi-agent
│
├── 3_training/                # Script di addestramento
│   ├── smartlight_train_pisa_real.py    # Training su rete reale di Pisa
│   ├── smartlight_train_simulated.py    # Training su rete simulata
│   ├── smartlight_train_ppo.py          # Training con PPO
│   ├── smartlight_train_parallel.py     # Training parallelo DQN
│   └── smartlight_train_multi_agent.py  # Training multi-agent
│
├── 4_evaluation/              # Benchmark e valutazione
│   ├── scripts/               # Script di benchmark
│   └── results/               # Risultati (markdown e txt)
│
├── 5_deployment/              # Inferenza e demo
│   ├── smartlight_run_trained_model.py  # Esecuzione modello addestrato
│   ├── smartlight_live_video_dqn.py     # Demo video live
│   └── pitch_animation.html             # Animazione per presentazioni
│
├── checkpoints/               # Modelli addestrati (.pth)
│   ├── dqn/                   # DQN v1.8 → v3.1
│   ├── dueling_dqn/           # Dueling DQN v3.3 → v3.4
│   ├── ppo/                   # PPO v4.0
│   ├── multi_agent/           # Multi-agent (ep5 → ep200 + best)
│   └── edge_first/            # Edge-first architecture
│
├── map_analysis/              # Analisi rete stradale e semafori
│   ├── scripts/               # Script di mappatura (TraCI, OSM)
│   └── output/                # Report e mappe generate
│
├── data/                      # Dataset (immagini YOLO, labels)
├── docs/                      # Documentazione (Business Strategy, Roadmap, Shadow Mode)
├── sumo_tools/                # Installazione locale SUMO
├── smartlight_env/            # Virtual environment Python
│
├── requirements.txt           # Dipendenze Python
├── .gitignore
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Attiva il virtual environment
.\smartlight_env\Scripts\activate

# 2. Installa le dipendenze
pip install -r requirements.txt

# 3. Lancia un training DQN sulla rete di Pisa
python 3_training/smartlight_train_pisa_real.py

# 4. Esegui un benchmark
python 4_evaluation/scripts/benchmark_dqn_vs_ppo.py

# 5. Lancia il modello addestrato
python 5_deployment/smartlight_run_trained_model.py
```

## 🧠 Algoritmi Implementati

| Algoritmo | Versioni | Descrizione |
|-----------|----------|-------------|
| **DQN** | v1.8 → v3.1 | Deep Q-Network single-intersection |
| **Dueling DQN** | v3.3 → v3.4 | Architettura con value/advantage stream separati |
| **PPO** | v4.0 | Proximal Policy Optimization |
| **Multi-Agent** | ep5 → ep200 | Controllo distribuito su più intersezioni |
| **Edge-First** | best | Architettura ottimizzata per edge computing |

## 📄 Documentazione

- [Business Strategy](docs/Business_Strategy.pdf)
- [Roadmap](docs/Roadmap.pdf)
- [Shadow Mode Testing Plan](docs/Shadow_Mode_Testing_Plan.pdf)

## 🛠️ Tech Stack

- **Simulazione**: SUMO (Simulation of Urban Mobility)
- **RL Framework**: PyTorch
- **Interfaccia SUMO**: TraCI
- **Computer Vision**: YOLOv8 (opzionale)
- **Python**: 3.10+
