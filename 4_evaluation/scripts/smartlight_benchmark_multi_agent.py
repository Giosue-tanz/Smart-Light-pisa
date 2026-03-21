import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import time
from torch.distributions import Categorical

from smartlight_multi_agent_env import MultiAgentSumoEnv

# ==============================
# RETE NEURALE ACTOR-CRITIC
# ==============================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=15, act_dim=2, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def deterministic_act(self, obs):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            probs, _ = self.forward(obs_tensor)
            action = torch.argmax(probs, dim=1).item()
        return action

def run_benchmark(steps=3600, agents=10, min_lanes=8, use_gui=False, model_path=None):
    print("=" * 60)
    print("🚦 SmartLight Pisa - Benchmark Multi-Agent: Fixed vs IPPO")
    print("=" * 60)
    
    # 1. FIXED TIMER (Baseline di default di SUMO)
    print("\n--- ESECUZIONE FIXED TIMER ---")
    env_fixed = MultiAgentSumoEnv(
        use_gui=use_gui,
        max_steps=steps,
        top_n_tls=agents,
        min_controlled_lanes=min_lanes,
    )
    env_fixed.reset()
    start_time = time.time()
    
    for _ in range(steps):
        # Noi passiamo azioni 0 a tutti per far continuare il ciclo normale del semaforo
        actions = {agent_id: 0 for agent_id in env_fixed.agent_ids}
        _, _, done, _, _ = env_fixed.step(actions)
        if done:
            break
            
    fixed_metrics = env_fixed.get_global_metrics()
    print(f"Completato in {time.time() - start_time:.1f}s")
    env_fixed.close()
    
    # 2. SMART AI (IPPO)
    print("\n--- ESECUZIONE SMART AI (IPPO) ---")
    if not os.path.exists(model_path):
        print(f"Errore: modello {model_path} non trovato.")
        return
        
    try:
        data = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        data = torch.load(model_path, map_location='cpu')

    env_ai = MultiAgentSumoEnv(
        use_gui=use_gui,
        max_steps=steps,
        top_n_tls=agents,
        min_controlled_lanes=min_lanes,
    )
    obs_dict, _ = env_ai.reset()
    
    # Load models
    models = {}
    is_shared = 'model_state' in data
    
    # create independent or shared models
    if is_shared:
        shared_model = ActorCritic(obs_dim=env_ai.obs_dim, act_dim=env_ai.act_dim)
        shared_model.load_state_dict(data['model_state'])
        shared_model.eval()
        for agent_id in env_ai.agent_ids:
            models[agent_id] = shared_model
    else:
        for agent_id in env_ai.agent_ids:
            model = ActorCritic(obs_dim=env_ai.obs_dim, act_dim=env_ai.act_dim)
            safe_key = agent_id.replace('#', '_').replace(' ', '_')[:50]
            if f'model_{safe_key}' in data:
                model.load_state_dict(data[f'model_{safe_key}'])
            model.eval()
            models[agent_id] = model
            
    start_time = time.time()
    for _ in range(steps):
        actions = {}
        for agent_id in env_ai.agent_ids:
            actions[agent_id] = models[agent_id].deterministic_act(obs_dict[agent_id])
            
        obs_dict, _, done, _, _ = env_ai.step(actions)
        if done:
            break
            
    ai_metrics = env_ai.get_global_metrics()
    print(f"Completato in {time.time() - start_time:.1f}s")
    env_ai.close()
    
    # =================
    # RISULTATI
    # =================
    print("\n" + "="*60)
    print("📊 RISULTATI BENCHMARK (Multi-Agent, {} semafori)".format(env_ai.n_agents))
    print("="*60)
    print(f"{'METRICA':<25} | {'FIXED TIMER':<15} | {'SMART AI':<12} | {'MIGLIORAMENTO':<15}")
    print("-" * 75)
    
    wait_fix = fixed_metrics['total_waiting_time']
    wait_ai = ai_metrics['total_waiting_time']
    wait_diff = wait_ai - wait_fix
    wait_pct = (wait_diff / max(1, wait_fix)) * 100
    
    halt_fix = fixed_metrics['total_halting']
    halt_ai = ai_metrics['total_halting']
    halt_diff = halt_ai - halt_fix
    halt_pct = (halt_diff / max(1, halt_fix)) * 100
    
    spd_fix = fixed_metrics['avg_speed']
    spd_ai = ai_metrics['avg_speed']
    spd_diff = spd_ai - spd_fix
    
    print(f"{'Tempo Attesa (s)':<25} | {wait_fix:<15.1f} | {wait_ai:<12.1f} | {wait_diff:+.1f}s ({wait_pct:+.1f}%)")
    print(f"{'Veicoli in Coda':<25} | {halt_fix:<15} | {halt_ai:<12} | {halt_diff:+} ({halt_pct:+.1f}%)")
    print(f"{'Velocità Media (m/s)':<25} | {spd_fix:<15.2f} | {spd_ai:<12.2f} | {spd_diff:+.2f}")
    
    with open("benchmark_results_multi_agent.md", "w", encoding="utf-8") as f:
        f.write("# Benchmark Multi-Agent: SmartLight IPPO vs Fixed Timer\n\n")
        f.write(f"Testato su **{env_ai.n_agents} incroci** in parallelo.\n\n")
        f.write("| Metrica | Fixed Timer | Smart AI (IPPO) | Miglioramento |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Tempo Diffuso Attesa (s)** | {wait_fix:.1f} | {wait_ai:.1f} | **{wait_pct:+.1f}%** |\n")
        f.write(f"| **Veicoli Fermi (Halting)** | {halt_fix} | {halt_ai} | **{halt_pct:+.1f}%** |\n")
        f.write(f"| **Velocità Media (m/s)** | {spd_fix:.2f} | {spd_ai:.2f} | **{spd_diff:+.2f} m/s** |\n")
    print("\nReport salvato in 'benchmark_results_multi_agent.md'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--min-lanes", type=int, default=8)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--model", type=str, default="smartlight_multi_agent_best.pth")
    args = parser.parse_args()
    
    run_benchmark(steps=args.steps, agents=args.agents, min_lanes=args.min_lanes, use_gui=args.gui, model_path=args.model)
