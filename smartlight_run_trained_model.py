"""
SmartLight Pisa - Run Trained Model
====================================
Script per caricare il modello addestrato e visualizzare la simulazione intelligente.
"""

import os
import sys
import torch
import numpy as np
from smartlight_multi_agent_env import MultiAgentSumoEnv
from smartlight_train_multi_agent import ActorCritic

# Configurazione percorsi
os.chdir(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = "smartlight_multi_agent_best.pth"

def run_trained_demo():
    print(f"Caricamento modello da: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    
    # Crea l'ambiente (GUI attiva)
    env = MultiAgentSumoEnv(
        use_gui=True,
        max_steps=3600,  # 1 ora di simulazione
        top_n_tls=5,
        min_controlled_lanes=3,
        alpha=0.7
    )
    
    obs_dict, _ = env.reset()
    
    # Carica la rete neurale
    model = ActorCritic(obs_dim=15, act_dim=2, hidden_dim=128)
    
    # Verifica se il modello è a parametri condivisi o indipendenti
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        shared = True
        print("Modello a parametri CONDIVISI rilevato.")
    else:
        shared = False
        print("Modello a parametri INDIPENDENTI rilevato.")
        # Creiamo un dizionario di modelli per ogni agente
        models = {}
        for agent_id in env.agent_ids:
            models[agent_id] = ActorCritic(obs_dim=15, act_dim=2, hidden_dim=128)
            safe_key = f"model_{agent_id.replace('#', '_').replace(' ', '_')[:50]}"
            if safe_key in checkpoint:
                models[agent_id].load_state_dict(checkpoint[safe_key])
            else:
                print(f"Attenzione: pesi non trovati per {agent_id}, uso inizializzazione casuale.")

    print("\nSimulazione INTELLIGENTE avviata...")
    print("Guarda la finestra di SUMO-GUI per vedere gli agenti in azione.")
    
    total_reward = 0
    step = 0
    
    try:
        while True:
            actions = {}
            for agent_id in env.agent_ids:
                obs = obs_dict[agent_id]
                if shared:
                    action, _, _ = model.act(obs)
                else:
                    action, _, _ = models[agent_id].act(obs)
                actions[agent_id] = action
            
            obs_dict, rewards, done, truncated, infos = env.step(actions)
            total_reward += np.mean(list(rewards.values()))
            step += 1
            
            if step % 100 == 0:
                metrics = env.get_global_metrics()
                print(f"Step {step} | Waiting: {metrics['total_waiting_time']:.1f}s | Speed: {metrics['avg_speed']:.2f}m/s")
            
            if done:
                break
                
    except KeyboardInterrupt:
        print("\nSimulazione interrotta dall'utente.")
    finally:
        env.close()
        print(f"\nFine simulazione. Reward medio finale: {total_reward/max(1,step):.4f}")

if __name__ == "__main__":
    run_trained_demo()
