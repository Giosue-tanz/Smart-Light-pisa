"""
SmartLight Pisa - Multi-Agent Training con IPPO (Independent PPO)
=================================================================
Addestra N agenti PPO indipendenti che controllano ciascuno un semaforo
nella mappa reale di Pisa.

Ogni agente ha:
- La sua rete neurale (Actor-Critic)
- Osservazione locale (11 dim) + info vicini (4 dim) = 15 dim
- Reward misto: α * locale + (1-α) * globale

Utilizzo:
    python smartlight_train_multi_agent.py
    python smartlight_train_multi_agent.py --gui        # con visualizzazione
    python smartlight_train_multi_agent.py --agents 10  # 10 agenti
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import defaultdict

# Aggiungi il percorso SUMO 
SUMO_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_tools", "sumo-1.22.0")
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME
sumo_bin_dir = os.path.join(SUMO_HOME, 'bin')
if sumo_bin_dir not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + sumo_bin_dir

from smartlight_multi_agent_env import MultiAgentSumoEnv


# ==============================
# RETE NEURALE ACTOR-CRITIC
# ==============================
class ActorCritic(nn.Module):
    """
    Rete Actor-Critic condivisa per PPO.
    Usata da ogni agente indipendentemente (IPPO).
    """
    def __init__(self, obs_dim=15, act_dim=2, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Feature extractor condiviso
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
            nn.Softmax(dim=-1),
        )
        
        # Critic (value function)
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
    
    def act(self, obs):
        """Seleziona azione con sampling dalla policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            probs, value = self.forward(obs_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate(self, obs, actions):
        """Valuta azioni date per il calcolo del loss PPO."""
        probs, values = self.forward(obs)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


# ==============================
# BUFFER PER PPO
# ==============================
class RolloutBuffer:
    """Buffer per raccogliere le traiettorie di un singolo agente."""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, obs, action, log_prob, reward, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns(self, gamma=0.99, lam=0.95):
        """Calcola i returns con GAE (Generalized Advantage Estimation)."""
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        return returns, advantages
    
    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


# ==============================
# PPO UPDATE
# ==============================
def ppo_update(model, optimizer, buffer, epochs=4, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """Esegue l'aggiornamento PPO per un singolo agente."""
    returns, advantages = buffer.compute_returns()
    
    # Converti in tensori
    obs = torch.FloatTensor(np.array(buffer.observations))
    actions = torch.LongTensor(buffer.actions)
    old_log_probs = torch.FloatTensor(buffer.log_probs)
    returns_t = torch.FloatTensor(returns)
    advantages_t = torch.FloatTensor(advantages)
    
    # Normalizza i vantaggi
    if len(advantages_t) > 1:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
    
    total_loss = 0
    for _ in range(epochs):
        log_probs, values, entropy = model.evaluate(obs, actions)
        
        # Policy loss (clipped)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_t
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values, returns_t)
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Loss totale
        loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / epochs


# ==============================
# TRAINING LOOP PRINCIPALE
# ==============================
def train(args):
    print("=" * 60)
    print("🚦 SmartLight Pisa - Multi-Agent IPPO Training")
    print("=" * 60)
    print(f"  Agenti: top {args.agents}")
    print(f"  Episodi: {args.episodes}")
    print(f"  Steps/episodio: {args.steps}")
    print(f"  Alpha (locale/globale): {args.alpha}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Hidden Dim: {args.hidden}")
    print(f"  GUI: {args.gui}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    # Crea ambiente
    env = MultiAgentSumoEnv(
        use_gui=args.gui,
        max_steps=args.steps,
        top_n_tls=args.agents,
        min_controlled_lanes=args.min_lanes,
        min_green=5,
        alpha=args.alpha,
        neighbor_radius=args.radius,
    )
    
    # Primo reset per scoprire gli agenti
    obs_dict, _ = env.reset()
    
    # Crea un modello PPO per ogni agente (IPPO)
    models = {}
    optimizers = {}
    buffers = {}
    
    # Se shared_params=True, tutti gli agenti condividono gli stessi pesi
    if args.shared_params:
        shared_model = ActorCritic(obs_dim=env.obs_dim, act_dim=env.act_dim, hidden_dim=args.hidden)
        shared_optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
        for agent_id in env.agent_ids:
            models[agent_id] = shared_model
            optimizers[agent_id] = shared_optimizer
            buffers[agent_id] = RolloutBuffer()
        print(f"\n🔗 Parametri CONDIVISI tra tutti gli agenti")
    else:
        for agent_id in env.agent_ids:
            models[agent_id] = ActorCritic(obs_dim=env.obs_dim, act_dim=env.act_dim, hidden_dim=args.hidden)
            optimizers[agent_id] = optim.Adam(models[agent_id].parameters(), lr=args.lr)
            buffers[agent_id] = RolloutBuffer()
        print(f"\n🔓 Parametri INDIPENDENTI per ogni agente")
    
    env.close()
    
    # Metriche di training
    episode_rewards_history = []
    best_avg_reward = float('-inf')
    
    for episode in range(args.episodes):
        ep_start = time.time()
        
        obs_dict, _ = env.reset()
        
        episode_rewards = defaultdict(float)
        episode_metrics = []
        
        for step in range(args.steps):
            actions = {}
            
            # Ogni agente sceglie la sua azione
            for agent_id in env.agent_ids:
                obs = obs_dict[agent_id]
                action, log_prob, value = models[agent_id].act(obs)
                actions[agent_id] = action
                
                # Salva nel buffer (il reward verrà aggiunto dopo)
                buffers[agent_id].add(
                    obs=obs,
                    action=action,
                    log_prob=log_prob,
                    reward=0,  # placeholder
                    value=value,
                    done=False,
                )
            
            # Step dell'ambiente
            obs_dict, rewards, done, truncated, infos = env.step(actions)
            
            # Aggiorna rewards nei buffer
            for agent_id in env.agent_ids:
                buffers[agent_id].rewards[-1] = rewards[agent_id]
                buffers[agent_id].dones[-1] = done
                episode_rewards[agent_id] += rewards[agent_id]
            
            # Raccogli metriche ogni 200 step
            if step % 200 == 0:
                metrics = env.get_global_metrics()
                episode_metrics.append(metrics)
            
            if done:
                break
        
        # PPO Update per ogni agente
        avg_loss = 0
        for agent_id in env.agent_ids:
            loss = ppo_update(models[agent_id], optimizers[agent_id], buffers[agent_id],
                            epochs=args.ppo_epochs, clip_eps=args.clip_eps)
            avg_loss += loss
            buffers[agent_id].clear()
        avg_loss /= max(1, len(env.agent_ids))
        
        # Metriche episodio
        mean_reward = np.mean(list(episode_rewards.values()))
        episode_rewards_history.append(mean_reward)
        ep_time = time.time() - ep_start
        
        # Log
        final_metrics = env.get_global_metrics()
        print(f"\n{'='*60}")
        print(f"📊 Episodio {episode+1}/{args.episodes} | Tempo: {ep_time:.1f}s")
        print(f"   Mean Agent Reward: {mean_reward:.4f}")
        print(f"   Avg Loss: {avg_loss:.4f}")
        print(f"   Total Waiting Time: {final_metrics['total_waiting_time']:.1f}s")
        print(f"   Total Halting: {final_metrics['total_halting']}")
        print(f"   Avg Speed: {final_metrics['avg_speed']:.2f} m/s")
        
        # Salva il modello migliore
        if mean_reward > best_avg_reward:
            best_avg_reward = mean_reward
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     f"smartlight_multi_agent_best.pth")
            save_dict = {
                'episode': episode,
                'best_reward': best_avg_reward,
                'n_agents': env.n_agents,
                'agent_ids': env.agent_ids,
            }
            if args.shared_params:
                save_dict['model_state'] = shared_model.state_dict()
            else:
                for agent_id in env.agent_ids:
                    safe_key = agent_id.replace('#', '_').replace(' ', '_')[:50]
                    save_dict[f'model_{safe_key}'] = models[agent_id].state_dict()
            
            torch.save(save_dict, save_path)
            print(f"   💾 Nuovo migliore! Salvato in {save_path}")
        
        # Salva checkpoint periodico
        if (episode + 1) % args.save_every == 0:
            ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      f"smartlight_multi_agent_ep{episode+1}.pth")
            save_dict = {
                'episode': episode,
                'reward': mean_reward,
                'n_agents': env.n_agents,
            }
            if args.shared_params:
                save_dict['model_state'] = shared_model.state_dict()
            else:
                for agent_id in env.agent_ids:
                    safe_key = agent_id.replace('#', '_').replace(' ', '_')[:50]
                    save_dict[f'model_{safe_key}'] = models[agent_id].state_dict()
            
            torch.save(save_dict, ckpt_path)
            print(f"   💾 Checkpoint salvato: {ckpt_path}")
    
    env.close()
    
    # Riepilogo finale
    print(f"\n{'='*60}")
    print(f"🏁 TRAINING COMPLETATO")
    print(f"{'='*60}")
    print(f"  Episodi: {args.episodes}")
    print(f"  Miglior Reward Medio: {best_avg_reward:.4f}")
    if episode_rewards_history:
        last_10 = episode_rewards_history[-10:]
        print(f"  Media ultimi 10 episodi: {np.mean(last_10):.4f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="SmartLight Multi-Agent Training (IPPO)")
    
    # Ambiente
    parser.add_argument("--gui", action="store_true", help="Usa SUMO-GUI")
    parser.add_argument("--agents", type=int, default=5, help="Numero di agenti (top N semafori)")
    parser.add_argument("--steps", type=int, default=1800, help="Steps per episodio")
    parser.add_argument("--min-lanes", type=int, default=8, help="Min corsie per agente")
    parser.add_argument("--alpha", type=float, default=0.7, help="Peso reward locale (0-1)")
    parser.add_argument("--radius", type=float, default=300.0, help="Raggio vicini in metri")
    
    # Training
    parser.add_argument("--episodes", type=int, default=100, help="Numero di episodi")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dim della rete")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="Epoche PPO per update")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--shared-params", action="store_true", help="Condividi parametri tra agenti")
    parser.add_argument("--save-every", type=int, default=10, help="Salva checkpoint ogni N episodi")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
