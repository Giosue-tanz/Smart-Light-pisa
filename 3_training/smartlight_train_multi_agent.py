"""
SmartLight Pisa - Multi-Agent Training (EDGE-FIRST / PARAMETER SHARING)
=======================================================================
Addestra una singola Policy (Actor-Critic) condivisa tra tutti gli incroci ("Parameter Sharing").
Durante l'addestramento il "cervello" accumula le esperienze di tutti i semafori, 
imparando una regola universale che poi ogni semaforo applicherà in locale e in autonomia.

Utilizzo:
    python smartlight_train_multi_agent.py
    python smartlight_train_multi_agent.py --gui
"""

import os
import sys
import time
import argparse
import numpy as np
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import defaultdict

# Aggiungi il percorso SUMO
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_HOME = os.path.join(PROJECT_ROOT, "sumo_tools", "sumo-1.22.0")
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME
sumo_bin_dir = os.path.join(SUMO_HOME, 'bin')
if sumo_bin_dir not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + sumo_bin_dir

# Importa env dalla cartella 2_environment
sys.path.insert(0, os.path.join(PROJECT_ROOT, '2_environment'))
from smartlight_multi_agent_env import MultiAgentSumoEnv

# ==============================
# RETE NEURALE CONDIVISA
# ==============================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=11, act_dim=2, hidden_dim=128):
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
        return self.actor(features), self.critic(features)
    
    def act(self, obs):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            probs, value = self.forward(obs_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate(self, obs, actions):
        probs, values = self.forward(obs)
        dist = Categorical(probs)
        return dist.log_prob(actions), values.squeeze(-1), dist.entropy()


# ==============================
# BUFFER SHARING CONDIVISO
# ==============================
class SharedRolloutBuffer:
    """Raccoglie traiettorie di TUTTI gli agenti per addestrare l'unica rete."""
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
        advantages, returns, gae = [], [], 0
        for t in reversed(range(len(self.rewards))):
            next_value = 0 if t == len(self.rewards) - 1 else self.values[t + 1]
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


def ppo_update(model, optimizer, buffer, epochs=4, batch_size=64, clip_eps=0.2):
    returns, advantages = buffer.compute_returns()
    obs = torch.FloatTensor(np.array(buffer.observations))
    actions = torch.LongTensor(buffer.actions)
    old_log_probs = torch.FloatTensor(buffer.log_probs)
    returns_t = torch.FloatTensor(returns)
    advantages_t = torch.FloatTensor(advantages)
    
    if len(advantages_t) > 1:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
    
    dataset_size = len(obs)
    actual_batch_size = min(batch_size, dataset_size)
    if actual_batch_size == 0: return 0.0

    total_loss, num_updates = 0, 0
    for _ in range(epochs):
        indices = np.random.permutation(dataset_size)
        for start_idx in range(0, dataset_size, actual_batch_size):
            end_idx = min(start_idx + actual_batch_size, dataset_size)
            idx = indices[start_idx:end_idx]
            
            log_probs, values, entropy = model.evaluate(obs[idx], actions[idx])
            ratio = torch.exp(log_probs - old_log_probs[idx])
            
            surr1 = ratio * advantages_t[idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_t[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns_t[idx])
            
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            num_updates += 1
            
    return total_loss / max(1, num_updates)


def train(args):
    print("=" * 60)
    print("🚦 SmartLight Pisa - Edge-First Training (Parameter Sharing)")
    print("=" * 60)
    
    env = MultiAgentSumoEnv(
        use_gui=args.gui,
        max_steps=args.steps,
        top_n_tls=args.agents,
        min_controlled_lanes=args.min_lanes,
        min_green=10,
        max_green=60
    )
    
    # PARAMETER SHARING: Un unico modello per tutti
    shared_model = ActorCritic(obs_dim=env.obs_dim, act_dim=env.act_dim, hidden_dim=args.hidden)
    optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    
    # Usiamo bufffer separati per il tracking PPO corretto temporalmente per agente
    buffers = {}
    
    best_avg_reward = float('-inf')
    obs_dict, _ = env.reset()
    
    for agent_id in env.agent_ids:
        buffers[agent_id] = SharedRolloutBuffer()
        
    for episode in range(args.episodes):
        ep_start = time.time()
        
        # Generazione Dinamica dello Scenario Random
        seed = 42 + episode
        print(f"   [Generazione Scenario On-the-Fly con seed: {seed}]")
        gen_script = os.path.join(PROJECT_ROOT, "1_simulation", "genera_scenario_medio.py")
        subprocess.run([sys.executable, gen_script, "--seed", str(seed), "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        obs_dict, _ = env.reset()
        episode_rewards = defaultdict(float)
        
        for step in range(args.steps):
            actions = {}
            # Inference decentralizzata con la stessa rete
            for agent_id in env.agent_ids:
                obs = obs_dict[agent_id]
                action, log_prob, value = shared_model.act(obs)
                actions[agent_id] = action
                buffers[agent_id].add(obs, action, log_prob, 0, value, False)
            
            obs_dict, rewards, done, _, _ = env.step(actions)
            
            # Applicazione Reward 
            for agent_id in env.agent_ids:
                buffers[agent_id].rewards[-1] = rewards[agent_id]
                buffers[agent_id].dones[-1] = done
                episode_rewards[agent_id] += rewards[agent_id]
            
            if done: break
            
        # PPO Update (con fonderia di tutti i buffer in uno meta-buffer se volessimo esaltare il Parameter Sharing)
        # Ma per GAE è essenziale mantenere l'ordine temporale di ogni agente, quindi aggiorniamo sequenzialmente
        avg_loss = 0
        for agent_id in env.agent_ids:
            loss = ppo_update(shared_model, optimizer, buffers[agent_id], epochs=args.ppo_epochs, batch_size=args.batch_size)
            avg_loss += loss
            buffers[agent_id].clear()
        avg_loss /= env.n_agents
        
        mean_reward = np.mean(list(episode_rewards.values()))
        
        # Log 
        fm = env.get_global_metrics()
        print(f"\n📊 Episodio {episode+1}/{args.episodes} | Time: {time.time()-ep_start:.1f}s")
        print(f"   Reward Netto (Max Pressure Penalty): {mean_reward:.4f} | Loss: {avg_loss:.4f}")
        print(f"   Wait: {fm['total_waiting_time']:.1f}s | Halting: {fm['total_halting']} | Spd: {fm['avg_speed']:.2f} m/s")
        
        # Salvataggio
        if mean_reward > best_avg_reward:
            best_avg_reward = mean_reward
            save_path = "smartlight_edgefirst_best.pth"
            torch.save({
                'episode': episode,
                'model_state': shared_model.state_dict(),
                'obs_dim': env.obs_dim
            }, save_path)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--min-lanes", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    train(args)
