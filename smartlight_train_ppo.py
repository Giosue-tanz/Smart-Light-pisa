import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import time

# ==============================
# CONFIGURAZIONE GPU
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurazione
MIN_GREEN, MAX_GREEN = 10, 40
ACTION_SIZE = MAX_GREEN - MIN_GREEN + 1  # 31 azioni

# ==============================
# TIPI VEICOLI (COMPRESSI)
# ==============================
VEHICLE_WEIGHTS = {
    'micro': 1,
    'private': 1.5,
    'public': 35,
    'freight': 2,
    'emergency': 100,
}

def calculate_reward(queues, waits, has_disabled, has_emergency):
    micro_q, private_q, public_q, freight_q, emergency_q, ped_q = queues
    ped_wait, veh_wait = waits
    
    wait_penalty = (ped_wait ** 2 + veh_wait ** 2) * 0.3
    
    queue_penalty = (
        ped_q * 5 +
        micro_q * VEHICLE_WEIGHTS['micro'] * 3 +
        private_q * VEHICLE_WEIGHTS['private'] * 8 +
        public_q * VEHICLE_WEIGHTS['public'] * 12 +
        freight_q * VEHICLE_WEIGHTS['freight'] * 10
    )
    
    emergency_penalty = emergency_q * 15000
    disability_bonus = 2000 if has_disabled else 0
    
    total = -wait_penalty - queue_penalty - emergency_penalty + disability_bonus
    return total / 1000.0

# ==============================
# AMBIENTE CON STATE COMPRESSION
# ==============================
class TrafficSimEnv:
    def __init__(self):
        self.scenarios = [
            {'name': 'mattina_presto', 'hour': 7, 'mult': 0.7},
            {'name': 'rush_mattina', 'hour': 8.5, 'mult': 2.0},
            {'name': 'meta_giornata', 'hour': 11, 'mult': 1.0},
            {'name': 'pranzo', 'hour': 13, 'mult': 1.3},
            {'name': 'pomeriggio', 'hour': 15, 'mult': 1.0},
            {'name': 'rush_sera', 'hour': 18, 'mult': 2.2},
            {'name': 'sera', 'hour': 21, 'mult': 0.6},
            {'name': 'notte', 'hour': 1, 'mult': 0.2},
        ]
        self.reset()
        
    def reset(self):
        self.scenario = random.choice(self.scenarios)
        self.hour = self.scenario['hour'] + random.uniform(-0.5, 0.5)
        
        self.micro_q = random.randint(0, 8)
        self.private_q = random.randint(0, 15)
        self.public_q = random.randint(0, 3)
        self.freight_q = random.randint(0, 2)
        self.emergency_q = 0
        
        self.ped_q = random.randint(0, 10)
        self.dis_ped = 0
        
        self.ped_wait = 0
        self.veh_wait = 0
        self.prev_green = 20
        
        return self._get_state()

    def _get_state(self):
        return np.array([
            min(self.micro_q / 15, 1),
            min(self.private_q / 25, 1),
            min(self.public_q / 6, 1),
            min(self.freight_q / 5, 1),
            min(self.emergency_q / 2, 1),
            min(self.ped_q / 20, 1),
            self.dis_ped,
            min(self.ped_wait / 100, 1),
            min(self.veh_wait / 100, 1),
            (self.prev_green - MIN_GREEN) / (MAX_GREEN - MIN_GREEN),
            self.hour / 24,
        ], dtype=np.float32)

    def step(self, action_idx):
        green_time = MIN_GREEN + action_idx
        
        queues = (self.micro_q, self.private_q, self.public_q, 
                  self.freight_q, self.emergency_q, self.ped_q)
        waits = (self.ped_wait, self.veh_wait)
        reward = calculate_reward(queues, waits, self.dis_ped > 0, self.emergency_q > 0)
        
        ped_passed = min(self.ped_q, green_time // 2)
        self.ped_q -= ped_passed
        if self.ped_q == 0: self.ped_wait = 0
        if self.dis_ped > 0 and ped_passed > 0: self.dis_ped = 0
        
        capacity = green_time // 2
        
        if self.emergency_q > 0:
            em_pass = min(self.emergency_q, 2)
            self.emergency_q -= em_pass
            capacity -= em_pass
        
        if capacity > 0 and self.public_q > 0:
            pub_pass = min(self.public_q, capacity // 3)
            self.public_q -= pub_pass
            capacity -= pub_pass * 3
        
        if capacity > 0:
            micro_pass = min(self.micro_q, capacity * 2)
            self.micro_q -= micro_pass
        
        if capacity > 0:
            priv_pass = min(self.private_q, capacity)
            self.private_q -= priv_pass
            capacity -= priv_pass
        
        if capacity > 0 and self.freight_q > 0:
            freight_pass = min(self.freight_q, capacity // 3)
            self.freight_q -= freight_pass
        
        if self.micro_q + self.private_q + self.public_q + self.freight_q + self.emergency_q == 0:
            self.veh_wait = 0

        mult = self.scenario['mult']
        
        self.micro_q += np.random.poisson(2.0 * mult)
        self.private_q += np.random.poisson(3.0 * mult)
        
        if random.random() < 0.12 * mult:
            self.public_q += 1
        if random.random() < 0.05:
            self.freight_q += 1
        if random.random() < 0.008:
            self.emergency_q += 1
            
        self.ped_q += np.random.poisson(2.5 * mult)
        
        if random.random() < 0.03:
            self.dis_ped = 1

        if self.ped_q > 0: self.ped_wait += green_time
        if self.private_q + self.public_q > 0: self.veh_wait += green_time
        
        self.prev_green = green_time
        self.hour = (self.hour + green_time / 3600) % 24
        
        if random.random() < 0.02:
            self.scenario = random.choice(self.scenarios)
        
        return self._get_state(), reward, False


# ==============================
# PPO ACTOR-CRITIC NETWORK
# ==============================
class ActorCritic(nn.Module):
    """
    PPO Actor-Critic Architecture
    
    Actor: produce una distribuzione di probabilità sulle azioni
    Critic: stima il valore dello stato V(s)
    """
    def __init__(self, state_size=11, action_size=ACTION_SIZE):
        super().__init__()
        
        # ========== SHARED BACKBONE ==========
        self.shared = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        
        # ========== ACTOR HEAD ==========
        # Output: logits per ogni azione (poi softmax per probabilità)
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
        
        # ========== CRITIC HEAD ==========
        # Output: valore scalare V(s)
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        # Inizializzazione ortogonale (standard per PPO)
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        shared_features = self.shared(state)
        return shared_features
    
    def get_action_and_value(self, state, action=None):
        """
        Ritorna:
        - action: azione campionata (se action=None) o quella passata
        - log_prob: log probabilità dell'azione
        - entropy: entropia della distribuzione (per esplorazione)
        - value: V(s) stimato dal critic
        """
        shared_features = self.shared(state)
        
        # Actor: calcola logits e distribuzione
        logits = self.actor(shared_features)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        # Critic: stima valore
        value = self.critic(shared_features)
        
        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)
    
    def get_value(self, state):
        """Solo per calcolare V(s) senza campionare azioni"""
        shared_features = self.shared(state)
        return self.critic(shared_features).squeeze(-1)


# ==============================
# PPO ROLLOUT BUFFER
# ==============================
class RolloutBuffer:
    """Buffer per raccogliere esperienze durante il rollout"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def push(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Calcola returns e advantages usando GAE (Generalized Advantage Estimation)
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [False])
        
        # GAE calculation
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + np.array(self.values)
        
        return returns, advantages
    
    def get_batches(self, batch_size):
        """Genera mini-batch randomizzati"""
        n = len(self.states)
        indices = np.random.permutation(n)
        
        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield batch_indices


# ==============================
# PPO TRAINING
# ==============================
def train_ppo():
    print("=" * 70)
    print("SMARTLIGHT PISA v4.0 - PPO (Proximal Policy Optimization)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("-" * 70)
    print("[OK] Actor-Critic Architecture")
    print("[OK] GAE (Generalized Advantage Estimation)")
    print("[OK] PPO Clipping (ε=0.2)")
    print("[OK] Entropy Bonus for Exploration")
    print("[OK] LayerNorm (più stabile di BatchNorm per PPO)")
    print("=" * 70)
    
    # Hyperparameters PPO
    TOTAL_TIMESTEPS = 150_000      # Timesteps totali di training
    ROLLOUT_LENGTH = 2048          # Steps per rollout prima di update
    BATCH_SIZE = 256               # Mini-batch size
    N_EPOCHS = 10                  # Epoche di training per ogni rollout
    GAMMA = 0.99                   # Discount factor
    GAE_LAMBDA = 0.95              # GAE lambda
    CLIP_EPSILON = 0.2             # PPO clipping
    LR = 3e-4                      # Learning rate
    ENTROPY_COEF = 0.01            # Coefficiente entropia (esplorazione)
    VALUE_COEF = 0.5               # Coefficiente loss del critic
    MAX_GRAD_NORM = 0.5            # Gradient clipping
    
    env = TrafficSimEnv()
    model = ActorCritic(state_size=11, action_size=ACTION_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    buffer = RolloutBuffer()
    
    print(f"\nHyperparameters:")
    print(f"  Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Rollout Length: {ROLLOUT_LENGTH}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  PPO Epochs: {N_EPOCHS}")
    print(f"  Learning Rate: {LR}")
    print(f"  Clip Epsilon: {CLIP_EPSILON}")
    print()
    
    # Tracking
    best_avg_reward = -float('inf')
    episode_rewards = []
    current_ep_reward = 0
    
    state = env.reset()
    start_time = time.time()
    
    num_updates = TOTAL_TIMESTEPS // ROLLOUT_LENGTH
    timestep = 0
    
    for update in range(num_updates):
        # ========== ROLLOUT PHASE ==========
        model.eval()
        buffer.clear()
        
        for _ in range(ROLLOUT_LENGTH):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                action, log_prob, _, value = model.get_action_and_value(state_tensor)
                
            action_idx = action.item()
            next_state, reward, done = env.step(action_idx)
            
            buffer.push(
                state=state,
                action=action_idx,
                reward=reward,
                value=value.item(),
                log_prob=log_prob.item(),
                done=done
            )
            
            current_ep_reward += reward
            state = next_state
            timestep += 1
            
            # Fine episodio (120 steps come nel DQN)
            if timestep % 120 == 0:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0
                state = env.reset()
        
        # Calcola last_value per GAE
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            last_value = model.get_value(state_tensor).item()
        
        # Calcola returns e advantages
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, gamma=GAMMA, gae_lambda=GAE_LAMBDA
        )
        
        # Normalizza advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Converti in tensori
        states_t = torch.FloatTensor(np.array(buffer.states)).to(DEVICE)
        actions_t = torch.LongTensor(buffer.actions).to(DEVICE)
        old_log_probs_t = torch.FloatTensor(buffer.log_probs).to(DEVICE)
        returns_t = torch.FloatTensor(returns).to(DEVICE)
        advantages_t = torch.FloatTensor(advantages).to(DEVICE)
        
        # ========== PPO UPDATE PHASE ==========
        model.train()
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_batches = 0
        
        for epoch in range(N_EPOCHS):
            for batch_indices in buffer.get_batches(BATCH_SIZE):
                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, values = model.get_action_and_value(
                    batch_states, batch_actions
                )
                
                # Policy loss con PPO clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus (incoraggia esplorazione)
                entropy_loss = entropy.mean()
                
                # Loss totale
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                n_batches += 1
        
        # ========== LOGGING ==========
        if len(episode_rewards) > 0:
            recent_rewards = episode_rewards[-20:]
            avg_reward = np.mean(recent_rewards)
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(model.state_dict(), 'smartlight_ppo_pisa_v4.0_best.pth')
            
            elapsed = time.time() - start_time
            fps = timestep / elapsed
            eta = (TOTAL_TIMESTEPS - timestep) / fps if fps > 0 else 0
            
            if (update + 1) % 5 == 0:
                print(f"Update {update+1:3d}/{num_updates} | "
                      f"Steps: {timestep:6,} | "
                      f"Avg Reward: {avg_reward:8.2f} | "
                      f"Best: {best_avg_reward:8.2f} | "
                      f"FPS: {fps:.0f} | "
                      f"ETA: {eta:.0f}s")
    
    # ========== SAVE FINAL MODEL ==========
    total_time = time.time() - start_time
    torch.save(model.state_dict(), 'smartlight_ppo_pisa_v4.0.pth')
    
    print("\n" + "=" * 70)
    print("TRAINING PPO v4.0 COMPLETATO!")
    print(f"  Tempo totale: {total_time:.1f} secondi ({total_time/60:.1f} minuti)")
    print(f"  Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Episodi completati: {len(episode_rewards)}")
    print(f"  FPS medio: {TOTAL_TIMESTEPS/total_time:.0f}")
    print(f"  Modello finale: smartlight_ppo_pisa_v4.0.pth")
    print(f"  Modello migliore: smartlight_ppo_pisa_v4.0_best.pth")
    print(f"  Best Avg Reward (20 ep): {best_avg_reward:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    train_ppo()
