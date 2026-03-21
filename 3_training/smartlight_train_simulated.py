import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from collections import deque

# ==============================
# CONFIGURAZIONE GPU
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurazione
MIN_GREEN, MAX_GREEN = 10, 40
ACTION_SIZE = MAX_GREEN - MIN_GREEN + 1

# ==============================
# PRIORITIZED REPLAY BUFFER
# ==============================
class PrioritizedReplayBuffer:
    """Experience Replay con priorita basata su TD-error"""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(self.max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = self.max_priority
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None, None
            
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[i] for i in indices]
        
        return batch, indices, torch.FloatTensor(weights).to(DEVICE)
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-5
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return len(self.buffer)

# ==============================
# DUELING DQN CON BATCH NORMALIZATION (GPU)
# ==============================
class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture (Fase 2 - Priorita 5)
    
    Separa la funzione Q in:
    - Value Stream V(s): valore dello stato
    - Advantage Stream A(s,a): vantaggio di ogni azione
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    """
    def __init__(self, input_size=11):
        super().__init__()
        
        # ========== SHARED FEATURE LAYERS ==========
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        # ========== VALUE STREAM ==========
        # Stima quanto e' buono essere in uno stato (indipendente dall'azione)
        self.value_fc = nn.Linear(256, 128)
        self.value_bn = nn.BatchNorm1d(128)
        self.value_out = nn.Linear(128, 1)  # Output singolo: V(s)
        
        # ========== ADVANTAGE STREAM ==========
        # Stima il vantaggio relativo di ogni azione
        self.advantage_fc = nn.Linear(256, 128)
        self.advantage_bn = nn.BatchNorm1d(128)
        self.advantage_out = nn.Linear(128, ACTION_SIZE)  # Output: A(s,a) per ogni azione
        
        # Xavier initialization per tutti i layer
        for layer in [self.fc1, self.fc2, self.value_fc, self.value_out, 
                      self.advantage_fc, self.advantage_out]:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x):
        single_input = x.dim() == 1
        if single_input:
            x = x.unsqueeze(0)
        
        # Shared feature extraction
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        
        # Value stream: V(s)
        value = torch.relu(self.value_bn(self.value_fc(x)))
        value = self.value_out(value)  # Shape: (batch, 1)
        
        # Advantage stream: A(s,a)
        advantage = torch.relu(self.advantage_bn(self.advantage_fc(x)))
        advantage = self.advantage_out(advantage)  # Shape: (batch, ACTION_SIZE)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A))
        # Sottraggo la media per stabilita' (centra i vantaggi a zero)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        if single_input:
            q_values = q_values.squeeze(0)
        return q_values


# Alias per retrocompatibilita'
DQN = DuelingDQN

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
# TRAINING CON GPU + FASE 1 + FASE 3
# ==============================
def train():
    print("=" * 70)
    print("SMARTLIGHT PISA v3.3 - DUELING DQN")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("-" * 70)
    print("[OK] Prioritized Experience Replay (Fase 1)")
    print("[OK] Learning Rate Scheduler (Fase 1)")
    print("[OK] State Compression 17->11 (Fase 1)")
    print("[OK] Batch Normalization (Fase 1)")
    print("[OK] GPU Acceleration (Fase 3)")
    print("[OK] Dueling DQN Architecture (Fase 2)")
    print("=" * 70)
    
    env = TrafficSimEnv()
    
    # Reti su GPU
    policy_net = DQN(input_size=11).to(DEVICE)
    target_net = DQN(input_size=11).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.7)
    
    memory = PrioritizedReplayBuffer(capacity=100000, alpha=0.6, beta=0.4)
    
    episodes = 300
    batch_size = 256  # Batch piu grande con GPU
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.995
    
    best_reward = -float('inf')
    
    print(f"\nTraining: {episodes} episodi (batch={batch_size})\n")
    
    import time
    start_time = time.time()
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        policy_net.train()
        
        for _ in range(120):
            if random.random() <= epsilon:
                action = random.randrange(ACTION_SIZE)
            else:
                policy_net.eval()
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(DEVICE)
                    q_values = policy_net(state_tensor)
                    action = torch.argmax(q_values).item()
                policy_net.train()
            
            next_state, reward, done = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(memory) > batch_size:
                batch, indices, weights = memory.sample(batch_size)
                
                # Tensori su GPU
                states = torch.FloatTensor(np.array([x[0] for x in batch])).to(DEVICE)
                actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(1).to(DEVICE)
                rewards = torch.FloatTensor([x[2] for x in batch]).to(DEVICE)
                next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(DEVICE)
                dones = torch.FloatTensor([x[4] for x in batch]).to(DEVICE)
                
                q_eval = policy_net(states).gather(1, actions).squeeze()
                
                with torch.no_grad():
                    target_net.eval()
                    q_next = target_net(next_states).max(1)[0]
                    q_target = rewards + gamma * q_next * (1 - dones)
                
                td_errors = (q_eval - q_target).detach().cpu().numpy()
                memory.update_priorities(indices, td_errors)
                
                loss = (weights * (q_eval - q_target) ** 2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
        
        if e % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy_net.state_dict(), 'smartlight_dueling_dqn_pisa_v3.3_best.pth')
            
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        if (e + 1) % 15 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (e + 1) / elapsed
            eta = (episodes - e - 1) / eps_per_sec
            print(f"Ep {e+1:3d}/{episodes} | Reward: {total_reward:9.2f} | Best: {best_reward:9.2f} | LR: {current_lr:.6f} | ETA: {eta:.0f}s")

    total_time = time.time() - start_time
    torch.save(policy_net.state_dict(), 'smartlight_dueling_dqn_pisa_v3.3.pth')
    
    print("\n" + "=" * 70)
    print("TRAINING v3.3 (DUELING DQN) COMPLETATO!")
    print(f"  Tempo totale: {total_time:.1f} secondi ({total_time/60:.1f} minuti)")
    print(f"  Velocita: {episodes/total_time:.2f} episodi/secondo")
    print(f"  Modello finale: smartlight_dueling_dqn_pisa_v3.3.pth")
    print(f"  Modello migliore: smartlight_dueling_dqn_pisa_v3.3_best.pth")
    print(f"  Best Reward: {best_reward:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    train()
