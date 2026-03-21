"""
SmartLight Pisa v3.4 - PARALLEL TRAINING
=========================================
Fase 3 COMPLETA: GPU + Parallel Environment Simulation

Features implementate:
- [Fase 1] Prioritized Experience Replay
- [Fase 1] Learning Rate Scheduler
- [Fase 1] State Compression (17->11)
- [Fase 1] Batch Normalization
- [Fase 2] Dueling DQN Architecture
- [Fase 2] Double DQN (NEW!)
- [Fase 3] GPU Acceleration (CUDA)
- [Fase 3] Parallel Environment Simulation (NEW!)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from collections import deque
import multiprocessing as mp
from multiprocessing import Process, Pipe
import time

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
        
    def push_batch(self, experiences):
        """Push multipli experiences in batch (per ambienti paralleli)"""
        for exp in experiences:
            self.push(*exp)
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
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
# DUELING DQN CON BATCH NORMALIZATION
# ==============================
class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    """
    def __init__(self, input_size=11):
        super().__init__()
        
        # Shared Feature Layers
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Value Stream
        self.value_fc = nn.Linear(256, 128)
        self.value_bn = nn.BatchNorm1d(128)
        self.value_out = nn.Linear(128, 1)
        
        # Advantage Stream
        self.advantage_fc = nn.Linear(256, 128)
        self.advantage_bn = nn.BatchNorm1d(128)
        self.advantage_out = nn.Linear(128, ACTION_SIZE)
        
        # Xavier initialization
        for layer in [self.fc1, self.fc2, self.value_fc, self.value_out, 
                      self.advantage_fc, self.advantage_out]:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, x):
        single_input = x.dim() == 1
        if single_input:
            x = x.unsqueeze(0)
        
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        
        value = torch.relu(self.value_bn(self.value_fc(x)))
        value = self.value_out(value)
        
        advantage = torch.relu(self.advantage_bn(self.advantage_fc(x)))
        advantage = self.advantage_out(advantage)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        if single_input:
            q_values = q_values.squeeze(0)
        return q_values

DQN = DuelingDQN

# ==============================
# VEHICLE WEIGHTS
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
# TRAFFIC SIMULATION ENVIRONMENT
# ==============================
class TrafficSimEnv:
    def __init__(self, env_id=0):
        self.env_id = env_id
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
        # Seed diverso per ogni ambiente parallelo
        random.seed(env_id + int(time.time() * 1000) % 10000)
        np.random.seed(env_id + int(time.time() * 1000) % 10000)
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
# PARALLEL ENVIRONMENT WORKER
# ==============================
def env_worker(conn, env_id):
    """Worker process per ambiente parallelo"""
    env = TrafficSimEnv(env_id=env_id)
    
    while True:
        cmd, data = conn.recv()
        
        if cmd == 'step':
            state, reward, done = env.step(data)
            conn.send((state, reward, done))
        elif cmd == 'reset':
            state = env.reset()
            conn.send(state)
        elif cmd == 'close':
            conn.close()
            break

class ParallelEnvs:
    """
    Gestore di ambienti paralleli per raccolta dati veloce
    
    Fase 3: Parallel Environment Simulation
    - Esegue N ambienti in parallelo usando multiprocessing
    - Raccoglie esperienze N volte più velocemente
    - Aumenta la diversità dei dati di training
    """
    
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.processes = []
        self.conns = []
        
        for i in range(num_envs):
            parent_conn, child_conn = Pipe()
            p = Process(target=env_worker, args=(child_conn, i))
            p.start()
            self.processes.append(p)
            self.conns.append(parent_conn)
    
    def reset_all(self):
        """Reset di tutti gli ambienti"""
        for conn in self.conns:
            conn.send(('reset', None))
        states = [conn.recv() for conn in self.conns]
        return np.array(states)
    
    def step_all(self, actions):
        """Esegue un'azione in tutti gli ambienti in parallelo"""
        for conn, action in zip(self.conns, actions):
            conn.send(('step', action))
        
        results = [conn.recv() for conn in self.conns]
        states = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        
        return states, rewards, dones
    
    def close(self):
        """Chiude tutti i processi worker"""
        for conn in self.conns:
            conn.send(('close', None))
        for p in self.processes:
            p.join()

# ==============================
# VECTORIZED ENVIRONMENTS (THREAD-BASED ALTERNATIVE)
# ==============================
class VectorizedEnvs:
    """
    Ambienti vettorizzati senza multiprocessing (per Windows compatibility)
    Esegue N ambienti in modo sincronizzato
    """
    
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.envs = [TrafficSimEnv(env_id=i) for i in range(num_envs)]
    
    def reset_all(self):
        """Reset di tutti gli ambienti"""
        states = [env.reset() for env in self.envs]
        return np.array(states)
    
    def step_all(self, actions):
        """Esegue azioni in tutti gli ambienti"""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        states = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        return states, rewards, dones
    
    def close(self):
        """Cleanup (noop per questa implementazione)"""
        pass

# ==============================
# TRAINING CON FASE 3 COMPLETA
# ==============================
def train(num_parallel_envs=4, use_multiprocessing=False):
    """
    Training con Fase 3 completa:
    - GPU Acceleration
    - Parallel Environment Simulation
    - Double DQN (bonus Fase 2)
    """
    print("=" * 70)
    print("SMARTLIGHT PISA v3.4 - PARALLEL DUELING DOUBLE DQN")
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
    print("[OK] Dueling DQN Architecture (Fase 2)")
    print("[OK] Double DQN (Fase 2) [NEW!]")
    print("[OK] GPU Acceleration (Fase 3)")
    print(f"[OK] Parallel Environments x{num_parallel_envs} (Fase 3) [NEW!]")
    print("=" * 70)
    
    # Inizializza ambienti paralleli
    if use_multiprocessing:
        print("\nUsando MultiProcessing per ambienti paralleli...")
        envs = ParallelEnvs(num_envs=num_parallel_envs)
    else:
        print("\nUsando Vectorized Envs (thread-safe per Windows)...")
        envs = VectorizedEnvs(num_envs=num_parallel_envs)
    
    # Reti su GPU
    policy_net = DQN(input_size=11).to(DEVICE)
    target_net = DQN(input_size=11).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.7)
    
    memory = PrioritizedReplayBuffer(capacity=150000, alpha=0.6, beta=0.4)
    
    episodes = 300
    steps_per_episode = 120
    batch_size = 512  # Batch più grande con più dati paralleli
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.995
    
    best_reward = -float('inf')
    
    print(f"\nTraining: {episodes} episodi")
    print(f"  - {num_parallel_envs} ambienti paralleli")
    print(f"  - {steps_per_episode} steps/episodio")
    print(f"  - {num_parallel_envs * steps_per_episode} transizioni/episodio")
    print(f"  - Batch size: {batch_size}")
    print()
    
    start_time = time.time()
    total_transitions = 0
    
    for e in range(episodes):
        states = envs.reset_all()
        episode_rewards = np.zeros(num_parallel_envs)
        policy_net.train()
        
        for step in range(steps_per_episode):
            # Epsilon-greedy per tutti gli ambienti
            if random.random() <= epsilon:
                actions = np.random.randint(0, ACTION_SIZE, size=num_parallel_envs)
            else:
                policy_net.eval()
                with torch.no_grad():
                    states_tensor = torch.FloatTensor(states).to(DEVICE)
                    q_values = policy_net(states_tensor)
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()
                policy_net.train()
            
            # Step parallelo in tutti gli ambienti
            next_states, rewards, dones = envs.step_all(actions)
            
            # Salva tutte le esperienze
            for i in range(num_parallel_envs):
                memory.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
                episode_rewards[i] += rewards[i]
            
            states = next_states
            total_transitions += num_parallel_envs
            
            # Training step (più frequente con più dati)
            if len(memory) > batch_size and step % 2 == 0:
                batch, indices, weights = memory.sample(batch_size)
                
                # Tensori su GPU
                batch_states = torch.FloatTensor(np.array([x[0] for x in batch])).to(DEVICE)
                batch_actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(1).to(DEVICE)
                batch_rewards = torch.FloatTensor([x[2] for x in batch]).to(DEVICE)
                batch_next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(DEVICE)
                batch_dones = torch.FloatTensor([x[4] for x in batch]).to(DEVICE)
                
                # Current Q-values
                q_eval = policy_net(batch_states).gather(1, batch_actions).squeeze()
                
                # ========== DOUBLE DQN ==========
                # Fase 2: Seleziona azione con policy_net, valuta con target_net
                # Questo riduce l'overestimation dei Q-values
                with torch.no_grad():
                    # Policy net seleziona le migliori azioni
                    policy_net.eval()
                    best_actions = policy_net(batch_next_states).argmax(dim=1, keepdim=True)
                    policy_net.train()
                    
                    # Target net valuta quelle azioni
                    target_net.eval()
                    q_next = target_net(batch_next_states).gather(1, best_actions).squeeze()
                    q_target = batch_rewards + gamma * q_next * (1 - batch_dones)
                
                # TD-errors per prioritized replay
                td_errors = (q_eval - q_target).detach().cpu().numpy()
                memory.update_priorities(indices, td_errors)
                
                # Weighted loss
                loss = (weights * (q_eval - q_target) ** 2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
        
        # Update target network
        if e % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Reward medio tra tutti gli ambienti paralleli
        avg_reward = episode_rewards.mean()
        max_reward = episode_rewards.max()
        
        if max_reward > best_reward:
            best_reward = max_reward
            torch.save(policy_net.state_dict(), 'smartlight_dueling_dqn_pisa_v3.4_best.pth')
            
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        if (e + 1) % 10 == 0:
            elapsed = time.time() - start_time
            trans_per_sec = total_transitions / elapsed
            eps_per_sec = (e + 1) / elapsed
            eta = (episodes - e - 1) / eps_per_sec
            print(f"Ep {e+1:3d}/{episodes} | "
                  f"Avg: {avg_reward:8.2f} | Max: {max_reward:8.2f} | "
                  f"Best: {best_reward:8.2f} | "
                  f"Trans/s: {trans_per_sec:.0f} | "
                  f"ETA: {eta:.0f}s")

    # Cleanup
    envs.close()
    
    total_time = time.time() - start_time
    torch.save(policy_net.state_dict(), 'smartlight_dueling_dqn_pisa_v3.4.pth')
    
    print("\n" + "=" * 70)
    print("TRAINING v3.4 (PARALLEL DUELING DOUBLE DQN) COMPLETATO!")
    print(f"  Tempo totale: {total_time:.1f} secondi ({total_time/60:.1f} minuti)")
    print(f"  Velocita: {episodes/total_time:.2f} episodi/secondo")
    print(f"  Transizioni totali: {total_transitions:,}")
    print(f"  Transizioni/secondo: {total_transitions/total_time:.0f}")
    print(f"  Modello finale: smartlight_dueling_dqn_pisa_v3.4.pth")
    print(f"  Modello migliore: smartlight_dueling_dqn_pisa_v3.4_best.pth")
    print(f"  Best Reward: {best_reward:.2f}")
    print("=" * 70)
    
    return policy_net, best_reward

# ==============================
# BENCHMARK: PARALLEL vs SEQUENTIAL
# ==============================
def benchmark_speedup():
    """Confronta velocità di raccolta dati: sequenziale vs parallelo"""
    print("\n" + "=" * 50)
    print("BENCHMARK: SPEEDUP AMBIENTI PARALLELI")
    print("=" * 50)
    
    n_steps = 1000
    
    # Test sequenziale
    env_single = TrafficSimEnv()
    state = env_single.reset()
    
    start = time.time()
    for _ in range(n_steps):
        action = random.randrange(ACTION_SIZE)
        state, _, _ = env_single.step(action)
    seq_time = time.time() - start
    
    print(f"\nSequenziale (1 env): {n_steps} steps in {seq_time:.3f}s")
    print(f"  -> {n_steps/seq_time:.0f} steps/secondo")
    
    # Test parallelo
    for num_envs in [2, 4, 8]:
        envs = VectorizedEnvs(num_envs=num_envs)
        states = envs.reset_all()
        
        start = time.time()
        for _ in range(n_steps):
            actions = np.random.randint(0, ACTION_SIZE, size=num_envs)
            states, _, _ = envs.step_all(actions)
        par_time = time.time() - start
        
        total_steps = n_steps * num_envs
        speedup = (total_steps / par_time) / (n_steps / seq_time)
        
        print(f"\nParallelo ({num_envs} envs): {total_steps} steps in {par_time:.3f}s")
        print(f"  -> {total_steps/par_time:.0f} steps/secondo")
        print(f"  -> Speedup: {speedup:.2f}x")
        
        envs.close()
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    # Esegui benchmark opzionale
    # benchmark_speedup()
    
    # Training con 4 ambienti paralleli
    train(num_parallel_envs=4, use_multiprocessing=False)
