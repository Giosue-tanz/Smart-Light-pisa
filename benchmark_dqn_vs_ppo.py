"""
SmartLight Pisa - Benchmark Comparativo
DQN v3.4 (Dueling Double DQN) vs PPO v4.0

Confronta i due modelli sugli stessi scenari di traffico.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import random
from collections import defaultdict

# ==============================
# CONFIGURAZIONE
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_GREEN, MAX_GREEN = 10, 40
ACTION_SIZE = MAX_GREEN - MIN_GREEN + 1

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
# AMBIENTE
# ==============================
class TrafficSimEnv:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
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
        
        # Metriche per benchmark
        self.total_vehicles_passed = 0
        self.total_wait_time = 0
        self.emergency_handled = 0
        self.steps = 0
        
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
        self.steps += 1
        
        queues = (self.micro_q, self.private_q, self.public_q, 
                  self.freight_q, self.emergency_q, self.ped_q)
        waits = (self.ped_wait, self.veh_wait)
        reward = calculate_reward(queues, waits, self.dis_ped > 0, self.emergency_q > 0)
        
        # Track wait time
        self.total_wait_time += self.ped_wait + self.veh_wait
        
        ped_passed = min(self.ped_q, green_time // 2)
        self.ped_q -= ped_passed
        if self.ped_q == 0: self.ped_wait = 0
        if self.dis_ped > 0 and ped_passed > 0: self.dis_ped = 0
        
        capacity = green_time // 2
        vehicles_this_step = 0
        
        if self.emergency_q > 0:
            em_pass = min(self.emergency_q, 2)
            self.emergency_q -= em_pass
            capacity -= em_pass
            vehicles_this_step += em_pass
            self.emergency_handled += em_pass
        
        if capacity > 0 and self.public_q > 0:
            pub_pass = min(self.public_q, capacity // 3)
            self.public_q -= pub_pass
            capacity -= pub_pass * 3
            vehicles_this_step += pub_pass
        
        if capacity > 0:
            micro_pass = min(self.micro_q, capacity * 2)
            self.micro_q -= micro_pass
            vehicles_this_step += micro_pass
        
        if capacity > 0:
            priv_pass = min(self.private_q, capacity)
            self.private_q -= priv_pass
            capacity -= priv_pass
            vehicles_this_step += priv_pass
        
        if capacity > 0 and self.freight_q > 0:
            freight_pass = min(self.freight_q, capacity // 3)
            self.freight_q -= freight_pass
            vehicles_this_step += freight_pass
        
        self.total_vehicles_passed += vehicles_this_step
        
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
    
    def get_metrics(self):
        return {
            'vehicles_passed': self.total_vehicles_passed,
            'avg_wait': self.total_wait_time / max(1, self.steps),
            'emergency_handled': self.emergency_handled,
            'steps': self.steps
        }


# ==============================
# DUELING DQN (per caricare il modello)
# ==============================
class DuelingDQN(nn.Module):
    def __init__(self, input_size=11):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.value_fc = nn.Linear(256, 128)
        self.value_bn = nn.BatchNorm1d(128)
        self.value_out = nn.Linear(128, 1)
        
        self.advantage_fc = nn.Linear(256, 128)
        self.advantage_bn = nn.BatchNorm1d(128)
        self.advantage_out = nn.Linear(128, ACTION_SIZE)
    
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


# ==============================
# PPO ACTOR-CRITIC (per caricare il modello)
# ==============================
class ActorCritic(nn.Module):
    def __init__(self, state_size=11, action_size=ACTION_SIZE):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def get_action(self, state):
        shared_features = self.shared(state)
        logits = self.actor(shared_features)
        probs = Categorical(logits=logits)
        return probs.sample().item()


# ==============================
# FIXED TIMER (baseline)
# ==============================
def fixed_timer_action():
    """Semaforo fisso: sempre 25 secondi"""
    return 25 - MIN_GREEN  # azione = 15


# ==============================
# BENCHMARK
# ==============================
def run_benchmark(model, model_name, env, episodes=50, steps_per_episode=120):
    """Esegue il benchmark per un modello"""
    
    all_rewards = []
    all_vehicles = []
    all_waits = []
    all_emergencies = []
    
    for ep in range(episodes):
        # Reset con seed fisso per confronto equo
        random.seed(ep * 42)
        np.random.seed(ep * 42)
        state = env.reset()
        
        total_reward = 0
        
        for _ in range(steps_per_episode):
            if model is None:
                # Fixed timer
                action = fixed_timer_action()
            elif isinstance(model, DuelingDQN):
                # DQN
                model.eval()
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(DEVICE)
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()
            else:
                # PPO
                model.eval()
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    action = model.get_action(state_tensor)
            
            state, reward, _ = env.step(action)
            total_reward += reward
        
        metrics = env.get_metrics()
        all_rewards.append(total_reward)
        all_vehicles.append(metrics['vehicles_passed'])
        all_waits.append(metrics['avg_wait'])
        all_emergencies.append(metrics['emergency_handled'])
    
    return {
        'name': model_name,
        'avg_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'avg_vehicles': np.mean(all_vehicles),
        'avg_wait': np.mean(all_waits),
        'avg_emergency': np.mean(all_emergencies),
    }


def main():
    print("=" * 70)
    print("SMARTLIGHT PISA - BENCHMARK COMPARATIVO")
    print("DQN v3.4 vs PPO v4.0 vs Fixed Timer")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print()
    
    # Carica modelli
    print("[1/3] Caricamento modelli...")
    
    # DQN
    dqn_model = DuelingDQN(input_size=11).to(DEVICE)
    try:
        dqn_model.load_state_dict(torch.load('smartlight_dueling_dqn_pisa_v3.4_best.pth', 
                                              map_location=DEVICE, weights_only=True))
        print("  [OK] DQN v3.4 caricato")
    except FileNotFoundError:
        try:
            dqn_model.load_state_dict(torch.load('smartlight_dueling_dqn_pisa_v3.3_best.pth', 
                                                  map_location=DEVICE, weights_only=True))
            print("  [OK] DQN v3.3 caricato (v3.4 non trovato)")
        except:
            print("  [X] DQN non trovato!")
            dqn_model = None
    
    # PPO
    ppo_model = ActorCritic(state_size=11, action_size=ACTION_SIZE).to(DEVICE)
    try:
        ppo_model.load_state_dict(torch.load('smartlight_ppo_pisa_v4.0_best.pth', 
                                              map_location=DEVICE, weights_only=True))
        print("  [OK] PPO v4.0 caricato")
    except FileNotFoundError:
        print("  [X] PPO non trovato!")
        ppo_model = None
    
    print()
    
    # Benchmark
    print("[2/3] Esecuzione benchmark (50 episodi x 120 step)...")
    print()
    
    env = TrafficSimEnv()
    
    # Fixed Timer (baseline)
    print("  Testing Fixed Timer...")
    fixed_results = run_benchmark(None, "Fixed Timer (25s)", env)
    
    # DQN
    if dqn_model:
        print("  Testing DQN v3.4...")
        dqn_results = run_benchmark(dqn_model, "DQN v3.4 (Dueling Double)", env)
    
    # PPO
    if ppo_model:
        print("  Testing PPO v4.0...")
        ppo_results = run_benchmark(ppo_model, "PPO v4.0 (Actor-Critic)", env)
    
    print()
    
    # Risultati
    print("[3/3] RISULTATI")
    print("=" * 70)
    print()
    
    print(f"{'Metrica':<25} {'Fixed Timer':>15} {'DQN v3.4':>15} {'PPO v4.0':>15}")
    print("-" * 70)
    
    print(f"{'Reward Medio':<25} {fixed_results['avg_reward']:>15.2f} ", end="")
    if dqn_model:
        print(f"{dqn_results['avg_reward']:>15.2f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if ppo_model:
        print(f"{ppo_results['avg_reward']:>15.2f}")
    else:
        print(f"{'N/A':>15}")
    
    print(f"{'Veicoli Smaltiti':<25} {fixed_results['avg_vehicles']:>15.0f} ", end="")
    if dqn_model:
        print(f"{dqn_results['avg_vehicles']:>15.0f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if ppo_model:
        print(f"{ppo_results['avg_vehicles']:>15.0f}")
    else:
        print(f"{'N/A':>15}")
    
    print(f"{'Attesa Media':<25} {fixed_results['avg_wait']:>15.1f} ", end="")
    if dqn_model:
        print(f"{dqn_results['avg_wait']:>15.1f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if ppo_model:
        print(f"{ppo_results['avg_wait']:>15.1f}")
    else:
        print(f"{'N/A':>15}")
    
    print(f"{'Emergenze Gestite':<25} {fixed_results['avg_emergency']:>15.1f} ", end="")
    if dqn_model:
        print(f"{dqn_results['avg_emergency']:>15.1f} ", end="")
    else:
        print(f"{'N/A':>15} ", end="")
    if ppo_model:
        print(f"{ppo_results['avg_emergency']:>15.1f}")
    else:
        print(f"{'N/A':>15}")
    
    print()
    print("=" * 70)
    
    # Miglioramenti
    print("\n>> MIGLIORAMENTI vs Fixed Timer:")
    print("-" * 70)
    
    if dqn_model:
        reward_imp_dqn = ((dqn_results['avg_reward'] - fixed_results['avg_reward']) / abs(fixed_results['avg_reward'])) * 100
        veh_imp_dqn = ((dqn_results['avg_vehicles'] - fixed_results['avg_vehicles']) / fixed_results['avg_vehicles']) * 100
        wait_imp_dqn = ((fixed_results['avg_wait'] - dqn_results['avg_wait']) / fixed_results['avg_wait']) * 100
        
        print(f"DQN v3.4:  Reward {reward_imp_dqn:+.1f}% | Veicoli {veh_imp_dqn:+.1f}% | Attesa {wait_imp_dqn:+.1f}%")
    
    if ppo_model:
        reward_imp_ppo = ((ppo_results['avg_reward'] - fixed_results['avg_reward']) / abs(fixed_results['avg_reward'])) * 100
        veh_imp_ppo = ((ppo_results['avg_vehicles'] - fixed_results['avg_vehicles']) / fixed_results['avg_vehicles']) * 100
        wait_imp_ppo = ((fixed_results['avg_wait'] - ppo_results['avg_wait']) / fixed_results['avg_wait']) * 100
        
        print(f"PPO v4.0:  Reward {reward_imp_ppo:+.1f}% | Veicoli {veh_imp_ppo:+.1f}% | Attesa {wait_imp_ppo:+.1f}%")
    
    # Vincitore
    print()
    print("=" * 70)
    print("** VINCITORE **")
    
    if dqn_model and ppo_model:
        if dqn_results['avg_reward'] > ppo_results['avg_reward']:
            winner = "DQN v3.4"
            diff = dqn_results['avg_reward'] - ppo_results['avg_reward']
        else:
            winner = "PPO v4.0"
            diff = ppo_results['avg_reward'] - dqn_results['avg_reward']
        print(f"   {winner} (reward +{diff:.2f} rispetto all'altro)")
    elif dqn_model:
        print("   DQN v3.4 (PPO non disponibile)")
    elif ppo_model:
        print("   PPO v4.0 (DQN non disponibile)")
    
    print("=" * 70)


    # Scrivi su file
    with open("final_benchmark.txt", "w") as f:
        f.write("BENCHMARK RISULTATI\n")
        f.write("===================\n")
        f.write(f"Fixed Timer Reward: {fixed_results['avg_reward']:.2f}\n")
        if dqn_model:
            f.write(f"DQN Reward: {dqn_results['avg_reward']:.2f}\n")
        if ppo_model:
            f.write(f"PPO Reward: {ppo_results['avg_reward']:.2f}\n")
            
        f.write("\nCONFRONTO\n")
        if dqn_model and ppo_model:
            if dqn_results['avg_reward'] > ppo_results['avg_reward']:
                f.write(f"Vincitore: DQN (+{dqn_results['avg_reward'] - ppo_results['avg_reward']:.2f})\n")
            else:
                f.write(f"Vincitore: PPO (+{ppo_results['avg_reward'] - dqn_results['avg_reward']:.2f})\n")

if __name__ == "__main__":
    main()
