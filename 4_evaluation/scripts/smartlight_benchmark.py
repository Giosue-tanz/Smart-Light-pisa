import torch
import numpy as np
import random
import time
import torch.nn as nn

# ==============================
# CONFIGURAZIONE & MODELLO
# ==============================
MIN_GREEN, MAX_GREEN = 10, 40
ACTION_SIZE = MAX_GREEN - MIN_GREEN + 1
MODEL_PATH = 'smartlight_dqn_pisa_v1.9.pth'

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, ACTION_SIZE)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x)

# ==============================
# AMBIENTE DI BENCHMARK 4-VIE
# ==============================
class BenchmarkEnv4Way:
    def __init__(self, seed=42):
        self.seed = seed
        self.reset()
        
    def reset(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Code per due direzioni: 0=NS (Nord-Sud), 1=EW (Est-Ovest)
        self.veh_q = [0, 0] 
        self.ped_q = [0, 0] # Semplificato, accoppiato ai veicoli
        
        self.current_phase = 0 # 0=NS Green, 1=EW Green
        
        # Statistiche
        self.total_cars_passed = 0
        self.total_wait_time = 0
        self.max_q = 0
        self.co2 = 0
        
        self.hour = 8.0
        self.prev_green = 15
        
        return self._get_state()

    def _get_state(self):
        # L'AI vede solo la fase attiva come "sua" responsabilità
        # Mappiamo la fase attiva sugli input che l'AI si aspetta
        active = self.current_phase
        
        return np.array([
            self.ped_q[active] / 20,
            self.veh_q[active] / 10,
            0, # Disabili off per benchmark
            0, 0, # Wait times normalized
            (self.prev_green - MIN_GREEN) / (MAX_GREEN - MIN_GREEN),
            self.hour / 24,
            0
        ], dtype=np.float32)

    def step(self, green_time):
        active = self.current_phase
        inactive = 1 - active
        
        # 1. Deflusso (Solo fase attiva)
        # Capacità: 1 auto ogni 2.5s
        capacity = int(green_time / 2.5)
        passed = min(self.veh_q[active], capacity)
        
        self.veh_q[active] -= passed
        self.total_cars_passed += passed
        
        # 2. Statistiche & Attesa
        # Chi è nella fase attiva e non passa, aspetta
        self.total_wait_time += self.veh_q[active] * green_time
        # Chi è nella fase inattiva aspetta TUTTO il tempo
        self.total_wait_time += self.veh_q[inactive] * green_time
        
        self.co2 += (self.veh_q[active] + self.veh_q[inactive]) * green_time * 0.12
        
        current_max = max(self.veh_q[0], self.veh_q[1])
        if current_max > self.max_q: self.max_q = current_max

        # 3. Nuovi Arrivi (Asimmetrici per mettere in crisi il fisso)
        # NS (0): Traffico Pesante (es. strada principale)
        # EW (1): Traffico Leggero (es. traversa)
        arrival_rate_ns = 3.0 if (8 <= self.hour <= 9) else 1.5
        arrival_rate_ew = 0.8 # Costante basso
        
        new_ns = np.random.poisson(arrival_rate_ns * (green_time/10))
        new_ew = np.random.poisson(arrival_rate_ew * (green_time/10))
        
        self.veh_q[0] += new_ns
        self.veh_q[1] += new_ew
        
        # 4. Cambio Fase
        self.current_phase = inactive # Switch
        self.prev_green = green_time
        self.hour += (green_time / 3600)
        
        return self._get_state()

# ==============================
# STRATEGIE
# ==============================
def run_fixed_timer(duration_hours=1):
    env = BenchmarkEnv4Way(seed=999)
    # Ciclo fisso: 30s Green NS -> 30s Green EW
    # Totale ciclo 60s.
    cycles = int(duration_hours * 3600 / 60)
    
    for _ in range(cycles):
        env.step(30) # NS Green
        env.step(30) # EW Green
        
    return env

def run_smart_ai(duration_hours=1):
    env = BenchmarkEnv4Way(seed=999) # STESSO SEED
    
    net = DQN()
    try:
        net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        net.eval()
    except:
        print("ERRORE MODELLO")
        return env

    current_time = 0
    max_time = duration_hours * 3600
    
    state = env._get_state()
    
    while current_time < max_time:
        # L'AI decide quanto dare alla fase CORRENTE
        with torch.no_grad():
            q_values = net(torch.FloatTensor(state).unsqueeze(0))
            action = torch.argmax(q_values).item()
        
        green_time = MIN_GREEN + action
        state = env.step(green_time) # Esegue e switcha fase
        current_time += green_time
        
    return env

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("Esecuzione Benchmark 4-Vie: Asymmetric Traffic (Heavy NS vs Light EW)...")
    
    fixed_env = run_fixed_timer()
    ai_env = run_smart_ai()
    
    fixed_avg_wait = fixed_env.total_wait_time / max(1, fixed_env.total_cars_passed)
    ai_avg_wait = ai_env.total_wait_time / max(1, ai_env.total_cars_passed)
    
    print("\n" + "="*50)
    print("RISULTATI INCROCIO 4-VIE (1 Ora)")
    print("="*50)
    print(f"{'METRICA':<25} | {'FIXED (30s/30s)':<15} | {'SMART AI':<12} | {'DIFF':<10}")
    print("-" * 65)
    
    print(f"{'Auto Passate':<25} | {fixed_env.total_cars_passed:<15} | {ai_env.total_cars_passed:<12} | {ai_env.total_cars_passed - fixed_env.total_cars_passed:+d}")
    print(f"{'Attesa Media (sec)':<25} | {fixed_avg_wait:<15.1f} | {ai_avg_wait:<12.1f} | {ai_avg_wait - fixed_avg_wait:+.1f}")
    print(f"{'Max Coda (veicoli)':<25} | {fixed_env.max_q:<15} | {ai_env.max_q:<12} | {ai_env.max_q - fixed_env.max_q:+d}")
    print(f"{'CO2 (kg)':<25} | {fixed_env.co2/1000:<15.2f} | {ai_env.co2/1000:<12.2f} | {(ai_env.co2 - fixed_env.co2)/1000:+.2f}")
    
    # Salva report
    with open("benchmark_results_4way.md", "w") as f:
        f.write("# Benchmark 4-Way Intersection: Smart Light v1.9 vs Fixed Timer\n\n")
        f.write("Scenario: **Traffico Asimmetrico** (Nord-Sud Pesante, Est-Ovest Leggero).\n")
        f.write("Il Fixed Timer spreca tempo su EW mentre NS si intasa. L'AI dovrebbe bilanciare.\n\n")
        f.write("| Metrica | Fixed Timer | Smart AI | Miglioramento |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Auto Smaltite** | {fixed_env.total_cars_passed} | {ai_env.total_cars_passed} | **{ai_env.total_cars_passed - fixed_env.total_cars_passed:+d}** |\n")
        f.write(f"| **Attesa Media** | {fixed_avg_wait:.1f}s | {ai_avg_wait:.1f}s | **{ai_avg_wait - fixed_avg_wait:+.1f}s** |\n")
        f.write(f"| **Max Coda** | {fixed_env.max_q} | {ai_env.max_q} | **{ai_env.max_q - fixed_env.max_q:+d}** |\n")
        f.write(f"| **CO2 (kg)** | {fixed_env.co2/1000:.2f} | {ai_env.co2/1000:.2f} | **{(ai_env.co2 - fixed_env.co2)/1000:+.2f}** |\n")
