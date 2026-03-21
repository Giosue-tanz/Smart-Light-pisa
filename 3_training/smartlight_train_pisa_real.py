import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import time
import traci
import sumolib
# Aggiungi 1_simulation al path per importare il traffic generator
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, '1_simulation'))
from smartlight_traffic_gen import PisaTrafficGenerator

# ==============================
# CONFIGURAZIONE
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUMO_BINARY = "sumo-gui" # Usa gui per vedere, "sumo" per velocità
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_CONFIG = os.path.join(BASE_PATH, "1_simulation", "sumo_config", "pisa.sumocfg")
NET_FILE = os.path.join(BASE_PATH, "1_simulation", "sumo_config", "pisa.net.xml")

MIN_GREEN, MAX_GREEN = 10, 40
ACTION_SIZE = 31

# ==============================
# PPO MODEL (UGUALE A v4.0)
# ==============================
class ActorCritic(nn.Module):
    def __init__(self, state_size=11, action_size=31):
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
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        return self.shared(state)
    
    def get_action_and_value(self, state, action=None):
        shared_features = self.shared(state)
        logits = self.actor(shared_features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        value = self.critic(shared_features)
        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)

# ==============================
# AMBIENTE REALE (SUMO WRAPPER)
# ==============================
class SumoRealEnv:
    def __init__(self, use_gui=True):
        self.generator = PisaTrafficGenerator(NET_FILE)
        self.tls_id = None
        self.lanes = []
        self.use_gui = use_gui
        self.episode_count = 0
        
        # Tools setup
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
    
    def start(self, scenario="balanced"):
        self.episode_count += 1
        
        # 1. Genera Rotte Dinamiche
        # Nota: La prima volta non abbiamo gli edge, quindi SUMO parte con config base
        # Poi rileviamo gli edge e dalla seconda volta generiamo traffico mirato
        route_file = None
        if self.tls_id:
            print(f"Generazione scenario '{scenario}'...")
            route_file = self.generator.generate_route_file(self.episode_count, scenario)
        
        # 2. Comando di avvio
        sumo_bin = "sumo-gui" if self.use_gui else "sumo"
        cmd = [sumo_bin, "-c", SUMO_CONFIG, "--no-step-log", "true", "--waiting-time-memory", "1000"]
        
        if route_file:
            # Sovrascrive il file rotte del config
            cmd.extend(["-r", route_file])
            
        if self.use_gui:
            cmd.append("--start")
            
        traci.start(cmd)
        
        # 3. Auto-configurazione al primo avvio
        if not self.tls_id:
            all_tls = traci.trafficlight.getIDList()
            if all_tls:
                # Prendi il semaforo con più connessioni (probabilmente quello centrale)
                self.tls_id = max(all_tls, key=lambda x: len(traci.trafficlight.getControlledLinks(x)))
                print(f"Semaforo Target Rilevato: {self.tls_id}")
                
                # Rileva edge per il generatore
                edges = self.generator.detect_incoming_edges(self.tls_id)
                self.lanes = traci.trafficlight.getControlledLanes(self.tls_id)
                self.lanes = sorted(list(set(self.lanes))) # Rimuovi duplicati
            else:
                print("ERRORE: Nessun semaforo nella mappa!")
                traci.close()
                return None

        return self._get_state()

    def step(self, action_idx):
        green_time = MIN_GREEN + action_idx
        
        # Applica fase verde (semplificata per ora: estendi o switch)
        # Logica Reale: Dovremmo impostare la durata della fase corrente
        # Per ora: Lasciamo scorrere il tempo (simulando che l'azione sia "Durata del ciclo")
        
        # Esegui N step di simulazione pari al green_time
        rewards = 0
        params = []
        
        # Simuliamo il verde
        # Nota: gestione fasi complessa in SUMO, qui assumiamo che l'IA decida
        # la durata della fase corrente prima di switchare.
        
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        traci.trafficlight.setPhaseDuration(self.tls_id, green_time)
        
        # Avanziamo nel tempo finché la fase non cambia o finisce il tempo
        # In SUMO setPhaseDuration aggiunge tempo alla fase corrente
        
        steps_to_sim = green_time
        for _ in range(steps_to_sim):
            traci.simulationStep()
            
            # Calcolo Reward Accumulato (Wait time negativo)
            waiting = 0
            for lane in self.lanes:
                waiting += traci.lane.getWaitingTime(lane)
            rewards -= waiting
            
        # Normalizza reward
        rewards /= 1000.0
        
        # Switch automatico alla prossima fase (gestito da SUMO se non interveniamo force)
        # Ma per RL spesso vogliamo forzare lo switch DOPO il verde.
        # Qui lasciamo la logica base del semaforo attuata che passa al giallo -> rosso -> verde succ.
        # L'IA controlla solo la durata del VERDE.
        
        done = traci.simulation.getMinExpectedNumber() <= 0
        next_state = self._get_state()
        
        return next_state, rewards, done

    def _get_state(self):
        # 11-dimension vector like training
        # [micro, priv, pub, freight, emerg, ped, dis, ped_wait, veh_wait, time, hour]
        
        total_q = 0
        total_wait = 0
        
        for lane in self.lanes:
            total_q += traci.lane.getLastStepHaltingNumber(lane)
            total_wait += traci.lane.getWaitingTime(lane)
            
        state = np.array([
            0, # micro (non distinto in SUMO simple)
            min(total_q / 50.0, 1.0), # private proxy
            0, # public
            0, # freight
            0, # emerg
            0, # ped
            0, # dis
            0, # ped wait
            min(total_wait / 1000.0, 1.0),
            0.5, # time phases (mock)
            0.5 # hour
        ], dtype=np.float32)
        return state

    def close(self):
        try:
            traci.close()
        except:
            pass

# ==============================
# MAIN LOOP
# ==============================
def train_real():
    print("="*60)
    print("SMART LIGHT PISA: TRAINING SU MAPPA REALE (SUMO)")
    print("="*60)
    
    env = SumoRealEnv(use_gui=True)
    model = ActorCritic(state_size=11, action_size=31).to(DEVICE)
    
    # Carica pesi pre-allenati se esistono
    try:
        model.load_state_dict(torch.load("smartlight_ppo_pisa_v4.0_best.pth", map_location=DEVICE))
        print("[INFO] Modello pre-allenato caricato con successo!")
    except:
        print("[WARN] Nessun modello trovato, si parte da zero (LENTO!)")
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # LR basso per fine-tuning
    
    scenarios = ["morning_rush", "evening_rush", "balanced"]
    
    for episode in range(10): # Pochi episodi dimostrativi
        scenario = random.choice(scenarios)
        state = env.start(scenario)
        if state is None: break
        
        ep_reward = 0
        step = 0
        done = False
        
        print(f"Episodio {episode+1}: Scenario {scenario}")
        
        while not done and step < 30: # Short episodes for testing
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action, log_prob, _, value = model.get_action_and_value(state_t)
            action_idx = action.item()
            
            next_state, reward, done = env.step(action_idx)
            
            # Qui andrebbe il PPO update buffer (semplificato per test)
            
            state = next_state
            ep_reward += reward
            step += 1
            
        print(f"  -> Reward: {ep_reward:.2f}")
        env.close()

if __name__ == "__main__":
    train_real()
