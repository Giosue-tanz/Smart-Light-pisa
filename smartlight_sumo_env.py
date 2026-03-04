import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import sumolib

# ==============================
# CONFIGURAZIONE AMBIENTE SUMO
# ==============================
SUMO_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_tools", "sumo-1.22.0")
SUMO_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_config", "pisa.sumocfg")

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME
    print(f"SUMO_HOME impostata a: {SUMO_HOME}")

# Aggiungi la cartella bin al PATH per le DLL di SUMO
sumo_bin_dir = os.path.join(SUMO_HOME, 'bin')
if sumo_bin_dir not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + sumo_bin_dir
    print(f"Aggiunto {sumo_bin_dir} al PATH di sistema.")

tools = os.path.join(SUMO_HOME, 'tools')
sys.path.append(tools)

# Verifica preliminare file config
if not os.path.exists(SUMO_CONFIG):
    print(f"\n[ATTENZIONE] Il file {os.path.abspath(SUMO_CONFIG)} non è stato trovato!")
    print("Genera la configurazione con genera_scenario_medio.py\n")


class SumoIntersectionEnv(gym.Env):
    """
    Ambiente Gymnasium per controllare un singolo incrocio in SUMO.
    Compatibile con Stable Baselines3 e PPO custom.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, use_gui=False, max_steps=3600, tls_id=None, min_green=5):
        super(SumoIntersectionEnv, self).__init__()
        
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.step_counter = 0
        self.tls_id = tls_id  # None = auto-detect il semaforo più grande
        self.min_green = min_green
        
        # Action Space: 
        # 0: Keep Phase (Mantieni Verde corrente)
        # 1: Switch Phase (Passa alla prossima fase)
        self.action_space = spaces.Discrete(2)
        
        # Observation Space (11 dim):
        # [queue_N, queue_E, queue_S, queue_W, total_halting, total_waiting_norm,  
        #  current_phase, phase_duration_norm, avg_speed_norm, density_norm, hour_norm]
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        
        sumo_binary_name = "sumo-gui.exe" if self.use_gui else "sumo.exe"
        sumo_binary_path = os.path.join(SUMO_HOME, "bin", sumo_binary_name)
        
        self.sumo_cmd = [
            sumo_binary_path,
            "-c", SUMO_CONFIG,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--start",
        ]
        
        self.lanes = []
        self.current_phase_duration = 0
        self.num_phases = 4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        try:
            traci.close()
        except:
            pass
            
        traci.start(self.sumo_cmd)
        
        self.step_counter = 0
        self.current_phase_duration = 0

        # AUTO-CONFIGURAZIONE: Rileva semafori nella mappa
        all_tls = traci.trafficlight.getIDList()
        if not all_tls:
            raise ValueError("Nessun semaforo trovato nella mappa!")
        
        if self.tls_id is None or self.tls_id not in all_tls:
            # Scegli il semaforo con più corsie controllate
            best_tls = max(all_tls, key=lambda x: len(traci.trafficlight.getControlledLinks(x)))
            self.tls_id = best_tls
            print(f"[AUTO-SETUP] Selezionato semaforo principale: {self.tls_id}")

        # Corsie controllate (rimuovi duplicati)
        raw_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        self.lanes = sorted(list(set(raw_lanes)))
        
        # Numero di fasi
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)
        if logic:
            self.num_phases = len(logic[0].phases)
        
        print(f"[SETUP] TLS: {self.tls_id} | Corsie: {len(self.lanes)} | Fasi: {self.num_phases}")
        
        traci.trafficlight.setPhase(self.tls_id, 0)
        
        return self._get_observation(), {}

    def step(self, action):
        if action == 1 and self.current_phase_duration > self.min_green:
            current = traci.trafficlight.getPhase(self.tls_id)
            next_phase = (current + 1) % self.num_phases
            traci.trafficlight.setPhase(self.tls_id, next_phase)
            self.current_phase_duration = 0
        else:
            self.current_phase_duration += 1
            
        traci.simulationStep()
        self.step_counter += 1
        
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self.step_counter >= self.max_steps
        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        # Conta veicoli fermi per direzione (dividi corsie in 4 gruppi)
        queues = []
        for lane in self.lanes:
            try:
                queues.append(traci.lane.getLastStepHaltingNumber(lane))
            except:
                queues.append(0)
        
        # Dividi code in 4 direzioni (approssimativamente)
        n_groups = min(4, len(queues))
        group_size = max(1, len(queues) // n_groups) if n_groups > 0 else 1
        directional_queues = [0.0] * 4
        for i in range(min(4, n_groups)):
            start = i * group_size
            end = start + group_size if i < 3 else len(queues)
            directional_queues[i] = sum(queues[start:end])
        
        total_halting = sum(queues)
        total_wt = self._get_total_waiting_time()
        
        # Velocità media sulle corsie
        speeds = []
        for lane in self.lanes:
            try:
                speeds.append(traci.lane.getLastStepMeanSpeed(lane))
            except:
                speeds.append(0)
        avg_speed = np.mean(speeds) if speeds else 0
        
        # Densità (veicoli/corsia)
        total_vehicles = 0
        for lane in self.lanes:
            try:
                total_vehicles += traci.lane.getLastStepVehicleNumber(lane)
            except:
                pass
        density = total_vehicles / max(1, len(self.lanes))
        
        # Fase corrente
        try:
            current_phase = traci.trafficlight.getPhase(self.tls_id)
        except:
            current_phase = 0
        
        obs = np.array([
            min(directional_queues[0] / 20.0, 1.0),  # queue_N
            min(directional_queues[1] / 20.0, 1.0),  # queue_E
            min(directional_queues[2] / 20.0, 1.0),  # queue_S
            min(directional_queues[3] / 20.0, 1.0),  # queue_W
            min(total_halting / 50.0, 1.0),            # total halting
            min(total_wt / 1000.0, 1.0),               # total waiting time
            current_phase / max(1, self.num_phases - 1),  # current phase
            min(self.current_phase_duration / 60.0, 1.0), # phase duration
            min(avg_speed / 15.0, 1.0),                # avg speed
            min(density / 10.0, 1.0),                  # density
            (self.step_counter / 3600.0) % 1.0         # time of day
        ], dtype=np.float32)
        
        return obs

    def _get_total_waiting_time(self):
        total_wt = 0
        for lane in self.lanes:
            try:
                total_wt += traci.lane.getWaitingTime(lane)
            except:
                pass
        return total_wt

    def _compute_reward(self):
        wt = self._get_total_waiting_time()
        q = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])
        
        # Reward: penalizza waiting time e code
        penalty = wt + 10 * q
        return -penalty / 1000.0

    def close(self):
        try:
            traci.close()
        except:
            pass


# TEST RAPIDO SE ESEGUITO COME SCRIPT
if __name__ == "__main__":
    env = SumoIntersectionEnv(use_gui=True)
    obs, _ = env.reset()
    print("Ambiente SUMO inizializzato. Stato iniziale:", obs)
    
    total_reward = 0
    for i in range(2000):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if i % 50 == 0:
            print(f"Step {i}: Reward {reward:.4f} | Total Wait: {env._get_total_waiting_time():.1f}")
        
    print("\nTest completato. La finestra di SUMO rimarrà aperta.")
    print("Premi INVIO nel terminale per chiudere...")
    input()
    env.close()
