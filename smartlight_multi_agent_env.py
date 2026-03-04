"""
SmartLight Pisa - Multi-Agent SUMO Environment
================================================
Ambiente multi-agente dove ogni semaforo è controllato da un agente RL indipendente.
Supporta sia Independent PPO (IPPO) che Multi-Agent PPO (MAPPO) con shared critic.

Architettura:
- Ogni agente controlla un singolo semaforo
- Gli agenti possono comunicare informazioni sulle code in uscita ai vicini
- Reward globale = α * R_local + (1-α) * R_global
"""

import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import sumolib

# ==============================
# CONFIGURAZIONE
# ==============================
SUMO_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_tools", "sumo-1.22.0")
SUMO_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_config", "pisa.sumocfg")

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME

sumo_bin_dir = os.path.join(SUMO_HOME, 'bin')
if sumo_bin_dir not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + sumo_bin_dir

tools = os.path.join(SUMO_HOME, 'tools')
sys.path.append(tools)


class MultiAgentSumoEnv(gym.Env):
    """
    Multi-Agent Environment per SUMO.
    
    Ogni semaforo nella rete è controllato da un agente separato.
    L'env espone un'interfaccia che permette il training con IPPO o MAPPO.
    
    Parametri:
        use_gui: bool - Usa SUMO-GUI per visualizzare
        max_steps: int - Passi massimi per episodio
        top_n_tls: int - Numero di semafori "top" da controllare (per complessità)
        min_controlled_lanes: int - Minimo corsie per considerare un TLS "rilevante"
        min_green: int - Tempo minimo di verde in secondi
        alpha: float - Peso del reward locale vs globale (0.0 = solo globale, 1.0 = solo locale)
        neighbor_radius: float - Raggio in metri per determinare i vicini
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        use_gui=False,
        max_steps=3600,
        top_n_tls=10,
        min_controlled_lanes=3,
        min_green=5,
        alpha=0.7,
        neighbor_radius=300.0,
    ):
        super(MultiAgentSumoEnv, self).__init__()
        
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.top_n_tls = top_n_tls
        self.min_controlled_lanes = min_controlled_lanes
        self.min_green = min_green
        self.alpha = alpha
        self.neighbor_radius = neighbor_radius
        self.step_counter = 0
        
        # Saranno popolati al reset()
        self.agent_ids = []          # Lista di TLS IDs gestiti
        self.agent_lanes = {}        # {tls_id: [lista corsie]}
        self.agent_num_phases = {}   # {tls_id: num_fasi}
        self.agent_phase_dur = {}    # {tls_id: durata_fase_corrente}
        self.agent_neighbors = {}    # {tls_id: [lista tls_id vicini]}
        self.agent_positions = {}    # {tls_id: (x, y)}
        
        self.n_agents = 0
        
        # Dimensioni osservazione per agente:
        # [queue_0..3, total_halting, total_wait, phase, phase_dur, avg_speed, density, hour,
        #  neighbor_queue_0, neighbor_queue_1, neighbor_queue_2, neighbor_queue_3]  = 15
        self.obs_dim = 15
        self.act_dim = 2  # Keep / Switch
        
        # Spazi (saranno definiti per agente, ma usiamo un singolo spazio condiviso)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.act_dim)
        
        sumo_binary_name = "sumo-gui.exe" if self.use_gui else "sumo.exe"
        sumo_binary_path = os.path.join(SUMO_HOME, "bin", sumo_binary_name)
        
        self.sumo_cmd = [
            sumo_binary_path,
            "-c", SUMO_CONFIG,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--start",
        ]
        
        self._initialized = False
        
    @property
    def num_agents(self):
        return self.n_agents

    def _discover_agents(self):
        """Scopre i semafori rilevanti nella rete e li configura come agenti."""
        all_tls = traci.trafficlight.getIDList()
        
        # Filtra per numero minimo di corsie controllate
        tls_data = []
        for tls_id in all_tls:
            lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
            if len(lanes) >= self.min_controlled_lanes:
                tls_data.append((tls_id, lanes))
        
        # Ordina per numero di corsie (i più importanti prima)
        tls_data.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Prendi i top N
        selected = tls_data[:self.top_n_tls]
        
        self.agent_ids = []
        self.agent_lanes = {}
        self.agent_num_phases = {}
        self.agent_phase_dur = {}
        
        for tls_id, lanes in selected:
            self.agent_ids.append(tls_id)
            self.agent_lanes[tls_id] = lanes
            self.agent_phase_dur[tls_id] = 0
            
            # Numero di fasi
            logic = traci.trafficlight.getAllProgramLogics(tls_id)
            if logic:
                self.agent_num_phases[tls_id] = len(logic[0].phases)
            else:
                self.agent_num_phases[tls_id] = 4
            
            # Inizializza fase
            traci.trafficlight.setPhase(tls_id, 0)
        
        self.n_agents = len(self.agent_ids)
        
        # Determina posizioni dei semafori per il calcolo dei vicini
        self._compute_positions_and_neighbors()
        
        print(f"\n{'='*60}")
        print(f"🚦 MULTI-AGENT SETUP COMPLETO")
        print(f"{'='*60}")
        print(f"  Agenti attivi: {self.n_agents}")
        for i, tls_id in enumerate(self.agent_ids):
            n_lanes = len(self.agent_lanes[tls_id])
            n_phases = self.agent_num_phases[tls_id]
            n_neighbors = len(self.agent_neighbors.get(tls_id, []))
            short_id = tls_id[:40] + "..." if len(tls_id) > 40 else tls_id
            print(f"  Agent {i}: {short_id} | {n_lanes} corsie | {n_phases} fasi | {n_neighbors} vicini")
        print(f"{'='*60}\n")

    def _compute_positions_and_neighbors(self):
        """Calcola posizioni dei TLS e determina i vicini entro il raggio."""
        net = sumolib.net.readNet(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_config", "pisa.net.xml"),
            withPrograms=True
        )
        
        self.agent_positions = {}
        for tls_id in self.agent_ids:
            # Prendi le coordinate dal centroide delle corsie controllate
            coords = []
            for lane_id in self.agent_lanes[tls_id]:
                try:
                    lane = net.getLane(lane_id)
                    shape = lane.getShape()
                    # Usa il punto finale della corsia (più vicino all'incrocio)
                    coords.append(shape[-1])
                except:
                    pass
            
            if coords:
                x = np.mean([c[0] for c in coords])
                y = np.mean([c[1] for c in coords])
                self.agent_positions[tls_id] = (x, y)
            else:
                self.agent_positions[tls_id] = (0, 0)
        
        # Calcola vicini basandosi sulla distanza
        self.agent_neighbors = {}
        for tls_id in self.agent_ids:
            pos = self.agent_positions[tls_id]
            neighbors = []
            for other_id in self.agent_ids:
                if other_id == tls_id:
                    continue
                other_pos = self.agent_positions[other_id]
                dist = np.sqrt((pos[0] - other_pos[0])**2 + (pos[1] - other_pos[1])**2)
                if dist <= self.neighbor_radius:
                    neighbors.append(other_id)
            self.agent_neighbors[tls_id] = neighbors

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        try:
            traci.close()
        except:
            pass
        
        traci.start(self.sumo_cmd)
        self.step_counter = 0
        
        # Scopri agenti
        self._discover_agents()
        self._initialized = True
        
        # Ritorna osservazioni per tutti gli agenti
        observations = self.get_all_observations()
        return observations, {}

    def step(self, actions):
        """
        Esegue un passo per tutti gli agenti contemporaneamente.
        
        Args:
            actions: dict {tls_id: action} oppure list di azioni (ordinate come agent_ids)
        
        Returns:
            observations: dict {tls_id: obs}
            rewards: dict {tls_id: reward}
            terminated: bool
            truncated: bool
            infos: dict {tls_id: info}
        """
        # Converti lista in dizionario se necessario
        if isinstance(actions, (list, np.ndarray)):
            action_dict = {tls_id: int(actions[i]) for i, tls_id in enumerate(self.agent_ids)}
        else:
            action_dict = actions
        
        # Applica azioni per ogni agente
        for tls_id in self.agent_ids:
            action = action_dict.get(tls_id, 0)
            
            if action == 1 and self.agent_phase_dur[tls_id] > self.min_green:
                current = traci.trafficlight.getPhase(tls_id)
                next_phase = (current + 1) % self.agent_num_phases[tls_id]
                traci.trafficlight.setPhase(tls_id, next_phase)
                self.agent_phase_dur[tls_id] = 0
            else:
                self.agent_phase_dur[tls_id] += 1
        
        # Avanza simulazione
        traci.simulationStep()
        self.step_counter += 1
        
        # Raccogli risultati
        observations = self.get_all_observations()
        rewards = self._compute_all_rewards()
        terminated = self.step_counter >= self.max_steps
        truncated = False
        
        infos = {tls_id: {} for tls_id in self.agent_ids}
        
        return observations, rewards, terminated, truncated, infos

    def get_all_observations(self):
        """Ritorna le osservazioni per tutti gli agenti."""
        observations = {}
        for tls_id in self.agent_ids:
            observations[tls_id] = self._get_agent_observation(tls_id)
        return observations

    def _get_agent_observation(self, tls_id):
        """Calcola l'osservazione per un singolo agente."""
        lanes = self.agent_lanes[tls_id]
        
        # Code per direzione
        queues = []
        for lane in lanes:
            try:
                queues.append(traci.lane.getLastStepHaltingNumber(lane))
            except:
                queues.append(0)
        
        # Dividi in 4 direzioni
        n_groups = min(4, len(queues))
        group_size = max(1, len(queues) // n_groups) if n_groups > 0 else 1
        dir_queues = [0.0] * 4
        for i in range(min(4, n_groups)):
            start = i * group_size
            end = start + group_size if i < 3 else len(queues)
            dir_queues[i] = sum(queues[start:end])
        
        total_halting = sum(queues)
        total_wt = self._get_agent_waiting_time(tls_id)
        
        # Velocità media
        speeds = []
        for lane in lanes:
            try:
                speeds.append(traci.lane.getLastStepMeanSpeed(lane))
            except:
                speeds.append(0)
        avg_speed = np.mean(speeds) if speeds else 0
        
        # Densità
        total_veh = 0
        for lane in lanes:
            try:
                total_veh += traci.lane.getLastStepVehicleNumber(lane)
            except:
                pass
        density = total_veh / max(1, len(lanes))
        
        # Fase
        try:
            current_phase = traci.trafficlight.getPhase(tls_id)
        except:
            current_phase = 0
        
        # Informazioni dai vicini (code aggregate)
        neighbor_queues = [0.0] * 4
        neighbors = self.agent_neighbors.get(tls_id, [])
        for i, neighbor_id in enumerate(neighbors[:4]):
            try:
                n_lanes = self.agent_lanes.get(neighbor_id, [])
                n_halting = sum(traci.lane.getLastStepHaltingNumber(l) for l in n_lanes)
                neighbor_queues[i] = n_halting
            except:
                pass
        
        obs = np.array([
            # Stato locale (11 dim)
            min(dir_queues[0] / 20.0, 1.0),
            min(dir_queues[1] / 20.0, 1.0),
            min(dir_queues[2] / 20.0, 1.0),
            min(dir_queues[3] / 20.0, 1.0),
            min(total_halting / 50.0, 1.0),
            min(total_wt / 1000.0, 1.0),
            current_phase / max(1, self.agent_num_phases.get(tls_id, 4) - 1),
            min(self.agent_phase_dur.get(tls_id, 0) / 60.0, 1.0),
            min(avg_speed / 15.0, 1.0),
            min(density / 10.0, 1.0),
            (self.step_counter / 3600.0) % 1.0,
            # Info dai vicini (4 dim)
            min(neighbor_queues[0] / 50.0, 1.0),
            min(neighbor_queues[1] / 50.0, 1.0),
            min(neighbor_queues[2] / 50.0, 1.0),
            min(neighbor_queues[3] / 50.0, 1.0),
        ], dtype=np.float32)
        
        return obs

    def _get_agent_waiting_time(self, tls_id):
        """Tempo di attesa totale per un agente."""
        total_wt = 0
        for lane in self.agent_lanes.get(tls_id, []):
            try:
                total_wt += traci.lane.getWaitingTime(lane)
            except:
                pass
        return total_wt

    def _compute_all_rewards(self):
        """Calcola i reward per tutti gli agenti con componente locale+globale."""
        local_rewards = {}
        for tls_id in self.agent_ids:
            wt = self._get_agent_waiting_time(tls_id)
            q = sum(traci.lane.getLastStepHaltingNumber(l) for l in self.agent_lanes[tls_id])
            local_rewards[tls_id] = -(wt + 10 * q) / 1000.0
        
        # Reward globale = media dei reward locali
        global_reward = np.mean(list(local_rewards.values())) if local_rewards else 0
        
        # Reward finale = α * locale + (1-α) * globale
        rewards = {}
        for tls_id in self.agent_ids:
            rewards[tls_id] = self.alpha * local_rewards[tls_id] + (1 - self.alpha) * global_reward
        
        return rewards

    def get_global_metrics(self):
        """Metriche globali per benchmarking."""
        total_waiting = 0
        total_halting = 0
        total_vehicles = 0
        total_speed = 0
        n_lanes = 0
        
        for tls_id in self.agent_ids:
            for lane in self.agent_lanes[tls_id]:
                try:
                    total_waiting += traci.lane.getWaitingTime(lane)
                    total_halting += traci.lane.getLastStepHaltingNumber(lane)
                    total_vehicles += traci.lane.getLastStepVehicleNumber(lane)
                    total_speed += traci.lane.getLastStepMeanSpeed(lane)
                    n_lanes += 1
                except:
                    pass
        
        return {
            "total_waiting_time": total_waiting,
            "total_halting": total_halting,
            "total_vehicles": total_vehicles,
            "avg_speed": total_speed / max(1, n_lanes),
            "n_agents": self.n_agents,
            "step": self.step_counter,
        }

    def close(self):
        try:
            traci.close()
        except:
            pass


class MultiAgentWrapper:
    """
    Wrapper che espone l'ambiente multi-agent come N ambienti single-agent
    per compatibilità con algoritmi standard come PPO di SB3.
    
    Utilizzo con IPPO (Independent PPO):
        env = MultiAgentSumoEnv(top_n_tls=5)
        wrapper = MultiAgentWrapper(env)
        
        # Training loop
        obs_dict, _ = wrapper.reset()
        for step in range(max_steps):
            actions = {}
            for agent_id in wrapper.agent_ids:
                action = policy.predict(obs_dict[agent_id])
                actions[agent_id] = action
            obs_dict, rewards, done, truncated, infos = wrapper.step(actions)
    """
    
    def __init__(self, env: MultiAgentSumoEnv):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    @property
    def agent_ids(self):
        return self.env.agent_ids
    
    @property
    def num_agents(self):
        return self.env.num_agents
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, actions):
        return self.env.step(actions)
    
    def close(self):
        self.env.close()


# ============================================================
# TEST / DEMO
# ============================================================
if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("🚦 SmartLight Pisa - Multi-Agent Demo")
    print("=" * 60)
    
    # Configura l'ambiente
    env = MultiAgentSumoEnv(
        use_gui=True,         # Mostra la GUI di SUMO
        max_steps=1000,       # 1000 step per la demo
        top_n_tls=5,          # Controlla i 5 semafori più importanti
        min_controlled_lanes=8,
        alpha=0.7,            # 70% reward locale, 30% globale
        neighbor_radius=300,  # Vicini entro 300m
    )
    
    print("\nAvvio simulazione multi-agent...")
    obs_dict, _ = env.reset()
    
    total_rewards = {agent_id: 0 for agent_id in env.agent_ids}
    
    for step in range(1000):
        # Azioni casuali per tutti gli agenti (sarà sostituito dal modello RL)
        actions = {agent_id: env.action_space.sample() for agent_id in env.agent_ids}
        
        obs_dict, rewards, done, truncated, infos = env.step(actions)
        
        for agent_id in env.agent_ids:
            total_rewards[agent_id] += rewards[agent_id]
        
        if step % 100 == 0:
            metrics = env.get_global_metrics()
            print(f"\n📊 Step {step}:")
            print(f"   Total Waiting: {metrics['total_waiting_time']:.1f}s")
            print(f"   Total Halting: {metrics['total_halting']}")
            print(f"   Avg Speed: {metrics['avg_speed']:.2f} m/s")
            print(f"   Vehicles on monitored lanes: {metrics['total_vehicles']}")
    
    print("\n" + "=" * 60)
    print("📊 RISULTATI FINALI")
    print("=" * 60)
    for i, agent_id in enumerate(env.agent_ids):
        short_id = agent_id[:40] + "..." if len(agent_id) > 40 else agent_id
        print(f"  Agent {i} ({short_id}): Total Reward = {total_rewards[agent_id]:.2f}")
    
    final_metrics = env.get_global_metrics()
    print(f"\n  🌐 Metriche Globali Finali:")
    print(f"     Total Waiting Time: {final_metrics['total_waiting_time']:.1f}s")
    print(f"     Total Halting Vehicles: {final_metrics['total_halting']}")
    print(f"     Average Speed: {final_metrics['avg_speed']:.2f} m/s")
    print("=" * 60)
    
    print("\nPremi INVIO per chiudere...")
    input()
    env.close()
