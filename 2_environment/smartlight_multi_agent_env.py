"""
SmartLight Pisa - Multi-Agent SUMO Environment (EDGE-FIRST / MAX PRESSURE)
==========================================================================
Ambiente multi-agente per SUMO con logica decentralizzata.
Ogni semaforo agisce indipendentemente, ma osservando lo spazio libero a valle (1-hop).
Reward basata sul teorema "Max Pressure" per evitare lo spillback.
"""

import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
try:
    import libsumo
except ImportError:
    libsumo = None
import sumolib

# Configurazione Paths SUMO
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_HOME = os.path.join(PROJECT_ROOT, "sumo_tools", "sumo-1.22.0")
SUMO_CONFIG = os.path.join(PROJECT_ROOT, "1_simulation", "sumo_config", "pisa.sumocfg")

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME
sumo_bin_dir = os.path.join(SUMO_HOME, 'bin')
if sumo_bin_dir not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + sumo_bin_dir
sys.path.append(os.path.join(SUMO_HOME, 'tools'))


class MultiAgentSumoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        use_gui=False,
        max_steps=3600,
        top_n_tls=10,
        min_controlled_lanes=3,
        min_green=10,      # GUARDRAIL: Verde minimo garantito
        max_green=60,      # GUARDRAIL: Previene il ROSSO INFINTIO sulle altre direttrici
    ):
        super(MultiAgentSumoEnv, self).__init__()
        
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.top_n_tls = top_n_tls
        self.min_controlled_lanes = min_controlled_lanes
        self.min_green = min_green
        self.max_green = max_green
        self.step_counter = 0
        
        # Strutture Dati Agenti
        self.agent_ids = []
        self.agent_incoming = {}   # {tls_id: [lanes ingresso]}
        self.agent_outgoing = {}   # {tls_id: [lanes uscita]}
        self.agent_links = {}      # {tls_id: [(in_lane, out_lane)] mappate sulla stringa semaforo}
        self.agent_num_phases = {} 
        self.agent_phase_dur = {}  
        
        self.n_agents = 0
        
        # OSSERVAZIONE: 11 Dimensioni complessive ("Local + 1-Hop Downstream")
        # 4x Code ingresso | 1x Densità | 4x Spazio Libero Downstream | 1x Fase Corrente | 1x Durata Fase
        self.obs_dim = 11
        self.act_dim = 2  # 0=Keep, 1=Switch
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.act_dim)
        
        sumo_binary_name = "sumo-gui.exe" if self.use_gui else "sumo.exe"
        self.sumo_cmd = [
            os.path.join(sumo_bin_dir, sumo_binary_name),
            "-c", SUMO_CONFIG,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--start",
        ]
        self._using_libsumo = False
        self._initialized = False
        
    @property
    def num_agents(self):
        return self.n_agents

    def _discover_agents(self):
        """Mappa la rete e scopre la topologia 1-hop locale."""
        all_tls = self.traci.trafficlight.getIDList()
        
        tls_data = []
        for tls_id in all_tls:
            lanes = list(set(self.traci.trafficlight.getControlledLanes(tls_id)))
            if len(lanes) >= self.min_controlled_lanes:
                tls_data.append((tls_id, lanes))
                
        tls_data.sort(key=lambda x: len(x[1]), reverse=True)
        selected = tls_data[:self.top_n_tls]
        
        self.agent_ids = [s[0] for s in selected]
        self.n_agents = len(self.agent_ids)
        
        for tls_id in self.agent_ids:
            self.agent_phase_dur[tls_id] = 0
            
            logic = self.traci.trafficlight.getAllProgramLogics(tls_id)
            self.agent_num_phases[tls_id] = len(logic[0].phases) if logic else 4
            self.traci.trafficlight.setPhase(tls_id, 0)
            
            # Estrazione Mappa Topologica (Incoming -> Outgoing)
            links = self.traci.trafficlight.getControlledLinks(tls_id)
            tls_links = []
            for link_group in links:
                if link_group:
                    in_lane = link_group[0][0]
                    out_lane = link_group[0][1]
                    tls_links.append((in_lane, out_lane))
                else:
                    tls_links.append((None, None))
                    
            self.agent_links[tls_id] = tls_links
            self.agent_incoming[tls_id] = list(set(l[0] for l in tls_links if l[0]))
            self.agent_outgoing[tls_id] = list(set(l[1] for l in tls_links if l[1]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            if hasattr(self, 'traci') and self.traci is not None:
                self.close()
        except:
            pass
        
        if self.use_gui or libsumo is None:
            self.traci = traci
            self.traci.start(self.sumo_cmd)
            self._using_libsumo = False
        else:
            self.traci = libsumo
            self.traci.start(self.sumo_cmd[1:])
            self._using_libsumo = True
            
        self.step_counter = 0
        self._discover_agents()
        self._initialized = True
        
        return self.get_all_observations(), {}

    def step(self, actions):
        """Avanzamento con GUARDRAILS deterministici."""
        # actions = dict {tls_id: int_action}
        if isinstance(actions, (list, np.ndarray)):
            actions = {tls_id: int(actions[i]) for i, tls_id in enumerate(self.agent_ids)}
            
        for tls_id in self.agent_ids:
            action = actions.get(tls_id, 0)
            state_str = self.traci.trafficlight.getRedYellowGreenState(tls_id)
            current_phase = self.traci.trafficlight.getPhase(tls_id)
            
            # GUARDRAIL 1: Il giallo è intoccabile dall'AI
            if 'y' in state_str.lower():
                self.agent_phase_dur[tls_id] += 1
                continue
            
            force_switch = False
            # GUARDRAIL 2: Stop infinito (Rosso Max) -> Verde Max per la fase corrente
            if self.agent_phase_dur[tls_id] >= self.max_green:
                force_switch = True
                
            want_switch = (action == 1)
            # GUARDRAIL 3: Tempo minimo garantito
            can_switch = (self.agent_phase_dur[tls_id] >= self.min_green)
            
            if force_switch or (want_switch and can_switch):
                next_phase = (current_phase + 1) % self.agent_num_phases[tls_id]
                self.traci.trafficlight.setPhase(tls_id, next_phase)
                self.agent_phase_dur[tls_id] = 0
            else:
                self.agent_phase_dur[tls_id] += 1
                
        # Advance simulation
        if self._using_libsumo:
            libsumo.simulationStep()
        else:
            self.traci.simulationStep()
        self.step_counter += 1
        
        observations = self.get_all_observations()
        rewards = self._compute_max_pressure_rewards()
        terminated = self.step_counter >= self.max_steps
        
        infos = {tls_id: {} for tls_id in self.agent_ids}
        return observations, rewards, terminated, False, infos

    def get_all_observations(self):
        obs = {}
        for tls_id in self.agent_ids:
            obs[tls_id] = self._get_agent_observation(tls_id)
        return obs

    def _get_agent_observation(self, tls_id):
        """Observation Space focalizzata sulla Capacity Downstream."""
        in_lanes = self.agent_incoming[tls_id]
        out_lanes = self.agent_outgoing[tls_id]
        
        # 1. Code in Ingresso
        queues = []
        for lane in in_lanes:
            try: queues.append(self.traci.lane.getLastStepHaltingNumber(lane))
            except: queues.append(0)
            
        # 2. Spazio libero in Uscita (Downstream Capacity = 1-Hop Neighbor data)
        free_spaces = []
        for lane in out_lanes:
            try:
                cap = self.traci.lane.getLength(lane) / 5.0 # Max auto (5m per auto)
                veh = self.traci.lane.getLastStepVehicleNumber(lane)
                free_spaces.append(max(0, cap - veh))
            except:
                free_spaces.append(0)
                
        # Aggregate to 4 bins
        def bin_data(data, bins=4):
            n = len(data)
            if n == 0: return [0.0]*bins
            group_size = max(1, n // bins)
            res = [0.0]*bins
            for i in range(min(bins, n)):
                start = i * group_size
                end = start + group_size if i < bins-1 else n
                res[i] = sum(data[start:end])
            return res

        dir_queues = bin_data(queues, 4)
        dir_free_space = bin_data(free_spaces, 4)
        
        # Densità locale aggregates
        total_veh = sum([self.traci.lane.getLastStepVehicleNumber(l) for l in in_lanes])
        density = total_veh / max(1, len(in_lanes))
        
        current_phase = self.traci.trafficlight.getPhase(tls_id)
        
        obs = np.array([
            min(dir_queues[0] / 20.0, 1.0),
            min(dir_queues[1] / 20.0, 1.0),
            min(dir_queues[2] / 20.0, 1.0),
            min(dir_queues[3] / 20.0, 1.0),
            min(density / 10.0, 1.0),
            min(dir_free_space[0] / 30.0, 1.0),
            min(dir_free_space[1] / 30.0, 1.0),
            min(dir_free_space[2] / 30.0, 1.0),
            min(dir_free_space[3] / 30.0, 1.0),
            current_phase / max(1, self.agent_num_phases[tls_id] - 1),
            min(self.agent_phase_dur[tls_id] / float(self.max_green), 1.0)
        ], dtype=np.float32)
        
        return obs

    def _compute_max_pressure_rewards(self):
        """
        REWARD FUNCTION: Basata sul Teorema della "Max Pressure".
        Penalizza l'agente SE sta dando verde verso una corsia che è già piena.
        """
        rewards = {}
        for tls_id in self.agent_ids:
            state_str = self.traci.trafficlight.getRedYellowGreenState(tls_id)
            links = self.agent_links[tls_id]
            
            pressure = 0.0
            
            # Calcolo Pressione solo per le direzioni che l'agente sta forzando a fluire (Verdi)
            for i, char in enumerate(state_str):
                if char.lower() == 'g':
                    in_l, out_l = links[i]
                    if in_l and out_l:
                        try:
                            # Auto che vogliono passare
                            n_in = self.traci.lane.getLastStepHaltingNumber(in_l)
                            
                            # Spazio che c'è davvero dopo l'incrocio (Capacità - Occupazione)
                            out_cap = self.traci.lane.getLength(out_l) / 5.0
                            out_veh = self.traci.lane.getLastStepVehicleNumber(out_l)
                            free_space = max(0, out_cap - out_veh)
                            
                            # Max Pressure formula variant
                            p_mov = n_in - free_space
                            if p_mov > 0:
                                pressure += p_mov # Sto creando uno spillback (ingorgo)
                        except:
                            pass
                            
            # La penalità base è la pressione creata. Zero pressione = zero penalità.
            rewards[tls_id] = -(pressure / 10.0)
            
        return rewards

    def get_global_metrics(self):
        total_waiting = 0
        total_halting = 0
        total_speed = 0
        n_lanes = 0
        
        for tls_id in self.agent_ids:
            for lane in self.agent_incoming[tls_id]:
                try:
                    total_waiting += self.traci.lane.getWaitingTime(lane)
                    total_halting += self.traci.lane.getLastStepHaltingNumber(lane)
                    total_speed += self.traci.lane.getLastStepMeanSpeed(lane)
                    n_lanes += 1
                except:
                    pass
                    
        return {
            "total_waiting_time": total_waiting,
            "total_halting": total_halting,
            "avg_speed": total_speed / max(1, n_lanes) if n_lanes > 0 else 0,
            "n_agents": self.n_agents,
        }

    def close(self):
        try:
            if self._using_libsumo:
                libsumo.close()
            else:
                self.traci.close()
        except:
            pass
