import traci
import numpy as np
import random

class PisaTrafficGenerator:
    def __init__(self, net_file):
        """
        Generatore di traffico dinamico per la mappa di Pisa.
        Invece di usare file .rou.xml statici, genera flussi probabilitstici.
        """
        self.net_file = net_file
        # Definizione dei flussi principali (questi andrebbero mappati sugli Edge ID reali di Pisa)
        # Per ora useremo l'auto-detection nel training loop per trovare gli edge in ingresso
        self.incoming_edges = [] 
        
    def detect_incoming_edges(self, tls_id):
        """Rileva automaticamente le strade che entrano nell'incrocio"""
        valid_edges = []
        # Ottieni le corsie controllate dal semaforo
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        # Estrai gli edge unici
        edges = sorted(list(set([traci.lane.getEdgeID(lane) for lane in lanes])))
        print(f"[Generator] Rilevati {len(edges)} edge in ingresso al semaforo {tls_id}: {edges}")
        self.incoming_edges = edges
        return edges

    def generate_route_file(self, episode_seed, scenario_name="balanced"):
        """
        Genera un file routes .xml temporaneo per l'episodio corrente.
        Questo assicura che ogni episodio sia unico ma coerente con lo scenario.
        """
        np.random.seed(episode_seed)
        random.seed(episode_seed)
        
        # Configurazione Scenario
        if scenario_name == "morning_rush":
            # Traffico intenso, sbilanciato (es. pendolari in entrata)
            period = 1.5  # secondi tra veicoli (alta densità)
            bias_weights = [0.6, 0.2, 0.1, 0.1] # Molto traffico sul primo edge
        elif scenario_name == "evening_rush":
            period = 2.0
            bias_weights = [0.1, 0.5, 0.3, 0.1] # Traffico su altri edge
        elif scenario_name == "night":
            period = 8.0 # Bassa densità
            bias_weights = [0.25, 0.25, 0.25, 0.25] # Bilanciato
        else: # Balanced
            period = 4.0
            bias_weights = None # Uniforme

        N_STEPS = 3600 # 1 ora simulata
        
        # Se non abbiamo edge rilevati, non possiamo generare rotte valide
        # Restituiamo None, il training dovrà fare fallback o rilevare prima
        if not self.incoming_edges:
            return None

        # Distribuzione pesi sugli edge
        if bias_weights is None or len(bias_weights) != len(self.incoming_edges):
            bias_weights = [1.0 / len(self.incoming_edges)] * len(self.incoming_edges)
        
        # Generazione XML
        filename = f"dynamic_routes_{episode_seed}.rou.xml"
        with open(filename, "w") as routes:
            routes.write("""<routes>
            <vType id="car" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="13.89" color="1,1,0"/>
            <vType id="bus" vClass="bus" accel="1.2" decel="4.0" sigma="0.3" length="12.0" minGap="3.0" maxSpeed="11.11" color="1,0,0"/>
            """)
            
            veh_id = 0
            for step in range(0, N_STEPS):
                # Probabilità di spawn basata sul periodo (distribuzione di Weibull per realismo o esponenziale)
                if np.random.uniform(0, 1) < (1.0 / period):
                    # Scegli edge di partenza in base ai pesi dello scenario
                    edge_idx = np.random.choice(len(self.incoming_edges), p=bias_weights)
                    start_edge = self.incoming_edges[edge_idx]
                    
                    # Per ora destinazione random (in una griglia reale sarebbe meglio calcolarla)
                    # In SUMO senza route file completo, le auto svaniscono alla fine dell'edge o serve trip
                    # Usiamo flussi semplici: entra ed esci
                    
                    vtype = "bus" if np.random.uniform(0,1) < 0.05 else "car"
                    
                    routes.write(f'    <trip id="{veh_id}" type="{vtype}" depart="{step}" from="{start_edge}" />\n')
                    veh_id += 1
            
            routes.write("</routes>")
        
        return filename
