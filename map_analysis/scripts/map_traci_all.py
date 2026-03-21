import os
import sys
import traci

# Configurazione
SUMO_HOME = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_tools", "sumo-1.22.0")
SUMO_CONFIG = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_config", "pisa.sumocfg")

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

sumo_bin = os.path.join(SUMO_HOME, "bin", "sumo.exe")
cmd = [sumo_bin, "-c", SUMO_CONFIG, "--no-step-log", "true", "--start"]

def map_all_traci_tls():
    print("Avvio SUMO per mappatura completa...")
    traci.start(cmd)
    
    tls_ids = traci.trafficlight.getIDList()
    print(f"TraCI ha trovato {len(tls_ids)} programmi semaforici.")
    
    output_path = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "mappa_traci_completa.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"MAPPATURA COMPLETA SEMAFORI (LIVE TRACI) - PISA\n")
        f.write(f"Totale programmi: {len(tls_ids)}\n")
        f.write("=" * 70 + "\n\n")
        
        for tls_id in tls_ids:
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            street_names = set()
            for lane in lanes:
                edge_id = traci.lane.getEdgeID(lane)
                # In SUMO, gli edge possono avere nomi. Cerchiamo di ottenerlo se possibile.
                # Nota: TraCI non dà direttamente il nome della via dall'edge ID facilmente senza sumolib,
                # ma l'ID dell'edge spesso contiene informazioni o possiamo cross-referenziarlo.
                # Tuttavia, gli edge ID di OSM sono numerici.
                street_names.add(edge_id)
            
            f.write(f"ID PROGRAMMA: {tls_id}\n")
            f.write(f"EDGES COINVOLTI: {', '.join(sorted(list(street_names)))}\n")
            f.write("-" * 40 + "\n")
            
    print(f"Mappatura completata. Risultato salvato in: {output_path}")
    traci.close()

if __name__ == "__main__":
    map_all_traci_tls()
