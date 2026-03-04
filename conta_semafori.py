import os
import sys
import traci

# Configurazione - percorsi relativi allo script
SUMO_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_tools", "sumo-1.22.0")
SUMO_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_config", "pisa.sumocfg")

# Setup Ambiente
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

sumo_bin_dir = os.path.join(SUMO_HOME, 'bin')
if sumo_bin_dir not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + sumo_bin_dir

sumo_bin = os.path.join(SUMO_HOME, "bin", "sumo.exe")  # Headless
cmd = [sumo_bin, "-c", SUMO_CONFIG, "--no-step-log", "true", "--start"]

print("Avvio SUMO per conteggio semafori...")
try:
    traci.start(cmd)
    tls_list = traci.trafficlight.getIDList()
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_semafori.txt")
    with open(report_path, "w") as f:
        f.write(f"Totale Semafori Trovati: {len(tls_list)}\n")
        f.write("--------------------------------\n")
        tls_data = []
        for tls in tls_list:
            lanes = traci.trafficlight.getControlledLanes(tls)
            tls_data.append((tls, len(lanes)))
        
        tls_data.sort(key=lambda x: x[1], reverse=True)
        
        for tls, num_lanes in tls_data:
            f.write(f"ID: {tls} | Corsie controllate: {num_lanes}\n")
            
    print(f"Report salvato in {report_path}")
    traci.close()
except Exception as e:
    print(f"Errore: {e}")
