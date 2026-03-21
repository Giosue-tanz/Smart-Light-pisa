"""Benchmark per calcolare la velocità media base (default SUMO) senza RL."""
import sys
import os
import traci

SUMO_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_tools", "sumo-1.22.0")
SUMO_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_config", "pisa.sumocfg")

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME

sys.path.append(os.path.join(SUMO_HOME, 'tools'))
sumo_bin = os.path.join(SUMO_HOME, "bin", "sumo.exe")

def run_benchmark(steps=1800):
    cmd = [sumo_bin, "-c", SUMO_CONFIG, "--no-step-log", "true", "--start"]
    traci.start(cmd)
    
    total_speed = 0
    total_vehicles = 0
    step_count = 0
    
    print(f"Esecuzione benchmark standard (senza RL) per {steps} step...")
    
    for _ in range(steps):
        traci.simulationStep()
        
        # Misure globali
        veh_ids = traci.vehicle.getIDList()
        if veh_ids:
            speeds = [traci.vehicle.getSpeed(v) for v in veh_ids]
            total_speed += sum(speeds)
            total_vehicles += len(veh_ids)
        
        step_count += 1
    
    avg_speed = total_speed / total_vehicles if total_vehicles > 0 else 0
    traci.close()
    return avg_speed

if __name__ == "__main__":
    avg_s = run_benchmark(1800)
    print(f"\n[RISULTATO BENCHMARK BASELINE]")
    print(f"Velocità media standard (tempi default): {avg_s:.2f} m/s")
