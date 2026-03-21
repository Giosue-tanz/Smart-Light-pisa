"""
Genera scenario medio realistico per Pisa con tutti i tipi di veicoli.
Approccio: genera trip per auto (passenger), poi assegna tipi diversi.
"""
import subprocess
import os
import sys
import xml.etree.ElementTree as ET
import random
import argparse

SUMO_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_tools", "sumo-1.22.0")
NET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_config", "pisa.net.xml")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_config")
RANDOM_TRIPS = os.path.join(SUMO_HOME, "tools", "randomTrips.py")
DURATION = 3600

# Tipi di veicolo con le loro proporzioni e parametri
VEHICLE_TYPES = [
    {
        "id": "car", "vClass": "passenger", "guiShape": "passenger",
        "color": "0.4,0.6,1.0", "maxSpeed": "13.89", "accel": "2.6",
        "decel": "4.5", "length": "4.5", "width": "1.8", "minGap": "2.5",
        "sigma": "0.5", "proportion": 0.48,  # 48%
    },
    {
        "id": "truck", "vClass": "passenger", "guiShape": "truck",
        "color": "0.8,0.4,0.1", "maxSpeed": "11.11", "accel": "1.3",
        "decel": "4.0", "length": "10.0", "width": "2.5", "minGap": "3.0",
        "sigma": "0.4", "proportion": 0.05,  # 5%
    },
    {
        "id": "bus", "vClass": "passenger", "guiShape": "bus",
        "color": "0.1,0.8,0.2", "maxSpeed": "11.11", "accel": "1.2",
        "decel": "4.0", "length": "12.0", "width": "2.5", "minGap": "3.0",
        "sigma": "0.3", "proportion": 0.03,  # 3%
    },
    {
        "id": "motorcycle", "vClass": "passenger", "guiShape": "motorcycle",
        "color": "1.0,0.2,0.2", "maxSpeed": "16.67", "accel": "4.0",
        "decel": "6.0", "length": "2.2", "width": "0.8", "minGap": "1.5",
        "sigma": "0.6", "proportion": 0.07,  # 7%
    },
    {
        "id": "bicycle", "vClass": "passenger", "guiShape": "bicycle",
        "color": "1.0,1.0,0.0", "maxSpeed": "5.56", "accel": "1.2",
        "decel": "3.0", "length": "1.8", "width": "0.6", "minGap": "1.0",
        "sigma": "0.5", "proportion": 0.12,  # 12%
    },
    {
        "id": "van", "vClass": "passenger", "guiShape": "delivery",
        "color": "0.6,0.3,0.8", "maxSpeed": "12.50", "accel": "2.0",
        "decel": "4.0", "length": "5.5", "width": "2.0", "minGap": "2.5",
        "sigma": "0.4", "proportion": 0.05,  # 5%
    },
]

# Il restante 20% sarà auto di colori diversi per varietà
EXTRA_CAR_COLORS = [
    "0.9,0.9,0.9",  # Bianco
    "0.2,0.2,0.2",  # Nero
    "0.8,0.1,0.1",  # Rosso scuro
    "0.6,0.6,0.6",  # Grigio
]

TOTAL_VEHICLES = 625  # Veicoli motorizzati
PEDESTRIAN_COUNT = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args, _ = parser.parse_known_args()
    
    seed = args.seed
    random.seed(seed)
    
    if not args.quiet:
        print("=" * 55)
        print(f"🚦 GENERATORE SCENARIO MEDIO - Pisa Città Completa (Seed: {seed})")
        print("=" * 55)
    
    # ===== STEP 1: Genera trip base con randomTrips =====
    print("\n📍 Step 1: Generazione trip base...")
    trip_file = os.path.join(OUT_DIR, "temp_trips.trips.xml")
    route_file = os.path.join(OUT_DIR, "temp_routes.rou.xml")
    
    period = DURATION / TOTAL_VEHICLES
    cmd = [
        sys.executable, RANDOM_TRIPS,
        "-n", NET_FILE,
        "-o", trip_file,
        "-r", route_file,
        "-e", str(DURATION),
        "-p", str(period),
        "--fringe-factor", "5",
        "--validate",
        "--seed", str(seed),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Errore: {result.stderr[:300]}")
        return
    print(f"  ✅ Generati {TOTAL_VEHICLES} trip base")
    
    # ===== STEP 2: Genera pedoni =====
    print("\n🚶 Step 2: Generazione pedoni...")
    ped_trip_file = os.path.join(OUT_DIR, "temp_ped_trips.trips.xml")
    ped_route_file = os.path.join(OUT_DIR, "temp_ped_routes.rou.xml")
    
    ped_period = DURATION / PEDESTRIAN_COUNT
    cmd_ped = [
        sys.executable, RANDOM_TRIPS,
        "-n", NET_FILE,
        "-o", ped_trip_file,
        "-r", ped_route_file,
        "-e", str(DURATION),
        "-p", str(ped_period),
        "--pedestrians",
        "--prefix", "ped",
        "--validate",
        "--seed", str(seed + 57),
    ]
    
    result = subprocess.run(cmd_ped, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠️ Pedoni non generati: {result.stderr[:200]}")
        ped_route_file = None
    else:
        print(f"  ✅ Generati {PEDESTRIAN_COUNT} pedoni")
    
    # ===== STEP 3: Parsing e assegnamento tipi =====
    print("\n🔧 Step 3: Assegnamento tipi di veicolo...")
    
    tree = ET.parse(route_file)
    root_src = tree.getroot()
    
    vehicles = list(root_src.iter("vehicle"))
    print(f"  Trovati {len(vehicles)} veicoli da tipizzare")
    
    # Crea distribuzione veicoli
    type_assignments = []
    for vtype in VEHICLE_TYPES:
        count = int(len(vehicles) * vtype["proportion"])
        type_assignments.extend([vtype["id"]] * count)
    
    # Riempi il resto con auto extra
    remaining = len(vehicles) - len(type_assignments)
    for i in range(remaining):
        type_assignments.append(f"car_extra_{i % len(EXTRA_CAR_COLORS)}")
    
    random.shuffle(type_assignments)
    
    # ===== STEP 4: Costruisci il file finale =====
    print("\n📦 Step 4: Costruzione file finale...")
    
    final_root = ET.Element("routes")
    
    # Aggiungi vTypes
    for vtype in VEHICLE_TYPES:
        vt = ET.SubElement(final_root, "vType")
        for key, val in vtype.items():
            if key != "proportion":
                vt.set(key, str(val))
    
    # Aggiungi auto extra con colori diversi
    for i, color in enumerate(EXTRA_CAR_COLORS):
        vt = ET.SubElement(final_root, "vType")
        vt.set("id", f"car_extra_{i}")
        vt.set("vClass", "passenger")
        vt.set("guiShape", "passenger")
        vt.set("color", color)
        vt.set("maxSpeed", str(round(random.uniform(12.5, 15.28), 2)))  # 45-55 km/h
        vt.set("accel", "2.6")
        vt.set("decel", "4.5")
        vt.set("length", str(round(random.uniform(4.0, 5.0), 1)))
        vt.set("width", "1.8")
        vt.set("minGap", "2.5")
        vt.set("sigma", "0.5")
    
    # vType pedone
    ped_vt = ET.SubElement(final_root, "vType")
    ped_vt.set("id", "pedestrian")
    ped_vt.set("vClass", "pedestrian")
    ped_vt.set("color", "1.0,0.5,1.0")
    ped_vt.set("width", "0.5")
    ped_vt.set("length", "0.3")
    ped_vt.set("minGap", "0.5")
    ped_vt.set("maxSpeed", "1.39")
    ped_vt.set("guiShape", "pedestrian")
    
    # Raccogli tutti gli elementi con depart time
    all_elements = []
    
    for i, veh in enumerate(vehicles):
        veh.set("type", type_assignments[i])
        depart = float(veh.get("depart", "0"))
        all_elements.append((depart, veh))
    
    # Aggiungi pedoni
    if ped_route_file and os.path.exists(ped_route_file):
        ped_tree = ET.parse(ped_route_file)
        for person in ped_tree.getroot().iter("person"):
            depart = float(person.get("depart", "0"))
            all_elements.append((depart, person))
    
    # Ordina per tempo
    all_elements.sort(key=lambda x: x[0])
    
    for _, elem in all_elements:
        final_root.append(elem)
    
    # Scrivi file
    output_file = os.path.join(OUT_DIR, "pisa.rou.xml")
    final_tree = ET.ElementTree(final_root)
    ET.indent(final_tree, space="    ")
    final_tree.write(output_file, encoding="UTF-8", xml_declaration=True)
    
    # ===== STEP 5: Pulizia =====
    print("\n🧹 Step 5: Pulizia file temporanei...")
    for f in os.listdir(OUT_DIR):
        if f.startswith("temp_"):
            os.remove(os.path.join(OUT_DIR, f))
    trip_base = os.path.join(OUT_DIR, "trips.trips.xml")
    if os.path.exists(trip_base):
        os.remove(trip_base)
    
    # ===== RIEPILOGO =====
    # Conta effettivo
    counts = {}
    for ta in type_assignments:
        base = ta.split("_extra_")[0] if "extra" in ta else ta
        if base == "car":
            base = "car" if "extra" not in ta else "car(extra)"
        counts[base] = counts.get(base, 0) + 1
    
    print("\n" + "=" * 55)
    print("📊 RIEPILOGO SCENARIO MEDIO PISA")
    print("=" * 55)
    emojis = {"car": "🚗", "truck": "🚛", "bus": "🚌", "motorcycle": "🏍️", 
              "bicycle": "🚲", "van": "🚐", "car(extra)": "🚙"}
    
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        emoji = emojis.get(name, "🚗")
        print(f"  {emoji} {name:15s}: {count:4d}")
    
    print(f"  🚶 {'pedoni':15s}: {PEDESTRIAN_COUNT:4d}")
    total = len(all_elements)
    print(f"\n  📊 TOTALE ENTITÀ: {total}")
    print(f"  ⏱️  Durata: {DURATION}s (1 ora)")
    print("=" * 55)


if __name__ == "__main__":
    main()
