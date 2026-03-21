import os
import sys
import sumolib

# Configurazione
SUMO_HOME = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_tools", "sumo-1.22.0")
NET_FILE = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_config", "pisa.net.xml")

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

def map_all_traffic_lights():
    print(f"Caricamento rete: {NET_FILE}...")
    net = sumolib.net.readNet(NET_FILE)
    nodes = net.getNodes()
    
    tls_full_map = []
    
    for node in nodes:
        if node.getType() == "traffic_light":
            node_id = node.getID()
            incoming = node.getIncoming()
            outgoing = node.getOutgoing()
            
            # Raccogliamo i nomi unici delle vie
            street_names = set()
            for edge in incoming + outgoing:
                name = edge.getName()
                if name:
                    street_names.add(name)
            
            # Se non ci sono nomi (vie senza nome in OSM), raccogliamo gli ID degli edge
            if not street_names:
                for edge in incoming + outgoing:
                    street_names.add(f"Unnamed_Edge_{edge.getID()}")
            
            tls_full_map.append({
                "id": node_id,
                "streets": sorted(list(street_names)),
                "num_incoming": len(incoming),
                "num_outgoing": len(outgoing),
                "coord": node.getCoord()
            })
    
    # Ordiniamo per ID per leggibilità
    tls_full_map.sort(key=lambda x: x['id'])
    
    output_path = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "mappa_completa_semafori.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"MAPPATURA COMPLETA SEMAFORI - PISA\n")
        f.write(f"Totale semafori individuati: {len(tls_full_map)}\n")
        f.write("=" * 70 + "\n\n")
        
        for entry in tls_full_map:
            f.write(f"ID SEMAFORO: {entry['id']}\n")
            f.write(f"COORDINATE: {entry['coord']}\n")
            f.write(f"VIE COINVOLTE:\n")
            for street in entry['streets']:
                f.write(f"  - {street}\n")
            f.write("-" * 40 + "\n")
            
    print(f"Mappatura completata. Risultato salvato in: {output_path}")
    return len(tls_full_map)

if __name__ == "__main__":
    count = map_all_traffic_lights()
    print(f"Processati {count} semafori.")
