import os
import sys
import sumolib

# Configurazione
SUMO_HOME = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_tools", "sumo-1.22.0")
NET_FILE = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_config", "pisa.net.xml")

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

def map_all_tls_final():
    print(f"Caricamento rete: {NET_FILE}...")
    net = sumolib.net.readNet(NET_FILE)
    
    tls_list = net.getTrafficLights()
    print(f"Trovate {len(tls_list)} logiche semaforiche.")
    
    output_path = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "mappa_completa_semafori_v4.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"MAPPATURA COMPLETA SEMAFORI - PISA\n")
        f.write(f"Totale programmi semaforici (IDs): {len(tls_list)}\n")
        f.write("=" * 70 + "\n\n")
        
        # Ordiniamo per ID
        sorted_tls = sorted(tls_list, key=lambda x: x.getID())
        
        for tls in sorted_tls:
            tls_id = tls.getID()
            edges = tls.getEdges()
            
            street_names = set()
            for edge in edges:
                name = edge.getName()
                if name:
                    street_names.add(name)
            
            f.write(f"ID PROGRAMMA: {tls_id}\n")
            f.write(f"VIE COINVOLTE:\n")
            if street_names:
                for street in sorted(list(street_names)):
                    f.write(f"  - {street}\n")
            else:
                # Se non ci sono nomi, mettiamo gli ID degli edge
                edge_ids = [e.getID() for e in edges]
                f.write(f"  - (Senza nome) Edges: {', '.join(edge_ids[:5])}...\n")
            f.write("-" * 40 + "\n")
            
    print(f"Mappatura completata. Risultato salvato in: {output_path}")
    return len(tls_list)

if __name__ == "__main__":
    count = map_all_tls_final()
    print(f"Processati {count} semafori.")
