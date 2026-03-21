import os
import sys
import sumolib

# Configurazione
SUMO_HOME = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_tools", "sumo-1.22.0")
NET_FILE = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_config", "pisa.net.xml")

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

def map_all_tls_v3():
    print(f"Caricamento rete: {NET_FILE}...")
    net = sumolib.net.readNet(NET_FILE)
    
    tls_to_streets = {}
    
    for node in net.getNodes():
        # Verifichiamo se il nodo ha un programma semaforico associato
        tls_id = node.getTlsID()
        if tls_id and tls_id.strip() != "":
            if tls_id not in tls_to_streets:
                tls_to_streets[tls_id] = {
                    "nodes": set(),
                    "streets": set()
                }
            
            tls_to_streets[tls_id]["nodes"].add(node.getID())
            
            # Esaminiamo gli edge in ingresso e uscita per trovare i nomi delle vie
            for edge in node.getIncoming() + node.getOutgoing():
                name = edge.getName()
                if name:
                    tls_to_streets[tls_id]["streets"].add(name)
    
    output_path = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "mappa_completa_semafori_v3.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"MAPPATURA COMPLETA SEMAFORI - PISA\n")
        f.write(f"Totale programmi semaforici (IDs): {len(tls_to_streets)}\n")
        f.write("=" * 70 + "\n\n")
        
        # Ordiniamo per ID
        for tls_id in sorted(tls_to_streets.keys()):
            data = tls_to_streets[tls_id]
            f.write(f"ID PROGRAMMA: {tls_id}\n")
            f.write(f"VIE COINVOLTE:\n")
            if data["streets"]:
                for street in sorted(list(data['streets'])):
                    f.write(f"  - {street}\n")
            else:
                f.write(f"  - (Vie senza nome / Svincoli tecnici)\n")
            f.write("-" * 40 + "\n")
            
    print(f"Mappatura completata. Risultato salvato in: {output_path}")
    return len(tls_to_streets)

if __name__ == "__main__":
    count = map_all_tls_v3()
    print(f"Processati {count} programmi semaforici.")
