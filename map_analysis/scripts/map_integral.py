import os
import sys
import sumolib

# Configurazione
SUMO_HOME = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_tools", "sumo-1.22.0")
NET_FILE = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_config", "pisa.net.xml")

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

def map_all_streets_to_tls():
    print(f"Mappatura integrale in corso...")
    net = sumolib.net.readNet(NET_FILE)
    tls_list = net.getTrafficLights()
    
    output_path = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "mappa_totale_pisa_vie.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("MAPPATURA INTEGRALE SEMAFORI E VIE - PISA\n")
        f.write(f"Numero di logiche semaforiche: {len(tls_list)}\n")
        f.write("=" * 80 + "\n\n")
        
        for tls in sorted(tls_list, key=lambda x: x.getID()):
            tls_id = tls.getID()
            edges = tls.getEdges()
            
            # Dividiamo vie in entrata e uscita per precisione
            in_streets = set()
            out_streets = set()
            
            # Cerchiamo i nodi associati per capire meglio la topologia
            nodes = set()
            
            for edge in edges:
                name = edge.getName()
                if name:
                    # Non sappiamo se è in o out solo dall'oggetto TLS facilmente
                    # ma possiamo dedurlo guardando i nodi
                    in_streets.add(name)
            
            f.write(f"ID SEMAFORO: {tls_id}\n")
            f.write(f"VIE PRINCIPALI:\n")
            if in_streets:
                for s in sorted(list(in_streets)):
                    f.write(f"  - {s}\n")
            else:
                f.write("  - (Vie senza nome / Svincoli)\n")
            
            # Aggiungiamo coordinate medie per riferimento
            all_nodes = []
            # Il TLS object ha i node IDs? No, ma possiamo trovarli
            # In realtà getEdges() è già buono.
            
            f.write("-" * 50 + "\n")
            
    print(f"Mappatura completata: {output_path}")

if __name__ == "__main__":
    map_all_streets_to_tls()
