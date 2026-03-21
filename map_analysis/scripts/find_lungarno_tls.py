import os
import sys
import sumolib

# Configurazione
SUMO_HOME = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_tools", "sumo-1.22.0")
NET_FILE = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_config", "pisa.net.xml")

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

def find_tls_on_street(street_name_part):
    net = sumolib.net.readNet(NET_FILE)
    tls_on_street = []
    
    # Cerchiamo tutti i nodi (junctions) che sono traffic lights
    for node in net.getNodes():
        if node.getType() == "traffic_light":
            # Controlliamo i nomi degli edge in entrata e uscita
            edges = node.getIncoming() + node.getOutgoing()
            found = False
            street_names = set()
            for edge in edges:
                name = edge.getName()
                if name:
                    street_names.add(name)
                    if street_name_part.lower() in name.lower():
                        found = True
            
            if found:
                tls_on_street.append({
                    "id": node.getID(),
                    "coord": node.getCoord(),
                    "streets": list(street_names)
                })
    
    return tls_on_street

if __name__ == "__main__":
    results = find_tls_on_street("Lungarno")
    print(f"Semafori trovati su 'Lungarno': {len(results)}")
    print("-" * 50)
    for res in results:
        print(f"ID: {res['id']}")
        print(f"Posizione: {res['coord']}")
        print(f"Strade coinvolte: {', '.join(res['streets'])}")
        print("-" * 30)
