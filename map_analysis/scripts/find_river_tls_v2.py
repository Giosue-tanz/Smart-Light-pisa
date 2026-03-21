import os
import sys
import sumolib

# Configurazione
SUMO_HOME = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_tools", "sumo-1.22.0")
NET_FILE = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_config", "pisa.net.xml")

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

def find_all_river_tls():
    net = sumolib.net.readNet(NET_FILE)
    tls_found = []
    
    # Keyword estese per il fiume Arno e i Ponti di Pisa
    keywords = [
        "lungarno", "ponte", "piazza garibaldi", "piazza xx settembre", 
        "piazza solferino", "piazza mazzini", "piazza della gherardesca"
    ]
    
    for node in net.getNodes():
        if node.getType() == "traffic_light":
            edges = node.getIncoming() + node.getOutgoing()
            street_names = set()
            is_river_related = False
            
            for edge in edges:
                name = edge.getName()
                if name:
                    street_names.add(name)
                    if any(kw.lower() in name.lower() for kw in keywords):
                        is_river_related = True
            
            if is_river_related:
                tls_found.append({
                    "id": node.getID(),
                    "coord": node.getCoord(),
                    "streets": list(street_names)
                })
    
    return tls_found

if __name__ == "__main__":
    results = find_all_river_tls()
    print(f"Semafori trovati lungo l'Arno/Ponti: {len(results)}")
    print("=" * 60)
    # Ordiniamo per coordinata X per seguire il fiume da Ovest a Est (approssimativamente)
    results.sort(key=lambda x: x['coord'][0])
    
    for res in results:
        print(f"ID: {res['id']}")
        print(f"Strade: {', '.join(res['streets'])}")
        print(f"Coord: {res['coord']}")
        print("-" * 30)
