import os
import sys
import sumolib

# Configurazione
SUMO_HOME = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_tools", "sumo-1.22.0")
NET_FILE = os.path.join("c:\\Users\\giosu\\Desktop\\smartlight_pisa", "sumo_config", "pisa.net.xml")

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = SUMO_HOME
sys.path.append(os.path.join(SUMO_HOME, 'tools'))

def find_river_corridor():
    net = sumolib.net.readNet(NET_FILE)
    corridor_ids = [
        '33681447', '278246118', '251208763', '704411639', '31770712',
        'cluster_12358102210_12358102217_12358102223_2674835073_#1more',
        'cluster_8254824560_8443645582_8443645594_8443645595'
    ]
    
    # Cerchiamo altri vicini a questi
    all_tls = [n for n in net.getNodes() if n.getType() == "traffic_light"]
    
    final_list = []
    for tls in all_tls:
        node_id = tls.getID()
        edges = tls.getIncoming() + tls.getOutgoing()
        names = [e.getName() for e in edges if e.getName()]
        
        # Filtro geografico e per nome
        is_in = False
        if node_id in corridor_ids:
            is_in = True
        else:
            kws = ['lungarno', 'ponte', 'piazza', 'garibaldi', 'matteotti', 'vittoria', 'fortezza', 'imperatore', 'impero']
            if any(any(kw in n.lower() for kw in kws) for n in names):
                is_in = True
        
        if is_in:
            final_list.append({
                "id": node_id,
                "streets": list(set(names)),
                "coord": tls.getCoord()
            })
            
    return final_list

if __name__ == "__main__":
    res = find_river_corridor()
    print(f"Totale Semafori Corridoio Arno found: {len(res)}")
    res.sort(key=lambda x: x['coord'][0])
    for r in res:
        print(f"{r['id']} | {r['coord']} | {r['streets']}")
