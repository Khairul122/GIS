from flask import Flask, render_template, request, jsonify
import networkx as nx
import json
import osmnx as ox
from haversine import haversine
import os
import time
import numpy as np
from sklearn.neighbors import BallTree

app = Flask(__name__)

# Setup folder data
if not os.path.exists('data'):
    os.makedirs('data')

# Konfigurasi
CITY_NAME = "Lhokseumawe, Indonesia"
OSM_GRAPH_FILE = 'data/lhokseumawe_graph.graphml'
LHOKSEUMAWE_LAT, LHOKSEUMAWE_LON = 5.1801, 97.1510

# Variabel global untuk graph
global_graph = None

# Deteksi versi OSMNX
OSMNX_VERSION = ox.__version__
print(f"Menggunakan OSMNX versi {OSMNX_VERSION}")

def get_route_edge_attributes(G, route, attribute):
    """Implementasi manual untuk mendapatkan atribut edge dari rute"""
    print(f"Menggunakan implementasi manual get_route_edge_attributes untuk route dengan {len(route)} node")
    
    attributes = []
    for u, v in zip(route[:-1], route[1:]):
        # Periksa jika edge memiliki atribut yang diminta
        if G.has_edge(u, v):
            # Cek semua edge key (multi-edge)
            edge_keys = list(G[u][v].keys()) if isinstance(G[u][v], dict) else [0]
            for k in edge_keys:
                if attribute in G[u][v][k]:
                    attributes.append(G[u][v][k][attribute])
                    break
            else:
                # Jika tidak ada edge key yang memiliki atribut
                if attribute == 'length':
                    try:
                        # Estimasi jarak dari koordinat
                        u_y, u_x = G.nodes[u]['y'], G.nodes[u]['x']
                        v_y, v_x = G.nodes[v]['y'], G.nodes[v]['x']
                        dist_km = haversine((u_y, u_x), (v_y, v_x))
                        # Konversi km ke meter untuk konsistensi dengan OSM
                        attributes.append(dist_km * 1000)
                    except:
                        attributes.append(0)
                else:
                    attributes.append(0)
        elif G.has_edge(v, u):
            # Coba edge di arah yang berlawanan jika graf tidak terarah
            edge_keys = list(G[v][u].keys()) if isinstance(G[v][u], dict) else [0]
            for k in edge_keys:
                if attribute in G[v][u][k]:
                    attributes.append(G[v][u][k][attribute])
                    break
            else:
                if attribute == 'length':
                    try:
                        u_y, u_x = G.nodes[u]['y'], G.nodes[u]['x']
                        v_y, v_x = G.nodes[v]['y'], G.nodes[v]['x']
                        dist_km = haversine((u_y, u_x), (v_y, v_x))
                        attributes.append(dist_km * 1000)
                    except:
                        attributes.append(0)
                else:
                    attributes.append(0)
        else:
            # Jika tidak ada edge sama sekali
            if attribute == 'length':
                try:
                    u_y, u_x = G.nodes[u]['y'], G.nodes[u]['x']
                    v_y, v_x = G.nodes[v]['y'], G.nodes[v]['x']
                    dist_km = haversine((u_y, u_x), (v_y, v_x))
                    attributes.append(dist_km * 1000)
                except:
                    attributes.append(0)
            else:
                attributes.append(0)
                
    return attributes

def predownload_osm_data():
    """Download data OSM untuk kota Lhokseumawe jika belum ada"""
    if os.path.exists(OSM_GRAPH_FILE):
        print(f"File OSM graph sudah ada di {OSM_GRAPH_FILE}")
        return True
    
    print(f"Mendownload data OSM untuk {CITY_NAME}...")
    try:
        # Gunakan versi terbaru OSMNX untuk download
        G = ox.graph_from_place(CITY_NAME, network_type="drive", simplify=True)
        
        # Jika berhasil, simpan graph
        ox.save_graphml(G, OSM_GRAPH_FILE)
        print(f"Data OSM berhasil didownload dan disimpan di {OSM_GRAPH_FILE}")
        return True
    except Exception as e:
        print(f"Error saat download data OSM: {str(e)}")
        
        # Coba pendekatan alternatif dengan point dan radius
        try:
            print("Mencoba pendekatan alternatif dengan titik pusat dan radius...")
            G = ox.graph_from_point((LHOKSEUMAWE_LAT, LHOKSEUMAWE_LON), 
                                  dist=10000, 
                                  network_type="drive", 
                                  simplify=True)
            ox.save_graphml(G, OSM_GRAPH_FILE)
            print(f"Data OSM berhasil didownload dengan pendekatan alternatif")
            return True
        except Exception as e2:
            print(f"Gagal download data dengan pendekatan alternatif: {str(e2)}")
            return False

def initialize_graph():
    global global_graph
    
    # Cek dan download data OSM jika diperlukan
    predownload_osm_data()
    
    # Coba muat graph yang sudah ada
    if os.path.exists(OSM_GRAPH_FILE):
        print(f"Memuat graph dari {OSM_GRAPH_FILE}...")
        try:
            G = ox.load_graphml(OSM_GRAPH_FILE)
            
            # Pastikan graph tidak kosong
            if len(G.nodes) == 0:
                raise ValueError("Graph kosong")
                
            # Pastikan semua node memiliki koordinat
            for node, data in G.nodes(data=True):
                if 'x' not in data or 'y' not in data:
                    print(f"Node {node} tidak memiliki koordinat, mencoba memperbaiki...")
                    
            # Ubah menjadi graph tidak terarah untuk routing yang lebih baik
            G = G.to_undirected()
            
            # Bersihkan graph dari node-node yang terisolasi
            isolated_nodes = list(nx.isolates(G))
            if isolated_nodes:
                print(f"Menghapus {len(isolated_nodes)} node terisolasi...")
                G.remove_nodes_from(isolated_nodes)
            
            # Dapatkan komponen terbesar untuk memastikan routing berfungsi
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            
            print(f"Graph berhasil dimuat dan dibersihkan: {len(G.nodes)} node dan {len(G.edges)} edge")
            global_graph = G
            return
        except Exception as e:
            print(f"Error memuat graph: {str(e)}")
            if os.path.exists(OSM_GRAPH_FILE):
                print(f"Menghapus file graph yang rusak...")
                os.remove(OSM_GRAPH_FILE)
            
            # Coba buat ulang
            predownload_osm_data()
            if os.path.exists(OSM_GRAPH_FILE):
                try:
                    G = ox.load_graphml(OSM_GRAPH_FILE)
                    global_graph = G
                    return
                except Exception as e2:
                    print(f"Error saat percobaan kedua: {str(e2)}")

    # Jika sampai di sini berarti perlu membuat graph baru
    try:
        print("Membuat graph baru karena file tidak ada atau rusak...")
        G = ox.graph_from_point((LHOKSEUMAWE_LAT, LHOKSEUMAWE_LON), 
                              dist=10000, 
                              network_type="drive", 
                              simplify=True)
                              
        # Konversi CRS jika diperlukan
        if G.graph['crs'] != 'epsg:4326':
            print("Mengonversi CRS ke WGS84...")
            G = ox.project_graph(G, to_crs='epsg:4326')
        
        # Simpan graph yang baru
        ox.save_graphml(G, OSM_GRAPH_FILE)
        global_graph = G
        print(f"Graph baru berhasil dibuat dengan {len(G.nodes)} node dan {len(G.edges)} edge")
        return
    except Exception as e:
        print(f"Error total dalam pembuatan graph: {str(e)}")
        # Sebagai fallback terakhir, buat graph sederhana
        global_graph = nx.Graph()
        print("Membuat graph kosong sebagai fallback terakhir")

def get_nearest_node(lat, lon, G):
    """Mencari node terdekat dalam graph dengan scikit-learn"""
    try:
        # Metode 1: Gunakan fungsi OSMNX
        node = ox.distance.nearest_nodes(G, lon, lat)
        return node
    except Exception as e:
        print(f"Error dengan ox.distance.nearest_nodes: {str(e)}")
        
        # Metode 2: Implementasi manual dengan scikit-learn
        try:
            # Ekstrak semua koordinat node
            node_ids = list(G.nodes())
            node_coords = np.array([[G.nodes[node]['y'], G.nodes[node]['x']] for node in node_ids])
            
            # Buat BallTree untuk pencarian kNN
            tree = BallTree(node_coords, metric='haversine')
            
            # Cari node terdekat
            point = np.array([[lat, lon]])
            dist, ind = tree.query(point, k=1)
            nearest_node_idx = ind[0][0]
            
            return node_ids[nearest_node_idx]
        except Exception as e2:
            print(f"Error dengan scikit-learn: {str(e2)}")
            
            # Fallback ke pencarian linear sederhana
            min_dist = float('inf')
            nearest_node = None
            
            for node, data in G.nodes(data=True):
                try:
                    node_lat = data['y']
                    node_lon = data['x']
                    
                    # Hitung jarak haversine
                    dist = haversine((lat, lon), (node_lat, node_lon))
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node = node
                except KeyError:
                    continue
            
            return nearest_node

def calculate_heuristic(node1, node2, G):
    """Menghitung heuristik untuk A* (jarak haversine)"""
    y1, x1 = G.nodes[node1]['y'], G.nodes[node1]['x']
    y2, x2 = G.nodes[node2]['y'], G.nodes[node2]['x']
    return haversine((y1, x1), (y2, x2))

def find_optimal_route(start_lat, start_lon, end_lat, end_lon, G):
    """Mencari rute optimal menggunakan algoritma A* dengan fallback ke Dijkstra"""
    if G is None or len(G.nodes) == 0:
        return {
            'path': [[start_lat, start_lon], [end_lat, end_lon]],
            'fallback': True,
            'message': 'Graph jalan tidak tersedia'
        }
    
    # Debug info
    print(f"Mencari rute dari {start_lat},{start_lon} ke {end_lat},{end_lon}")
    print(f"Graph memiliki {len(G.nodes)} node dan {len(G.edges)} edge")
    
    try:
        # Dapatkan node terdekat
        start_node = get_nearest_node(start_lat, start_lon, G)
        end_node = get_nearest_node(end_lat, end_lon, G)

        if not start_node or not end_node:
            raise ValueError("Node awal/akhir tidak ditemukan")
            
        print(f"Node terdekat: start={start_node}, end={end_node}")

        # Periksa apakah node berada di komponen yang sama (terhubung)
        if not nx.has_path(G, start_node, end_node):
            raise ValueError("Tidak ada jalur yang menghubungkan titik awal dan akhir")

        # Coba algoritma A*
        try:
            # Hitung rute dengan A* algorithm
            print("Mencoba algoritma A*...")
            route = nx.astar_path(
                G, 
                start_node, 
                end_node, 
                heuristic=lambda n1, n2: calculate_heuristic(n1, n2, G),
                weight='length'
            )
            algorithm = 'A*'
        except Exception as astar_error:
            print(f"A* gagal: {str(astar_error)}, mencoba Dijkstra...")
            # Fallback ke Dijkstra
            route = nx.shortest_path(G, start_node, end_node, weight='length')
            algorithm = 'Dijkstra (fallback)'

        # Ekstrak koordinat
        if not route:
            raise ValueError("Rute tidak ditemukan")

        print(f"Rute ditemukan dengan {len(route)} node menggunakan {algorithm}")
        
        path_coords = []
        for node in route:
            point = G.nodes[node]
            try:
                path_coords.append([point['y'], point['x']])
            except KeyError:
                # Jika koordinat tidak ditemukan, skip node ini
                print(f"Koordinat tidak ditemukan untuk node {node}")
                continue

        # Hitung jarak total dan waktu
        # Tangani kasus edge tidak memiliki atribut 'length'
        total_length = 0
        try:
            edge_lengths = get_route_edge_attributes(G, route, 'length')
            if edge_lengths:
                total_length = sum(edge_lengths)/1000  # dalam km
        except Exception as length_err:
            print(f"Error menghitung panjang rute: {str(length_err)}")
            # Estimasi panjang rute dari koordinat
            for i in range(1, len(path_coords)):
                total_length += haversine(
                    (path_coords[i-1][0], path_coords[i-1][1]), 
                    (path_coords[i][0], path_coords[i][1])
                )
                
        estimated_time = total_length / 40 * 60  # asumsi 40 km/jam

        return {
            'path': path_coords,
            'total_distance': round(total_length, 2),
            'estimated_time': round(estimated_time),
            'algorithm': algorithm,
            'node_count': len(route)
        }

    except Exception as e:
        print(f"Error routing: {str(e)}")
        
        # Jika node awal dan akhir valid tetapi perhitungan rute gagal,
        # coba lagi dengan pendekatan berbasis komponen terhubung
        if start_node and end_node:
            try:
                print("Mencoba analisis komponen terhubung...")
                # Dapatkan komponen terhubung
                connected_components = list(nx.connected_components(G.to_undirected()))
                print(f"Graf memiliki {len(connected_components)} komponen terhubung")
                
                # Cari komponen yang berisi start dan end node
                start_component = None
                end_component = None
                
                for i, component in enumerate(connected_components):
                    if start_node in component:
                        start_component = i
                    if end_node in component:
                        end_component = i
                
                if start_component is not None and end_component is not None:
                    if start_component == end_component:
                        print(f"Kedua node berada dalam komponen yang sama ({start_component})")
                        # Coba sekali lagi dengan Dijkstra pada subgraph
                        subgraph = G.subgraph(connected_components[start_component])
                        route = nx.shortest_path(subgraph, start_node, end_node, weight='length')
                        
                        # Proses rute
                        path_coords = []
                        for node in route:
                            point = G.nodes[node]
                            path_coords.append([point['y'], point['x']])
                        
                        # Hitung jarak perkiraan
                        total_length = 0
                        for i in range(1, len(path_coords)):
                            total_length += haversine(
                                (path_coords[i-1][0], path_coords[i-1][1]), 
                                (path_coords[i][0], path_coords[i][1])
                            )
                        
                        estimated_time = total_length / 40 * 60
                        
                        return {
                            'path': path_coords,
                            'total_distance': round(total_length, 2),
                            'estimated_time': round(estimated_time),
                            'algorithm': 'Dijkstra (komponen)',
                            'node_count': len(route)
                        }
                    else:
                        print(f"Nodes berada dalam komponen berbeda: start={start_component}, end={end_component}")
            except Exception as comp_error:
                print(f"Error dalam analisis komponen: {str(comp_error)}")
        
        # Fallback ke rute garis lurus
        return {
            'path': [[start_lat, start_lon], [end_lat, end_lon]],
            'fallback': True,
            'message': f'Tidak dapat menghitung rute: {str(e)}',
            'total_distance': round(haversine((start_lat, start_lon), (end_lat, end_lon)), 2),
            'estimated_time': round(haversine((start_lat, start_lon), (end_lat, end_lon)) / 40 * 60)
        }

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_route', methods=['POST'])
def find_route():
    data = request.json
    
    # Ekstrak koordinat dari input
    try:
        # Data mungkin berupa koordinat atau nama lokasi yang sudah di-geocode di client
        start_input = data['start']
        end_input = data['end']
        
        # Coba parse koordinat
        try:
            start = list(map(float, start_input.split(',')))
            end = list(map(float, end_input.split(',')))
            
            if len(start) != 2 or len(end) != 2:
                return jsonify({'error': 'Format koordinat tidak valid'}), 400
        except:
            return jsonify({'error': 'Format koordinat tidak valid'}), 400
    except:
        return jsonify({'error': 'Data tidak lengkap'}), 400

    result = find_optimal_route(start[0], start[1], end[0], end[1], global_graph)
    return jsonify(result)

@app.route('/graph_status')
def graph_status():
    status = {
        'graph_loaded': global_graph is not None,
        'node_count': len(global_graph.nodes) if global_graph else 0,
        'edge_count': len(global_graph.edges) if global_graph else 0,
        'graph_file': OSM_GRAPH_FILE,
        'osmnx_version': OSMNX_VERSION
    }
    return jsonify(status)

@app.route('/reload_graph')
def reload_graph():
    try:
        # Hapus cache graph jika ada
        if os.path.exists(OSM_GRAPH_FILE):
            os.remove(OSM_GRAPH_FILE)
        
        # Reinisialisasi
        initialize_graph()
        
        return jsonify({
            'success': True,
            'message': 'Graph berhasil direload',
            'node_count': len(global_graph.nodes) if global_graph else 0
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error saat reload graph: {str(e)}'
        })

if __name__ == '__main__':
    print("Menginisialisasi graph...")
    initialize_graph()
    app.run(debug=True)