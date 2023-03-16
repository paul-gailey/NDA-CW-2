import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import random
import folium
print(folium.__version__)

# City with larger size
query_place = 'Leeds, United Kingdom'
query_place_graph = ox.graph_from_place(query_place, network_type="drive")

# graph_project = ox.project_graph(query_place_graph)
ox.plot_graph_folium(query_place_graph)
