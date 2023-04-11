import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import random
import numpy as np
import os
import pandas as pd
import glob
from geopandas import GeoDataFrame
from shapely.geometry import Point
import spaghetti
import math
import pyproj
import esda
import seaborn as sns
import statistics


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the distance (in meters) between two points on the Earth's surface
    given their latitude and longitude coordinates using the Haversine formula.
    """
    R = 6371000  # radius of the Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R*c

# Checks to see if a point (lat, lon) is in a bounding box (N, S, E, W)
def point_in_bbox(point):
    N, S, E, W = leeds_bbox
    lat, lon = point

    if N >= lat >= S and E >= lon >= W:
        return True
    else:
        return False

center_lat = 53.802035
center_lon = -1.537810

# Define the center point of Leeds
leeds_center = (center_lat, center_lon)

# Create the graph for Leeds city center
leeds_graph = ox.graph.graph_from_point(leeds_center, dist=500, dist_type='bbox', network_type='drive')

# Get the bounding box
leeds_bbox = ox.utils_geo.bbox_from_point(leeds_center, dist=500)

# Create a polygon - use it later
leeds_polygon = ox.utils_geo.bbox_to_poly(leeds_bbox[0], leeds_bbox[1], leeds_bbox[2], leeds_bbox[3])

# Find the Area
base = haversine(leeds_bbox[0], leeds_bbox[2], leeds_bbox[0], leeds_bbox[3])
height = haversine(leeds_bbox[0], leeds_bbox[3], leeds_bbox[1], leeds_bbox[3])
leeds_area = base * height

ox.plot_graph(leeds_graph)
ox.plot_graph_folium(leeds_graph)

# Calculate Basic Stats
basic_stats = ox.stats.basic_stats(leeds_graph, area=leeds_area)

# Node and edge data
print("There are {} nodes and {} edges in this graph".format(basic_stats["n"], basic_stats["m"]))

# Average Street Length
print("Average street length: {}m".format(round(basic_stats["street_length_avg"], 2)))

# Node Density
print("Node density of {} nodes per square km".format(round(basic_stats["node_density_km"], 2)))

# Intersection Density
print("Intersection density of {} intersections per square km".format(basic_stats["intersection_count"]))

# Edge Density
print("Edge density of {} edges per square km".format(round(basic_stats["edge_density_km"], 2)))

print(basic_stats)

# Compute extended network statistics, including spatial diameter
spatial_diameter = ox.extended_stats(leeds_graph, ecc=True, bc=True, cc=True)['diameter']

# Print the spatial diameter
print("Spatial diameter of the network is: {} meters".format(round(spatial_diameter, 2)))

is_planar, kuratowski_subgraphs = nx.check_planarity(leeds_graph, counterexample=True)
if is_planar:
    print("Planar")
else:
    print("Not Planar")

plt.figure(figsize=(18, 8))
nx.draw(kuratowski_subgraphs, pos=nx.kamada_kawai_layout(kuratowski_subgraphs), arrows=True, arrowstyle='-|>', with_labels=True)
plt.show()

ACCIDENTS_DIR = "data"  # Assumes that the crime data is located in the 'data' subfolder
USE_COLS = ['Grid Ref: Easting', 'Grid Ref: Northing', 'Accident Date']  # Only use these columns

# Load all CSV files in the 'data' subfolders into a single dataframe
years_df = []
for path, subdir, _ in os.walk(ACCIDENTS_DIR):
    for csv_file in glob.glob(os.path.join(path, "*.csv")):
        year_df = pd.read_csv(csv_file, usecols=USE_COLS, encoding='iso-8859-1')
        years_df.append(year_df)

accidents_df = pd.concat(years_df, axis=0, ignore_index=True)
print("Found {} CSV files in '{}', total no. of accidents loaded: {}"
      .format(len(years_df), ACCIDENTS_DIR, len(accidents_df)))

# Filter out accidents that do not have a location (missing latitude or longitude)
located_accidents = accidents_df.dropna(subset=['Grid Ref: Easting', 'Grid Ref: Northing'])
print("Total no. of located accidents: {}".format(len(located_accidents)))

located_accidents.rename(columns={"Accident Date": "Accident Year"}, inplace=True)
located_accidents['Accident Year'] = ('20' + located_accidents['Accident Year'].str[-2:]).astype(int)
located_accidents.sort_values(by="Accident Year", inplace=True)
print(located_accidents)

#Translate to a GeoDataFrame where the geometry is given by a Point constructed from the longitude and latitude
geometry = [Point(xy) for xy in zip(located_accidents["Grid Ref: Easting"], located_accidents["Grid Ref: Northing"])]
accident_points = GeoDataFrame(located_accidents["Accident Year"], geometry=geometry)

# Define the input and output CRS
input_crs = "EPSG:27700"  # BNG
output_crs = "EPSG:4326"  # WGS 84, the most common CRS for latitude and longitude

# Create a PyProj transformer to convert between the CRSs
transformer = pyproj.Transformer.from_crs(input_crs, output_crs)

# Apply the transformer to the POINT geometry column
accident_points["lonlat"] = accident_points["geometry"].apply(lambda geom: transformer.transform(geom.x, geom.y))

# Replace the "geometry" column with the new "lonlat" column
accident_points["geometry"] = accident_points["lonlat"].apply(lambda lonlat: Point(lonlat[1], lonlat[0]))

# Apply a function to check whether each point is within a bounding box
accident_points["in_bbox"] = accident_points["lonlat"].apply(point_in_bbox)

# Filter out points outside the bounding box, drop the "lonlat" column, and sort by year
leeds_accidents = accident_points[accident_points["in_bbox"]].drop(columns=["lonlat"]).sort_values(by="Accident Year")

print(leeds_accidents)

# Convert the Leeds street network graph into two GeoDataFrames for nodes and edges
nodes_df, edges_df = ox.graph_to_gdfs(leeds_graph)

# Generate a Spaghetti Network from the edges DataFrame
leeds_points_graph = spaghetti.Network(in_data=edges_df)

# Plot the edges of the network as black lines on a new figure with a size of 15x15 inches and a default Z-order of 0
base_network = edges_df.plot(color="k", zorder=0, figsize=(15, 15))

# Plot the nodes of the network as red dots on the same figure with a Z-order of 2 (i.e., on top of the edges)
nodes_df.plot(ax=base_network, color="r", zorder=2)

# Generate a Spaghetti Network from the edges DataFrame

# We will now snap the accident we extracted earlier, i.e. position them at the closest point on the closest road
leeds_points_graph.snapobservations(leeds_accidents, 'accidents')

# We can see the difference between the original accident coordinates and their position when snapped to the road network
print("observation 1\ntrue coords:\t%s\nsnapped coords:\t%s" % (
    leeds_points_graph.pointpatterns["accidents"].points[0]["coordinates"],
    leeds_points_graph.pointpatterns["accidents"].snapped_coordinates[0]
))

# Show the network
base_network = edges_df.plot(color="k", zorder=0, figsize =(12, 12))
# Get a GeoDataFrame of the snapped crime locations to plot on the network image
snapped_accidents=spaghetti.element_as_gdf(
    leeds_points_graph, pp_name='accidents', snapped=True)

# Plot these on the road network
snapped_accidents.plot(
    color="r", marker="x",
    markersize=50, zorder=1, ax=base_network)

plt.show()


# Show the network
base_network = edges_df.plot(color="k", zorder=0, figsize =(12, 12))
# Get a GeoDataFrame of the non-snapped (real) crime locations to plot on the net
observed_accidents=spaghetti.element_as_gdf(
    leeds_points_graph, pp_name='accidents', snapped=False)

# Plot these on the road network
observed_accidents.plot(
    color="r", marker="x",
    markersize=50, zorder=1, ax=base_network)

# Create a new figure and axes object
fig, ax = plt.subplots(figsize=(20,20))

# Plot a KDE heatmap of the snapped accidents on the road network
# Increase sensitivity of the heatmap by decreasing the bandwidth
sns.kdeplot(
    x=snapped_accidents.geometry.x,
    y=snapped_accidents.geometry.y,
    cmap='Reds',
    shade=True,
    shade_lowest=False,
    alpha=0.5,
    ax=ax,
    bw_adjust=0.65  # Change this value to adjust the bandwidth
)

# Show the network on the same axes object
edges_df.plot(color="k", zorder=0, ax=ax)

# Display the plot
plt.show()


kres = leeds_points_graph.GlobalAutoK(
    leeds_points_graph.pointpatterns["accidents"],
    nsteps=50, permutations=100
)

kres.lam
kres.xaxis
kres.observed
kres.upperenvelope
kres.lowerenvelope
kres.sim

print(f"Density of points in the network (lambda): {kres.lam}")
print(f"Distances at which density is measured:\n{kres.xaxis}")

fig, ax = plt.subplots()

ax.plot(kres.xaxis, kres.observed, "b-", label="Observed")
ax.plot(kres.xaxis, kres.upperenvelope, "r--", label="Upper")
ax.plot(kres.xaxis, kres.lowerenvelope, "k--", label="Lower")

ax.legend(loc="best", fontsize="x-large")
ax.set_xlabel("Distance $(r)$")
ax.set_ylabel("$K(r)$")

fig.tight_layout()

# Get snapped point pattern
pointpat = leeds_points_graph.pointpatterns['accidents']
# Get count of points per network edge: a dictionary from each edge to the crime count on that edge
counts = leeds_points_graph.count_per_link(pointpat.obs_to_arc, graph=False)
print(counts)


# Get the weights matrix for edges in the graph (just the adjacency matrix with 1 where edges connect at a node, 0 otherwise)
weights = leeds_points_graph.w_network

# Get the edges included in the weights matrix: an enumerator for a list of edges
edges = weights.neighbors.keys()
# Construct an array of the counts values per edge in the same order as
# the weights matrix, with 0.0 where no counts recorded
values = [counts[edge] if edge in counts.keys () else 0. \
    for index, edge in enumerate(edges)]


moran = esda.moran.Moran(values, weights)
print(moran.I)
print(moran.p_sim)

moran.EI
moran.EI_sim

print(moran.EI)

sns.kdeplot(moran.sim, shade=True)
plt.vlines(moran.I, 0, 1, color='r')
plt.vlines(moran.EI, 0,1)
plt.xlabel("Moran's I")

print(moran.z_norm)

snapped_accidents=spaghetti.element_as_gdf(
    leeds_points_graph, pp_name='accidents', snapped=True)


# City with larger size
query_place = 'Leeds, United Kingdom'
full_leeds_graph = ox.graph_from_place(query_place, network_type="all")

# graph_project = ox.project_graph(query_place_graph)
ox.plot_graph(full_leeds_graph, figsize=(20,20), node_size=5)

NUMBER_OF_SEEDS = 10

all_nodes = set(full_leeds_graph.nodes)
seeds = random.choices(list(all_nodes), k=NUMBER_OF_SEEDS)

colours = ox.plot.get_colors(NUMBER_OF_SEEDS)

def nearest_from_list(node_distances):
    return sorted(node_distances, key=lambda node_length: node_length[1])[0] \
        if len(node_distances) > 0 else None

def nearest_seed(node, dist, seeds, cache={}):
    if node in cache:
        return cache[node]
    seed_distances = [(seed, dist[seed][node]) for seed in seeds if node in dist[seed]]
    nearest = nearest_from_list(seed_distances)
    cache[node] = nearest
    return nearest

def nearest_for_edge(edge, dist, seeds):
    nearest_to_ends_all = [nearest_seed(edge[0], dist, seeds), nearest_seed(edge[1], dist, seeds)]
    nearest_to_ends = [distance for distance in nearest_to_ends_all if distance]
    return nearest_from_list(nearest_to_ends)

def colour_for_seed_distance(seed):
    if seed and seed[0] in seeds:
        return colours[seeds.index(seed[0])]
    else:
        return 'k'  # Return black color for edges not connected to any seed


def print_voronoi_graph(G, seeds):
    distances = {seed: nx.single_source_dijkstra_path_length(G, seed, weight='length') for seed in seeds}

    edge_nearest_seeds = [nearest_for_edge(edge, distances, seeds) for edge in G.edges]
    # Note that edges not connected to a seed shown in black, so invisible on black background
    edge_colours = [colour_for_seed_distance(seed) if seed else 'k' for seed in edge_nearest_seeds]
    # For the road network nodes, we want the seeds to be coloured red and the non-seed nodes to be coloured white.
    node_colours = ['r' if node in seeds else 'w' for node in all_nodes]
    node_sizes = [30 if node in seeds else 0 for node in all_nodes]

    ox.plot.plot_graph(full_leeds_graph, figsize=(20, 20), edge_color=edge_colours, node_size=node_sizes,
                       node_color=node_colours, bgcolor='k', save=True, filepath='nvd.png')


print_voronoi_graph(full_leeds_graph, seeds)

# Load the OSMnx graph data for your area of interest
G = full_leeds_graph

# Create a list of all nodes in the graph
all_nodes = list(G.nodes())
# Randomly select the first seed node
seeds = [np.random.choice(all_nodes)]

# Continue randomly selecting seeds until we have enough
while len(seeds) < NUMBER_OF_SEEDS:
    # Calculate the minimum distance from each node to the existing seed nodes
    distances = []
    for node in all_nodes:
        min_distance = np.inf
        for seed in seeds:
            distance = ox.distance.euclidean_dist_vec(G.nodes[node]['x'], G.nodes[node]['y'], G.nodes[seed]['x'],
                                                      G.nodes[seed]['y'])
            min_distance = min(min_distance, distance)
        distances.append(min_distance)

    # Select the node with the maximum minimum distance as the next seed node
    new_seed = all_nodes[np.argmax(distances)]
    seeds.append(new_seed)

# Print the seed nodes
print(seeds)

print_voronoi_graph(full_leeds_graph, seeds)

# Calculate the node degree for each node
node_degrees = {}
for node in all_nodes:
    node_degrees[node] = len(list(G.neighbors(node)))

# Sort the nodes by degree in descending order
sorted_nodes = sorted(all_nodes, key=lambda node: node_degrees[node], reverse=True)

# Select the top num_seeds nodes by degree as the seed nodes
seeds = sorted_nodes[:NUMBER_OF_SEEDS]

# Print the seed nodes
print(seeds)

print_voronoi_graph(full_leeds_graph, seeds)

# Randomly select the first seed node
seeds = [np.random.choice(all_nodes)]

# Continue randomly selecting seeds until we have enough
while len(seeds) < NUMBER_OF_SEEDS:
    # Calculate the minimum distance from each node to the existing seed nodes
    dist = {}
    for seed in seeds:
        dist[seed] = nx.single_source_dijkstra_path_length(G, seed, weight='distance')
    min_distances = np.inf * np.ones(len(all_nodes))
    for seed, d in dist.items():
        for i, node in enumerate(all_nodes):
            if node in d:
                min_distances[i] = min(min_distances[i], d[node])

    # Select the node with the maximum minimum distance as the next seed node
    new_seed = all_nodes[np.argmax(min_distances)]
    seeds.append(new_seed)

print_voronoi_graph(full_leeds_graph, seeds)

def optmize_voronoi(G):
    # select 10 random seeds
    seeds = list(np.random.choice(list(G.nodes()), size=10, replace=False))
    # generate Voronoi regions for each seed
    converged = False
    count = 0
    while not converged:
        regions = {}
        # for seed in seeds:

        nodes = list(G.nodes())
        distances = {seed: nx.single_source_dijkstra_path_length(G, seed, weight='length') for seed in seeds}

        for seed in seeds:
            regions[seed] = []

        for node in nodes:

            seed_distances = [(s, distances[s][node]) for s in seeds if node in distances[s]]
            seed_distances.sort(key=lambda i: i[1])

            if len(seed_distances) > 0:
                closest_seed = seed_distances[0][0]

                if closest_seed in regions.keys():
                    regions[closest_seed].append(node)
                else:
                    regions[closest_seed] = [node]

        subgraphs = {}
        # Iterate over the keys in the dictionary
        for seed in regions:
            # Create the subgraph using the nodes corresponding to the seed
            nodes = regions[seed]
            subgraph = G.subgraph(nodes)
            # Add the subgraph to the list
            subgraphs[seed] = subgraph

        spatial_data = []
        distance_data = []

        for seed in subgraphs.keys():
            total_distance = 0

            for u, v, k, data in subgraphs[seed].edges(keys=True, data=True):
                total_distance += data["length"]

            spatial_data.append((subgraphs[seed], total_distance, seed))
            distance_data.append(total_distance)

        if count % 2 == 0:
            spatial_data.sort(key=lambda i: i[1], reverse=True)
        else:
            spatial_data.sort(key=lambda i: i[1])

        largest_graph_seed = spatial_data[0][2]
        largest_graph = subgraphs[largest_graph_seed]

        # Extract the nodes' coordinates
        lats, lons = zip(*[(data["y"], data["x"]) for node, data in largest_graph.nodes(data=True)])

        # Calculate the centroid of the nodes' coordinates
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)

        distances_from_center = {}
        for i, node in enumerate(largest_graph.nodes):
            lat, lon = lats[i], lons[i]
            distance = haversine(center_lat, center_lon, lat, lon)
            distances_from_center[node] = distance

        # Find the node with the minimum distance to the centroid
        closest_node = min(distances_from_center, key=distances_from_center.get)

        stdev = statistics.stdev(distance_data)
        median = statistics.median(distance_data)

        print("Median: {}".format(median))
        print("Standard Deviation: {}".format(stdev))
        print("divisor: {}".format(stdev / median))

        if stdev / median < 0.24:
            converged = True
            return seeds

        seeds.remove(largest_graph_seed)
        seeds.append(closest_node)
        count += 1

optmized_seeds = optmize_voronoi(G)

# select 10 random seeds
seeds = list(np.random.choice(list(G.nodes()), size=10, replace=False))
# generate Voronoi regions for each seed
converged = False
count = 0
while not converged:
    regions = {}
    #for seed in seeds:

    nodes = list(G.nodes())
    distances = {seed: nx.single_source_dijkstra_path_length(G, seed, weight='length') for seed in seeds}

    for seed in seeds:
        regions[seed] = []


    for node in nodes:

        seed_distances = [(s, distances[s][node]) for s in seeds if node in distances[s]]
        seed_distances.sort(key = lambda i:i[1])

        if len(seed_distances) > 0:
            closest_seed = seed_distances[0][0]

            if closest_seed in regions.keys():
                regions[closest_seed].append(node)
            else:
                regions[closest_seed] = [node]

    subgraphs = {}
    # Iterate over the keys in the dictionary
    for seed in regions:
        # Create the subgraph using the nodes corresponding to the seed
        nodes = regions[seed]
        subgraph = G.subgraph(nodes)
        # Add the subgraph to the list
        subgraphs[seed] = subgraph

    saved_subgraph = subgraph

    converged = True

print(saved_subgraph)


def find_marathons(some_subgraph):
    for node in some_subgraph.nodes():
        neighbors = some_subgraph.neighbors(node)
        for neighbor in neighbors:
            print(neighbor)


print([(data["y"], data["x"]) for node, data in largest_graph.nodes(data=True)])

find_marathons(saved_subgraph)