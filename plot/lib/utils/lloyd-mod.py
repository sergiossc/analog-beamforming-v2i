import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import uuid

# Point class
class Point:
    def __init__(self):
        self.posx = np.random.rand()
        self.posy = np.random.rand()
        self.id = uuid.uuid4()
        self.cluster_id = '' 
        self.d_min = -1

    def get_coordinates(self):
        return np.array([self.posx, self.posy])

# Cluster class
class Cluster:
    def __init__(self):
        self.posx = np.random.rand()
        self.posy = np.random.rand()
        self.id = uuid.uuid4()
        self.points_id_list = []

    def set_coordinates(self, posx, posy):
        self.posx = posx
        self.posy = posy
        self.points_id_list = []

    def get_coordinates(self):
        return np.array([self.posx, self.posy])

# Setting the set point
def set_points(n):
    points = {}
    for i in range(n):
        p = Point()       
        points[p.id] = p
    return points

# Setting the set point
def set_clusters(k):
    clusters = {}
    for i in range(k):
        c = Cluster()       
        clusters[c.id] = c
    return clusters

def set_clusters_from_points(points, k):
    points_list = [p for p_id, p in points.items()]
    indexes_points = np.random.choice(len(points_list), k, replace=False)
    print(indexes_points)
    clusters = {}
    for i in indexes_points:
        p = points_list[i]
        c = Cluster()
        c.posx = p.posx
        c.posy = p.posy
        clusters[c.id] = c
    return clusters

def plot_points(points, clusters):
    for c_id, c in clusters.items():
        if len(c.points_id_list) > 0:
            p_xcoordinates = [points[p_id].posx for p_id in c.points_id_list]
            p_ycoordinates = [points[p_id].posy for p_id in c.points_id_list]
            plt.plot(p_xcoordinates, p_ycoordinates, '*') 
            plt.plot(c.posx, c.posy, '>') 
    plt.show()

def get_sum_distance(points, clusters):
    sum_d = 0
    for c_id, c in clusters.items():

        if len(c.points_id_list) > 0:
            sum_d =+ np.sum(points[p_id].d_min for p_id in c.points_id_list)
    return sum_d

n = 100
k = 5
n_iteractions = 10

print('>>>> Getting a random set of points...')
points = set_points(n)

print('>>>> Getting random initial clusters...')
##clusters = set_clusters(k)
clusters = set_clusters_from_points(points, k)

sum_distance = np.zeros(n_iteractions)

print('**** Iteraction number: ', 0)
print('>>>> Setting points to an specific cluster...')
# getting the closest cluster from each point
for p_id, p in points.items():
    d_min, c_id = min([(norm(p.get_coordinates() - c.get_coordinates()), c_id) for c_id, c in clusters.items()])
    p.cluster_id = c_id
    p.d_min = d_min
    clusters[c_id].points_id_list.append(p_id)

# getting mean distance by clusters and its points
sum_distance[0] = get_sum_distance(points, clusters)

plot_points(points, clusters) 

for i in range(1, n_iteractions):
    print('**** Iteraction number: ', i)

    for c_id, c in clusters.items():

        if len(c.points_id_list) > 0:
            p_xcoordinates = [points[p_id].posx for p_id in c.points_id_list]
            p_ycoordinates = [points[p_id].posy for p_id in c.points_id_list]
            new_c_posx = 1/(len(c.points_id_list)) * np.sum(p_xcoordinates)
            new_c_posy = 1/(len(c.points_id_list)) * np.sum(p_ycoordinates)
            c.set_coordinates(new_c_posx, new_c_posy)
 
    for p_id, p in points.items():
        d_min, c_id = min([(norm(p.get_coordinates() - c.get_coordinates()), c_id) for c_id, c in clusters.items()])
        p.cluster_id = c_id
        p.d_min = d_min
        clusters[c_id].points_id_list.append(p_id)

   
    sum_distance[i] = get_sum_distance(points, clusters) 
    plot_points(points, clusters) 
print('sum_distance: ', sum_distance)

plt.plot(sum_distance, '-')
plt.show()
