#!/usr/bin/python3.8

import math
import random
import time
import copy
import numpy as np

# Calculating the distance between two points:
def euclidian_dist(a : np.array, b : np.array) -> float:
    result : float = 0
    aux : np.array = a - b
    aux = aux ** 2
    result = np.sum(aux)
    return math.sqrt(result)

# Class that will hold data about the clusters:
class Clusters:
    def __init__(self, k : int, data : np.array):
        self._k : int = k
        self._point_dim : tuple = (1, data.shape[1])
        
        # Setting up the points array:
        self._points_data = np.array(data)
        self._num_points : int = self._points_data.shape[0]
        np.reshape(self._points_data, (self._num_points, self._point_dim[1]))

        # Creating a random seed:
        random.seed(time.thread_time_ns())

        # Initializing the data arrays:
        self._clusters : np.array = np.zeros(self._num_points, dtype=int)
        self._centroids = list([])
        for i in range(self._k):
            self._centroids.append(np.zeros(self._point_dim))
        
        # Initializing the distubed data arrays:
        self._disturbed_clusters : np.array = np.zeros(self._num_points, dtype=int)
        self._disturbed_centroids = list([])
        for i in range(self._k):
            self._disturbed_centroids.append(np.zeros(self._point_dim))
        
        # Initializing the cost of the regular state and the disturbed state:
        self._sse : float = 0
        self._disturbed_sse : float = 0

        # Initializing the list of sse's for each cluster:
        self._sse_per_cluster : list = []
        for i in range(self._k):
            self._sse_per_cluster.append(0.)

    # Initializing the state of the clusters given an array of centroids:
    def initialize_state(self, centroids : list = None):
        if centroids != None:
            self.centroids = centroids
        else:
            # Calculating the initial values of the centroids:
            for i in range(self._k):
                index = np.random.choice(self._num_points, size=1, replace=False)[0]
                self._centroids[i] = self._points_data[index]

        for i in range(self._num_points):
            dist : float = np.inf
            closest : int = -1
            for j in range(self._k):
                d : float = euclidian_dist(self._points_data[i], self._centroids[j])
                if d < dist:
                    dist = d
                    closest = j
            self._clusters[i] = closest
        
        for i in range(self._k):
            self.calculate_sse_in_cluster(i, self._clusters, self._centroids)
        
        self._disturbed_sse = 0
        self._disturbed_clusters : np.array = np.zeros(self._num_points, dtype=int)
        self._disturbed_centroids = list([])
        for i in range(self._k):
            self._disturbed_centroids.append(np.zeros(self._point_dim))

        self._sse = self.calculate_sse()

    # Getting the number of clusters:
    def get_k(self) -> int:
        return self._k
    
    # Returning the dimension of a single point:
    def get_point_dim(self) -> tuple:
        return self._point_dim

    # Returning the cluster identifier of each point:
    def get_clusters(self) -> np.array:
        return self._clusters

    # Returning the cluster identifier array in the disturbed state:
    def get_disturbed_clusters(self) -> np.array:
        return self._disturbed_clusters

    # Returning the points data in the clusters:
    def get_points(self) -> np.array:
        return self._points_data
    
    # Returning the list of centroids of each cluster:
    def get_centroids(self) -> list:
        return self._centroids
    
    # Reseting the list of centroids of each cluster:
    def set_centroids(self, centroids : list):
        for i in range(self._k):
            self._centroids[i] = centroids[i]
    
    # Returning the list of centroids of each cluster in the disturbed state:
    def get_disturbed_centroids(self) -> list:
        return self._disturbed_centroids
    
    # Returning the number of points in the set:
    def get_num_points(self) -> int:
        return self._num_points
    
    # Getting a single point:
    def get_point(self, point_index : int = 0) -> np.array:
        return self._points_data[point_index]
    
    # Getting array of points that are in a given cluster:
    def get_points_in_cluster(self, cluster : int, clusters : np.array) -> np.array:
        indices : list = list([])
        for i in range(self._num_points):
            if clusters[i] == cluster:
                indices.append(i)
        result : np.array = np.take(self.points, indices, axis = 0)
        return result
    
    # Getting the number of points in a given cluster:
    def get_num_in_cluster(self, cluster : int, clusters : np.array) -> int:
        indices : list = list([])
        for i in range(self._num_points):
            if clusters[i] == cluster:
                indices.append(i)
        return len(indices)
    
    # Changing the cluster of a given point:
    def move_point(self, point_index : int = 0, cluster_id : int = 0):
        self._clusters[point_index] = cluster_id
    
    # Calculating the centroid (mean point) of the desired cluster:
    def cluster_centroid(self, cluster_id : int, clusters : np.array) -> np.array:
        centroid : np.array = np.zeros(self._point_dim[1]) # Initializing the centroid variable

        # Getting the points that are inside the target cluster:
        points : np.array = self.get_points_in_cluster(cluster_id, clusters)
        num_points : int = points.shape[0]

        # Calculating the centroid coordinates:
        for point in points:
            centroid += point / num_points
        
        return centroid
    
    # Getting the sum of squares of euclidian distances in the regular state:
    def get_sse(self) -> float:
        return self._sse
    
    # Getting the sum of squares of euclidian distances in the disturbed state:
    def get_disturbed_sse(self) -> float:
        return self._disturbed_sse

    # Calculating the sse in a given cluster:
    def calculate_sse_in_cluster(self, c : int, clusters : np.array, centroids : list):
        points : np.array = self.get_points_in_cluster(c, clusters)
        centroids[c] = self.cluster_centroid(c, clusters)
        for point in points:
            self._sse_per_cluster[c] += euclidian_dist(point, centroids[c]) ** 2

    # Calcuting the sum of squares of euclidian distances:
    def calculate_sse(self) -> float:
        result : float = 0
        for c in range(self._k):
            result += self._sse_per_cluster[c]
        return result

    # Getting the point in point_set that is furthest from c
    def furthest_in_set(self, point_set : np.array, c : np.array) -> int:
        f = lambda p : euclidian_dist(p, c)
        dist : float = 0
        max_index : int = 0
        for i in range(self.get_num_in_cluster(c, self._clusters)):
            d : float = f(point_set[i])
            if d >= dist:
                max_index = i
                dist = d
        return max_index

    def accept_disturbed(self):
        self._clusters = copy.deepcopy(self._disturbed_clusters)
        self._centroids = copy.deepcopy(self._disturbed_centroids)
        self._sse = self._disturbed_sse

    # Disturbing the current state to create a possible neighbour:
    def disturb(self):
        # Disturbance will be achieved by selecting a random cluster and removing the furthest point to the cluster with closest centroid
        cluster : int =  random.randint(0, self._k - 1)
        points : np.array = self.get_points_in_cluster(cluster, self._clusters)
        index : int = self.furthest_in_set(points, cluster)
        dist : float = np.inf
        closest : int = -1
        for c in range(self._k):
            d : float = euclidian_dist(self._points_data[index], self._centroids[c])
            if d < dist:
                dist = d
                closest = c
        self._disturbed_clusters[index] = closest
        self._disturbed_centroids[closest] = self.cluster_centroid(closest, self._disturbed_clusters)
        self._disturbed_centroids[cluster] = self.cluster_centroid(cluster, self._disturbed_clusters)
        self.calculate_sse_in_cluster(closest, self._disturbed_clusters, self._disturbed_centroids)
        self.calculate_sse_in_cluster(cluster, self._disturbed_clusters, self._disturbed_centroids)
        self._disturbed_sse = self.calculate_sse()

    k = property(get_k)
    point_dim = property(get_point_dim)
    points = property(get_points)
    num_points = property(get_num_points)
    sse = property(get_sse)
    disturbed_sse = property(get_disturbed_sse)
    clusters = property(get_clusters)
    disturbed_clusters = property(get_disturbed_clusters)
    centroids = property(get_centroids, set_centroids)
    disturbed_centroids = property(get_disturbed_centroids)