#!/usr/bin/python3.8

import numpy as np
from numpy.linalg import norm
from sklearn.base import BaseEstimator    

class KCentroidsBaseEstimator(BaseEstimator):

    def __init__(self, k=1):
        self.k = k

    # This must be implemented in the classes that extend this one
    def clustering(self, points):
        return None

    # Fitting the k-Centroids Base Estimator
    def fit(self, X, Y):
        # Getting the points in each class:
        self.__classes = np.unique(Y)
        points_by_classes = dict({})
        for c in self.__classes:
            indices = np.where(Y == c)
            points_by_classes[c] = np.take(X, indices[0], axis=0)

        # Applying the clustering algorithm for each points list and saving the centroids list:
        self.__centroids = dict({})
        for c, points in points_by_classes.items():
            self.__centroids[c] = self.clustering(points)

    def predict(self, X):
        # Prediction is made by getting the class that contains the centroid closest to the point being classified:
        Y = []
        for point in X:
            distances = dict({})
            for c in self.__classes:
                distances[c] = min([norm(centroid - point) for centroid in self.__centroids[c]])
            Y.append(min(distances))
        return Y
