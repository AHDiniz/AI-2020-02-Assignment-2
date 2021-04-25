#!/usr/bin/python3.8

import random
import numpy as np
from kcentroids import KCentroidsBaseEstimator
from sklearn.cluster import KMeans

class KMeansClassifier(KCentroidsBaseEstimator):
    
    # Implementing the clustering method with a naive k-means algorithm:
    def clustering(self, points):
        km = KMeans(n_clusters = self.k)
        km.fit(points)
        return km.cluster_centers_
