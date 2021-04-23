#!/usr/bin/python3.8

import random
import numpy as np
from kcentroids import KCentroidsBaseEstimator
from sklearn.kmeans import KMeans

class KMeansClassifier(KCentroidsBaseEstimator):
    
    # Implementing the clustering method with a naive k-means algorithm:
    def clustering(self, points):
        km = KMeans(n_clusters = self.__k)
        km.fit(data)
        return km.cluster_centers_
