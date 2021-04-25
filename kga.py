#!/usr/bin/python3.8

import random
import numpy as np
from kcentroids import KCentroidsBaseEstimator
from clustering import Clusters
from genetic import genetic, HyperParams

class GeneticClassifier(KCentroidsBaseEstimator):
    
    # Implementing the clustering method with a genetic algorithm:
    def clustering(self, points):
        params = HyperParams(10, .2, .95)
        init_state = Clusters(self.k, points)
        _, result = genetic(params, init_state)
        return result.centroids
