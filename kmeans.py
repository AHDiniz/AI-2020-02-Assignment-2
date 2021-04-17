#!/usr/bin/python3.8

import random
import numpy as np
from sklearn.base import BaseEstimator

class KMeansClassifier(BaseEstimator):
    def __init__(self, k):
        self.__k = k
    
    def fit(self, X, Y):
        print("KMeans fitting")
    
    def predict(self, X):
        return None
