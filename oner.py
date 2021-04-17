#!/usr/bin/python3.8

import random
import numpy as np
from sklearn.base import BaseEstimator

class OneRClassifier(BaseEstimator):
    
    # Training the One Rule estimator:
    def fit(self, X, Y):
        dimensions = np.shape(X)
        unique = np.unique(Y)
        self._num_classes = len(unique)

        # Creating the contingency table:
        contingency_tables = []
        # For each parameter:
        for i in dimensions[1]:
        # Create list of possible values for each parameter:
            possible_values = np.unique(X[:i])
            contingency_table = dict({})
            # For each possible value:
            for possible_value in possible_values:
                # Get indices of points where current parameter is equal to current possible value:
                indices = np.where(X[:i] == possible_value)
                # Get classes in those indices:
                classes = np.take(Y, indices)
                # Calculate amount of elements in each class:
                _, contingency_list = np.unique(classes, return_counts=True)
                contingency_table[possible_value] = contingency_list
            contingency_tables.append(contingency_table)

        self._best_parameter = -1 # Will hold the value of the best parameter to be used in the predictions
        best_sum = 0
        # For each contingency table:
        for i in range(len(contingency_tables)):
            # Get the sum of each class occurance for each parameter value:
            sum_in_paramenter = np.sum(np.apply_along_axis(lambda x: np.amax(x), 0, contingency_tables[i]))
            # Choosing the best parameter:
            if sum_in_paramenter > best_sum:
                best_sum = sum_in_paramenter
                self._best_parameter = i

        # Calculating the probabilties for each value:
        self._probability_table = dict({})
        self._possible_values = np.unique(X[:(self._best_parameter)])
        for possible_value in self._possible_values:
            self._probability_table[possible_value] = np.apply_along_axis(lambda x: x / best_sum, 0, contingency_tables[self._best_parameter][possible_value])

    def predict_value(self, x):
        for possible_value in self._possible_values:
            if x == possible_value:
                return np.random.choice(np.array(range(self._num_classes)), p = self._probability_table[possible_value])

    def predict(self, X):
        return np.apply_along_axis(self.predict_value, self._best_parameter, X)
