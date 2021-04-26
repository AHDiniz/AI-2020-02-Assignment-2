#!/usr/bin/python3.8

import random
import numpy as np
from sklearn.base import BaseEstimator

class OneRClassifier(BaseEstimator):
    
    # Training the One Rule estimator:
    def fit(self, X, Y):
        dimensions = np.shape(X[0])
        unique_classes = np.unique(Y)
        self._num_classes = len(unique_classes)

        # Creating the contingency table:
        contingency_tables = []
        # For each parameter:
        for i in range(dimensions[0]):
        # Create list of possible values for each parameter:
            possible_values = np.unique(X[:,i])
            contingency_table = dict({})
            # For each possible value:
            for possible_value in possible_values:
                # Get indices of points where current parameter is equal to current possible value:
                indices = np.where(X[:,i] == possible_value)
                # Get classes in those indices:
                classes = np.take(Y, indices)
                # Calculate amount of elements in each class:
                contingency_list = []
                detected_classes, classes_count = np.unique(classes, return_counts=True)
                j = 0
                for i in range(self._num_classes):
                    c = unique_classes[i]
                    if c in detected_classes:
                        contingency_list.append(classes_count[j])
                        j += 1
                    else:
                        contingency_list.append(0)
                contingency_table[possible_value] = contingency_list
            contingency_tables.append(contingency_table)

        self._best_parameter = -1 # Will hold the value of the best parameter to be used in the predictions
        best_sum = 0 # Will hold the value of the sums of possible occurrences of the best parameter
        # For each contingency table:
        for i in range(len(contingency_tables)):
            # Get the sum of each class occurance for each parameter value:
            sum_in_paramenter = np.sum(np.apply_along_axis(lambda x: np.amax(x), 1, list(contingency_tables[i].values())))
            # Choosing the best parameter:
            if sum_in_paramenter > best_sum:
                best_sum = sum_in_paramenter
                self._best_parameter = i

        # Calculating the probabilties for each value:
        self._probability_table = dict({})
        self._possible_values = np.unique(X[:,self._best_parameter])
        for possible_value in self._possible_values:
            total_items = np.sum(contingency_tables[self._best_parameter][possible_value])
            self._probability_table[possible_value] = np.apply_along_axis(lambda x: x / total_items, 0, contingency_tables[self._best_parameter][possible_value])


    def predict_value(self, x):
        for p in self._possible_values:
            if x == p:
                return np.random.choice(np.array(range(self._num_classes)), p = self._probability_table[p])

    def predict(self, X):
        Y = []
        for x in X[:,self._best_parameter]:
            Y.append(self.predict_value(x))
        return Y
