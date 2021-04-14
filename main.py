#!/usr/bin/python3.8

import sklearn as skl

iris_dataset = skl.datasets.load_iris()
digits_dataset = skl.datasets.load_digits()
wine_dataset = skl.datasets.load_wine()
breast_cancer_dataset = skl.datasets.load_breast_cancer()

# Algorithms to be used: Guassian Naive Bayes, k-Nearest Neighbours, Distance k-Nearest Neighbours, Decision Tree, Decision Forest, Uniform Random, Stratified Random
# Algorithms to be implemented: One Rule, k-Means Centroids, Genetic Algorithm Clustering (extend from BaseEstimator)

# First stage:
#   RepeatedStratifiedKFold (n_splits = 10, n_repeats = 3)
#   GridSearchCV
#   cross_val_score
#   mean
#   standard deviation
#   accuracy confidence interval

# Second stage:
#   RepeatedStratifiedKFold (n_splits = , n_repeats = )
#   GridSearchCV
#   cross_val_score
#   mean
#   standard deviation
#   accuracy confidence interval
