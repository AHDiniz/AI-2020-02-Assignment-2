#!/usr/bin/python3.8

from oner import OneREstimator
from stages import first_stage, second_stage
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

iris_dataset = load_iris()
digits_dataset = load_digits()
wine_dataset = load_wine()
breast_cancer_dataset = load_breast_cancer()

# Estimators to be passed in the firt stage:
zero_r = DummyClassifier(strategy = 'most_frequent')
random = DummyClassifier(strategy = 'random')
stratified = DummyClassifier(strategy = 'stratified')
naive_bayes = GaussianNB()
one_r = OneREstimator()


