#!/usr/bin/python3.8

from oner import OneRClassifier

from stages import first_stage, second_stage

from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from pandas import DataFrame

from seaborn import boxplot

from matplotlib.pyplot import savefig

# Storing the datasets:
datasets = {'Iris': load_iris(), 'Wine': load_wine(), 'Breast Cancer': load_breast_cancer(), 'Digits': load_digits()}

# Estimators to be passed in the firt stage:
zero_r = DummyClassifier(strategy = 'most_frequent')
random = DummyClassifier(strategy = 'random')
stratified = DummyClassifier(strategy = 'stratified')
naive_bayes = GaussianNB()
one_r = OneRClassifier()

first_estimators = {'Zero-R': zero_r, 'Random': random, 'Stratified': stratified, 'Naive Bayes': naive_bayes, 'One-R': one_r}

# Applying the first stage procedures to the estimators with no hyperparameters:
for estimator_name, estimator in first_estimators:
    use_discretizer = False # Will be set to True when using the One-R estimator
    scores = dict({}) # Will be used to create the DataFrame to be used in the boxplot
    
    if estimator_name == 'One-R':
        use_discretizer = True
    
    # Applying each dataset to the estimator:
    for dataset_name, dataset in datasets:
        result, mean, deviation, inf, sup = first_stage(dataset, estimator, use_discretizer)
        scores[dataset_name] = result

        # Printing statistical values related to the stage execution:
        print(estimator_name, dataset_name)
        print('Accuracy Mean:', mean, 'Accuracy Standard Deviation:', deviation)
        print('Accuracy Trust Interval (95%%): (', inf, ',', sup, ')')
    
    # Plotting the boxplot
    dataframe = DataFrame(scores)
    boxplot(data = dataframe)
    savefig(estimator_name + '.png')

# Estimators to be passed in the second stage and their hyperparameters grid:
knn = (KNeighborsClassifier(weights = 'uniform'), {'estimator__n_neighbors': [1, 3, 5, 7]})
dist_knn = (KNeighborsClassifier(weights = 'distance'), {'estimator__n_neighbors': [1, 3, 5, 7]})
decision = (DecisionTreeClassifier(), {'estimator__max_depth': [None, 3, 5, 10]})
forest = (RandomForestClassifier(), {'estimator__n_estimators': [10, 20, 50, 100]})
# ! Create k-Means Estimator here
# ! Create Genetic Estimator here

second_estimators = {'Uniform k-Neighbors': knn, 'Distance k-Neighbors': dist_knn, 'Decision Tree': decision, 'Decision Forest': forest}

for estimator_name, estimator_data in second_estimators:
    estimator = estimator_data[0]
    param_grid = estimator_data[1]
    scores = dict({})

    for dataset_name, dataset in datasets:
        result, mean, deviation, inf, sup = second_stage(dataset, estimator, param_grid)
        scores[dataset_name] = result

        # Printing statistical values related to the stage execution:
        print(estimator_name, dataset_name)
        print('Accuracy Mean:', mean, 'Accuracy Standard Deviation:', deviation)
        print('Accuracy Trust Interval (95%%): (', inf, ',', sup, ')')
    
    # Plotting the boxplot
    dataframe = DataFrame(scores)
    boxplot(data = dataframe)
    savefig(estimator_name + '.png')
