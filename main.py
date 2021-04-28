#!/usr/bin/python3.8

from numpy import array, concatenate

from oner import OneRClassifier

from stages import first_stage, second_stage

from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from kmeans import KMeansClassifier

from kga import GeneticClassifier

from pandas import DataFrame, Series

from seaborn import boxplot

from matplotlib.pyplot import savefig, close

from scipy.stats import ttest_rel, wilcoxon

# Storing the datasets:
datasets = {'Iris': load_iris(), 'Wine': load_wine(), 'Breast Cancer': load_breast_cancer(), 'Digits': load_digits()}

# Estimators to be passed in the firt stage:
zero_r = DummyClassifier(strategy = 'most_frequent')
rnd = DummyClassifier(strategy = 'uniform')
stratified = DummyClassifier(strategy = 'stratified')
naive_bayes = GaussianNB()
one_r = OneRClassifier()

first_estimators = {'0R': zero_r, 'RND': rnd, 'STR': stratified, 'NBA': naive_bayes, '1R': one_r}

# Scores dictionary:
scores = {'Iris': dict({}), 'Wine': dict({}), 'Breast Cancer': dict({}), 'Digits': dict({})}

# Applying the first stage procedures to the estimators with no hyperparameters:
for estimator_name, estimator in first_estimators.items():
    use_discretizer = False # Will be set to True when using the One-R estimator
    
    if estimator_name == '1R':
        use_discretizer = True
    
    # Applying each dataset to the estimator:
    for dataset_name, dataset in datasets.items():
        result, mean, deviation, inf, sup = first_stage(dataset, estimator, use_discretizer)
        scores[dataset_name][estimator_name] = result

        # Printing statistical values related to the stage execution:
        print(estimator_name, dataset_name)
        print('Accuracy Mean:', mean, 'Accuracy Standard Deviation:', deviation)
        print('Accuracy Trust Interval (95%%): (', inf, ',', sup, ')')
    print()

# Estimators to be passed in the second stage and their hyperparameters grid:
knn = (KNeighborsClassifier(weights = 'uniform'), {'estimator__n_neighbors': [1, 3, 5, 7]})
dist_knn = (KNeighborsClassifier(weights = 'distance'), {'estimator__n_neighbors': [1, 3, 5, 7]})
decision = (DecisionTreeClassifier(), {'estimator__max_depth': [None, 3, 5, 10]})
forest = (RandomForestClassifier(), {'estimator__n_estimators': [10, 20, 50, 100]})
kmeans = (KMeansClassifier(), {'estimator__k': [1, 3, 5, 7]})
genetic = (GeneticClassifier(), {'estimator__k': [1, 3, 5, 7]})

#second_estimators = {'UKN': knn, 'DKN': dist_knn, 'DTR': decision, 'DFR': forest}
second_estimators = {'UKN': knn, 'DKN': dist_knn, 'DTR': decision, 'DFR': forest, 'KMC': kmeans, 'GAC': genetic}

for estimator_name, estimator_data in second_estimators.items():
    estimator = estimator_data[0]
    param_grid = estimator_data[1]

    for dataset_name, dataset in datasets.items():
        result, mean, deviation, inf, sup = second_stage(dataset, estimator, param_grid)
        scores[dataset_name][estimator_name] = result

        # Printing statistical values related to the stage execution:
        print(estimator_name, dataset_name)
        print('Accuracy Mean:', mean, 'Accuracy Standard Deviation:', deviation)
        print('Accuracy Trust Interval (95%%): (', inf, ',', sup, ')')
    print()

for dataset_name, dataset in scores.items():
    # Plotting the boxplot
    dataframe = DataFrame(dict([(k, Series(v)) for k, v in dataset.items()]))
    boxplot(data = dataframe)
    savefig(dataset_name + '.png')
    close()

    print()
    print("Paired test in the", dataset_name, "dataset:")
    print()

    # Printing the paired tests data:
    for a_name, a_score in scores[dataset_name].items():
        used = list([])
        for b_name, b_score in scores[dataset_name].items():
            if not a_name == b_name and not b_name in used:
                print("T-Paired test with", a_name, "/", b_name, "=", ttest_rel(a_score, b_score))
                print("Wilcoxon test with", a_name, "/", b_name, "=", wilcoxon(a_score, b_score))
                used.append(b_name)
