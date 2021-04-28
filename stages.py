#!/usr/bin/python3.8

from scipy import stats
from numpy import sqrt, unique
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

# Function used to test the methods that don't use hyperparameters:
def first_stage(dataset, estimator, use_discretizer = False):
    scaler = StandardScaler()

    x = dataset['data']
    y = dataset['target']
    
    rfsk = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)

    score = None
    pipeline = None

    if use_discretizer: # This is used for the One Rule method
        discretizer = KBinsDiscretizer(2 * len(unique(dataset['target'])), encode = 'ordinal', strategy = 'kmeans')
        pipeline = Pipeline([('transformer', scaler), ('discretizer', discretizer), ('estimator', estimator)])
    else:
        pipeline = Pipeline([('transformer', scaler), ('estimator', estimator)])

    # Executing the test:
    scores = cross_val_score(pipeline, x, y, scoring = 'accuracy', cv = rfsk)

    # Getting statistical data:
    mean = scores.mean()
    deviation = scores.std()
    inf, sup = stats.norm.interval(0.95, loc = mean, scale = deviation / sqrt(len(scores)))

    return (scores, mean, deviation, inf, sup)

# Used for the methods that use hyperparameters:
def second_stage(dataset, estimator, param_grid):
    x = dataset['data']
    y = dataset['target']

    inner_cv = StratifiedKFold(n_splits = 4)

    scaler = StandardScaler()
    pipeline = Pipeline([('transformer', scaler), ('estimator', estimator)])

    # Getting the best hyperparameter configuration:
    grid_search = GridSearchCV(estimator = pipeline, param_grid = param_grid, scoring = 'accuracy', cv = inner_cv)
    grid_search.fit(x, y)

    scaler = StandardScaler()
    pipeline = Pipeline([('transformer', scaler), ('clf', grid_search)])

    outer_cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)

    # Making the test:
    scores = cross_val_score(pipeline, x, y, scoring = 'accuracy', cv = outer_cv)

    # Getting statistical data:
    mean = scores.mean()
    deviation = scores.std()
    inf, sup = stats.norm.interval(0.95, loc = mean, scale = deviation / sqrt(len(scores)))

    return (scores, mean, deviation, inf, sup)
