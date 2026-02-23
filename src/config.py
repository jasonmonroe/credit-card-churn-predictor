# src/config.py

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
# ==================================
#  CONSTANTS
# ==================================
SAMPLE_FILE = 'data/sample_data.csv'
SEEDER_FILE = 'data/seeder_data.csv'
DATASET_FILE = 'data/dataset.csv'


# Customer Churn Rate (16%)
CUST_CHURN_RATE = 0.16

# Specifies the number of base estimators (individual models) that will be trained
# and combined to form the final ensemble model.
BASE_ESTIMATOR_CNT = 100

# Number of folds to use in K-Fold Cross-Validation.
CV_FOLDS = 5

# Sample Data Split
DATA_TEMP_SPLIT = 0.3
DATA_TRAINING_SPLIT = 0.7
DATA_TESTING_SPLIT = 0.15
DATA_VALIDATION_SPLIT = 0.15
HALF_SPLIT = 0.5

# Number of features to consider when looking for the best split at each node of
# the individual decision trees built within the boosting process.
FEATURES_SPLIT_PCT = 0.5 # baseline for randomness (50%)

# In a Google Colab environment, n_jobs=-1 will use all cores allocated to your
# session, which usually means 2 or more cores, significantly accelerating the
# hyperparameter search compared to the default n_jobs=1 (which uses only one core).
MAX_PROC_THREADS = -1

# Specifies the minimum loss reduction required to make a further partition
# (split) on a leaf node of the tree.
MIN_TREE_SPLIT = 1
MSEC = 1000

# Controls the maximum number of levels (nodes) allowed in each individual decision
# tree within the forest.
NODE_RFC_CNT = 4
NODE_XGBOOST_CNT = 3

# Number of different parameter combinations that will be tried.
PARAM_DISTR_CNT = 50
PERCENTILE = 100
SECS_IN_MIN = 60
SEED = 42

# Hyperparameter that controls the fraction of the training samples used to train
# each individual tree.
# Sample size of training data for Stochastic Gradient Boosting (SGB).
# Increased robustness, lower variance, generally better generalization.
SUB_SAMPLE_SIZE = 0.7

# Tuning parameters
TUNED_ESTIMATOR_CNT = 20
TUNED_LEARNING_RATE = 0.1
UNTUNED_ESTIMATOR_CNT = 50
UNTUNED_LEARNING_RATE = 0.05

GB_PARAMS = {
    "init": [
        AdaBoostClassifier(random_state=SEED),
        DecisionTreeClassifier(random_state=SEED)
        ],
    "n_estimators": np.arange(50, 110, 25),
    "learning_rate": [0.01, 0.1, 0.05],
    "subsample": [0.7, 0.9],
    "max_features": [0.5, 0.7, 1],
}

ADA_BOOST_PARAMS = {
    "n_estimators": np.arange(50, 110, 25),
    "learning_rate": [0.01, 0.1, 0.05],
    "estimator": [
        DecisionTreeClassifier(max_depth=2, random_state=SEED),
        DecisionTreeClassifier(max_depth=3, random_state=SEED),
    ],
}

XGB_BOOST_PARAMS = {
    'n_estimators': np.arange(50, 110, 25),
    'scale_pos_weight': [1, 2, 5],
    'learning_rate': [0.01, 0.1, 0.05],
    'gamma': [1, 3, 5],
    'subsample': [0.7, 0.9]
}
