# src/config.py

# ==================================
#  CONSTANTS
# ==================================
BANK_NAME = '🏦 TERA BANK 🏦'
PROJ_NAME = '💳 CREDIT CARD CHURN PREDICTOR 💳'
DATASET_TYPES = ['original', 'oversampled', 'undersampled']

SAMPLE_FILE = 'data/sample_data.csv'
SEEDER_FILE = 'data/seeder_data.csv'
DATASET_FILE = 'data/dataset.csv'
OUTPUT_FILE = 'outputs/model_comparison_report.txt'

# Customer Churn Rate (16%)
CUST_CHURN_RATE = 0.16

# Specifies the number of base estimators (individual models) that will be trained and combined to form the final
# ensemble model.
BASE_ESTIMATOR_CNT = 100

# Number of folds to use in K-Fold Cross-Validation.
CV_FOLDS = 5

# Sample Data Split
DATA_TEMP_SPLIT = 0.3
HALF_SPLIT = 0.5

# In a Google Colab environment, n_jobs=-1 will use all cores allocated to your
# session, which usually means 2 or more cores, significantly accelerating the
# hyperparameter search compared to the default n_jobs=1 (which uses only one core).
MAX_PROC_THREADS = -1
MSEC = 1000

# Controls the maximum number of levels (nodes) allowed in each individual decision
# tree within the forest.
NODE_RFC_CNT = 4
NODE_XGBOOST_CNT = 3

# Number of different parameter combinations that will be tried.
PARAM_DIST_CNT = 50
PERCENTILE = 100
SECS_IN_MIN = 60
SEED = 42

# Tuning parameters
UNTUNED_ESTIMATOR_CNT = 50
UNTUNED_LEARNING_RATE = 0.05
