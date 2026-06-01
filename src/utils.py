# src/utils.py

import time
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)
from src.config import *

# ==================================
#  HELPER FUNCTIONS
# ==================================

# --- Functions --- #
def get_run_id() -> str:
    """ Generates a unique ID for the current run. """
    return ' --- RUN ID: ' + str(random.randint(10000, 99999)) + '---'

def start_timer() -> float:
    """
    Start a timer
    """
    return time.time()

def get_time(start_time_float: float) -> str:
    diff = abs(time.time() - start_time_float)
    _, remainder = divmod(diff, SECS_IN_MIN*SECS_IN_MIN)
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)

    ms = fractional_seconds * MSEC
    return f"{int(minutes)}m {int(seconds)}s {int(ms)}ms"

def show_timer(start_time_int: float) -> None:
    print(f"Run Time: {get_time(start_time_int)}")

def show_banner(title: str, section: str = '') -> None:
    """Prints a stylized banner for console readability."""
    padding = 4
    strlen = len(title) + padding
    line = '+-' + '-' * strlen + '-+'

    print('')
    print(line)
    print('|  ' + title.upper() + '  |')
    print(line)

    if section:
        print('| ' + section)

    print('')

# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target) -> pd.DataFrame:
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred) # to compute Accuracy
    recall = recall_score(target, pred) # to compute Recall
    precision = precision_score(target, pred) # to compute Precision
    f1 = f1_score(target, pred) # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1

        },
        index=[0],
    )

    return df_perf
