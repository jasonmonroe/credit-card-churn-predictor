# src/modeling.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import recall_score, make_scorer, precision_score
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.config import *
from src.utils import (
    start_timer,
    show_banner,
    show_timer,
    model_performance_classification_sklearn,
    plot_confusion_matrix
)


def build_models():
    models = [
        ('Bagging', bagging_model()),
        ('Random forest', random_forest_model()),
        ('AdaBoost', ada_boost_model()),
        ('Gradient Boosting', gradient_boosting_model()),
        ('XGBoost', xgboost_model())
    ]

    return models

def bagging_model():
    return BaggingClassifier(
        random_state=SEED,
        n_estimators=BASE_ESTIMATOR_CNT)


def random_forest_model():
    return RandomForestClassifier(
        max_depth=NODE_RFC_CNT,
        n_estimators=UNTUNED_ESTIMATOR_CNT,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt')


def ada_boost_model():
    return AdaBoostClassifier(random_state=SEED)

def gradient_boosting_model():
    return GradientBoostingClassifier(random_state=SEED)

def xgboost_model():
    return XGBClassifier(
        n_estimators=UNTUNED_ESTIMATOR_CNT,
        max_depth=NODE_XGBOOST_CNT,
        learning_rate= UNTUNED_LEARNING_RATE,
        reg_alpha=0.3,
        reg_lambda=0.3)


def show_fit_model_scores(
    mods: list,
    x_fit_data: pd.DataFrame,
    y_fit_data: pd.Series,
    x_data: pd.DataFrame,
    y_data: pd.Series
    ) -> None:

    for name, model in mods:
        model.fit(x_fit_data, y_fit_data)
        scores_val = recall_score(y_data, model.predict(x_data))
        print("{}: {}".format(name, scores_val))


def show_classification_model_perf(mods, x_data: pd.DataFrame, y_data: pd.Series):
    for name, model in mods:
        df_perf = model_performance_classification_sklearn(model, x_data, y_data)
        print(df_perf)


def run_model_performance(
    mods: list,
    x_fit_data: pd.DataFrame,
    y_fit_data: pd.Series,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    title: str,
    section: str='',
    show_classify: bool=False
    ) -> None:

    """
    mods: - list of models

    data_y: dependent variable
    data_x: independent variables
    section: str - title of the section
    show_classify: bool - determines which performance function to run

    Starts time for benchmarking, displays a banner for readability, and shows model performance.
    """
    start_time = start_timer()
    show_banner(title, section)

    if show_classify:
        show_classification_model_perf(mods, x_data, y_data)
    else:
        show_fit_model_scores(mods, x_fit_data, y_fit_data, x_data, y_data)

    show_timer(start_time)


def oversample_data(x_training_data, y_training_data):
    # Synthetic Minority Over Sampling Technique
    sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=SEED)
    x_training_oversample, y_training_oversample = sm.fit_resample(x_training_data, y_training_data)

    return x_training_oversample, y_training_oversample


def undersample_data(x_training_data, y_training_data):
    # Random under sampler for under sampling the data
    rus = RandomUnderSampler(sampling_strategy=1, random_state=SEED)
    x_training_undersample, y_training_undersample = rus.fit_resample(x_training_data, y_training_data)

    return x_training_undersample, y_training_undersample


def pick_top_model(xgb_model_scores: pd.DataFrame, xgb_models: list) -> XGBClassifier :
    """
    xgb_models: pd.DataFrame

    Compares the three XGBoost models and returns the best one.
    """
    f1_scores = []

    # Get F1 Scores
    for model in xgb_model_scores.columns:
        f1_scores.append(xgb_model_scores[model]['F1'])

    # Get index and variable of the top F1 score
    top_m_index = f1_scores.index(max(f1_scores))
    top_m_title = xgb_model_scores.columns[top_m_index]
    top_m = xgb_models[top_m_index]

    show_banner('TOP MODEL', top_m_title)
    print(xgb_model_scores[top_m_title]) # Fixed line: Use top_m_title (string) instead of top_m_index (integer)

    return top_m