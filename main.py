# src/main.py

import pandas as pd
import numpy as np
from src.config import *
from src.eda import run_eda
from src.modeling import (
    split_data,
    impute_missing_values,
    encode_data,
    build_model,
    run_model_performance,
    oversample_data,
    undersample_data,
    pick_top_model,
    model_performance_classification_sklearn,
    plot_confusion_matrix
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, precision_score

def main():
    # Load data
    data = pd.read_csv(CSV_FILE)

    # Create a copy of the data (that will be used later)
    df = data.copy()

    # Remove "N/A" from Education Level, "NaN" from Marital Status and "abc" from Income Category
    df['education_level'] = df['education_level'].replace('N/A', np.nan)
    df['marital_status'] = df['marital_status'].replace('NaN', np.nan)
    df['income_category'] = df['income_category'].replace('abc', np.nan)

    # Run EDA
    run_eda(df)

    # Split Data
    x_training_data, y_training_data, x_validation_data, y_validation_data, x_testing_data, y_testing_data = split_data(df)

    # Impute Missing Values
    x_training_data, x_validation_data, x_testing_data = impute_missing_values(x_training_data, x_validation_data, x_testing_data)

    # Encode Data
    x_training_data, y_training_data, x_validation_data, y_validation_data, x_testing_data, y_testing_data = encode_data(
        x_training_data, x_validation_data, x_testing_data, y_training_data, y_validation_data, y_testing_data
    )

    # Drop rows in X_train and y_train where y_train has NaN values
    y_training_data = y_training_data.dropna()
    x_training_data = x_training_data.loc[y_training_data.index]  # Keep only rows in X_train that match y_train's index

    if y_training_data.empty:
        print('Warning! Target training data is empty after dropping NaNs. Imputation cannot be performed.')
    else:
        y_training_data = y_training_data.fillna(y_training_data.mode()[0])  # Impute only if y_train is not empty

    # Build Model with original data
    models = build_model()
    run_model_performance(models, x_training_data, y_training_data, x_training_data, y_training_data, 'Original Data Model', 'Training Performance')
    run_model_performance(models, x_training_data, y_training_data, x_validation_data, y_validation_data, 'Original Data Model', 'Validation Performance')
    run_model_performance(models, None, None, x_training_data, y_training_data, 'Original Data Model', 'Classification by Model(s)', True)

    # Oversampled Data
    x_training_oversample, y_training_oversample = oversample_data(x_training_data, y_training_data)
    oversample_models = build_model()
    run_model_performance(oversample_models, x_training_oversample, y_training_oversample, x_training_oversample, y_training_oversample, 'Oversampled Data Models', 'Training Performance')
    run_model_performance(oversample_models, x_training_oversample, y_training_oversample, x_validation_data, y_validation_data, 'Oversampled Data Models', 'Validation Performance')
    run_model_performance(oversample_models, None, None, x_training_oversample, y_training_oversample, 'Oversampled Data Models', 'Classification by Model(s)', True)

    # Undersampled Data
    x_training_undersample, y_training_undersample = undersample_data(x_training_data, y_training_data)
    undersample_models = build_model()
    run_model_performance(undersample_models, x_training_undersample, y_training_undersample, x_training_undersample, y_training_undersample, 'Undersampled Data Models', 'Training Performance')
    run_model_performance(undersample_models, x_training_undersample, y_training_undersample, x_validation_data, y_validation_data, 'Undersampled Data Models', 'Validation Performance')
    run_model_performance(undersample_models, None, None, x_training_undersample, y_training_undersample, 'Undersampled Data Models', 'Classification by Model(s)', True)

    # Hyperparameter Tuning
    # Gradient Boosting
    gbc = GradientBoostingClassifier(random_state=SEED)
    scorer = make_scorer(precision_score)
    
    # Evaluate params for RandomizedSearchCV
    decision_tree_params = {
        "init": [
            AdaBoostClassifier(random_state=SEED),
            DecisionTreeClassifier(random_state=SEED)
        ],
        "n_estimators": np.arange(50, 110, 25),
        "learning_rate": [0.01, 0.1, 0.05],
        "subsample": [0.7, 0.9],
        "max_features": [0.5, 0.7, 1],
    }

    randomized_cv = RandomizedSearchCV(
        estimator=gbc,
        param_distributions=decision_tree_params,
        n_iter=PARAM_DISTR_CNT,
        n_jobs=MAX_PROC_THREADS,
        scoring=scorer,
        cv=CV_FOLDS,
        random_state=SEED
    )
    randomized_cv.fit(x_training_data, y_training_data)
    print("Best parameters are {} with CV score={}:".format(randomized_cv.best_params_, randomized_cv.best_score_))

    gbc_tuned = GradientBoostingClassifier(
        random_state=SEED,
        n_estimators=TUNED_ESTIMATOR_CNT,
        learning_rate=TUNED_LEARNING_RATE,
        subsample=SUB_SAMPLE_SIZE,
        max_features=FEATURES_SPLIT_PCT
    )
    gbc_tuned.fit(x_training_data, y_training_data)
    gbc_tuned_scores = model_performance_classification_sklearn(gbc_tuned, x_training_data, y_training_data)
    gbc_tuned_validation_scores = model_performance_classification_sklearn(gbc_tuned, x_validation_data, y_validation_data)

    # Gradient Boosting Oversampled
    gbc_oversample = GradientBoostingClassifier(random_state=SEED)
    randomized_cv = RandomizedSearchCV(
        estimator=gbc_oversample,
        param_distributions=decision_tree_params,
        n_iter=PARAM_DISTR_CNT,
        n_jobs=MAX_PROC_THREADS,
        scoring=scorer,
        cv=CV_FOLDS,
        random_state=SEED
    )
    randomized_cv.fit(x_training_oversample, y_training_oversample)
    print("Best parameters are {} with CV score={}:".format(randomized_cv.best_params_, randomized_cv.best_score_))

    gbc_tuned_oversample = GradientBoostingClassifier(
        random_state=SEED,
        n_estimators=TUNED_ESTIMATOR_CNT,
        learning_rate=TUNED_LEARNING_RATE,
        subsample=SUB_SAMPLE_SIZE,
        max_features=FEATURES_SPLIT_PCT
    )
    gbc_tuned_oversample.fit(x_training_oversample, y_training_oversample)
    gbc_tuned_oversample_scores = model_performance_classification_sklearn(gbc_tuned_oversample, x_training_oversample, y_training_oversample)
    gbc_tuned_oversample_validation_scores = model_performance_classification_sklearn(gbc_tuned_oversample, x_validation_data, y_validation_data)

    # Gradient Boosting Undersampled
    gbc_undersample = GradientBoostingClassifier(random_state=SEED)
    randomized_cv = RandomizedSearchCV(
        estimator=gbc_undersample,
        param_distributions=decision_tree_params,
        n_iter=PARAM_DISTR_CNT,
        n_jobs=MAX_PROC_THREADS,
        scoring=scorer,
        cv=CV_FOLDS,
        random_state=SEED
    )
    randomized_cv.fit(x_training_undersample, y_training_undersample)
    print("Best parameters are {} with CV score={}:".format(randomized_cv.best_params_, randomized_cv.best_score_))

    gbc_tuned_undersample = GradientBoostingClassifier(
        random_state=SEED,
        n_estimators=TUNED_ESTIMATOR_CNT,
        learning_rate=TUNED_LEARNING_RATE,
        subsample=SUB_SAMPLE_SIZE,
        max_features=FEATURES_SPLIT_PCT
    )
    gbc_tuned_undersample.fit(x_training_data, y_training_data)
    gbc_tuned_undersample_scores = model_performance_classification_sklearn(gbc_tuned_undersample, x_training_data, y_training_data)
    gbc_tuned_undersample_validation_scores = model_performance_classification_sklearn(gbc_tuned_undersample, x_validation_data, y_validation_data)

    # AdaBoost Original
    ada_boost_params = {
        "n_estimators": np.arange(50, 110, 25),
        "learning_rate": [0.01, 0.1, 0.05],
        "estimator": [
            DecisionTreeClassifier(max_depth=2, random_state=SEED),
            DecisionTreeClassifier(max_depth=3, random_state=SEED),
        ],
    }
    ada = AdaBoostClassifier(random_state=SEED)
    randomized_cv = RandomizedSearchCV(
        estimator=ada,
        param_distributions=ada_boost_params,
        n_iter=PARAM_DISTR_CNT,
        n_jobs=MAX_PROC_THREADS,
        scoring=scorer,
        cv=CV_FOLDS,
        random_state=SEED
    )
    randomized_cv.fit(x_training_data, y_training_data)
    print("Best parameters are {} with CV score={}:".format(randomized_cv.best_params_, randomized_cv.best_score_))

    ada_tuned = AdaBoostClassifier(
        random_state=SEED,
        n_estimators=TUNED_ESTIMATOR_CNT,
        learning_rate=TUNED_LEARNING_RATE,
        estimator=DecisionTreeClassifier(max_depth=2, random_state=SEED),
    )
    ada_tuned.fit(x_training_data, y_training_data)
    ada_tuned_scores = model_performance_classification_sklearn(ada_tuned, x_training_data, y_training_data)
    ada_tuned_validation_scores = model_performance_classification_sklearn(ada_tuned, x_validation_data, y_validation_data)

    # AdaBoost Oversampled
    ada_oversample = AdaBoostClassifier(random_state=SEED)
    randomized_cv = RandomizedSearchCV(
        estimator=ada_oversample,
        param_distributions=ada_boost_params,
        n_iter=PARAM_DISTR_CNT,
        n_jobs=MAX_PROC_THREADS,
        scoring=scorer,
        cv=CV_FOLDS,
        random_state=SEED
    )
    randomized_cv.fit(x_training_oversample, y_training_oversample)
    print("Best parameters are {} with CV score={}:".format(randomized_cv.best_params_, randomized_cv.best_score_))

    ada_tuned_oversample = AdaBoostClassifier(
        random_state=SEED,
        n_estimators=TUNED_ESTIMATOR_CNT,
        learning_rate=TUNED_LEARNING_RATE,
        estimator=DecisionTreeClassifier(max_depth=2, random_state=SEED),
    )
    ada_tuned_oversample.fit(x_training_oversample, y_training_oversample)
    ada_tuned_oversample_scores = model_performance_classification_sklearn(ada_tuned_oversample, x_training_oversample, y_training_oversample)
    ada_tuned_oversample_validation_scores = model_performance_classification_sklearn(ada_tuned_oversample, x_validation_data, y_validation_data)

    # AdaBoost Undersampled
    ada_undersample = AdaBoostClassifier(random_state=SEED)
    randomized_cv = RandomizedSearchCV(
        estimator=ada_undersample,
        param_distributions=ada_boost_params,
        n_iter=PARAM_DISTR_CNT,
        n_jobs=MAX_PROC_THREADS,
        scoring=scorer,
        cv=CV_FOLDS,
        random_state=SEED
    )
    randomized_cv.fit(x_training_undersample, y_training_undersample)
    print("Best parameters are {} with CV score={}:".format(randomized_cv.best_params_, randomized_cv.best_score_))

    ada_tuned_undersample = AdaBoostClassifier(
        random_state=SEED,
        n_estimators=TUNED_ESTIMATOR_CNT,
        learning_rate=TUNED_LEARNING_RATE,
        estimator=DecisionTreeClassifier(max_depth=2, random_state=SEED),
    )
    ada_tuned_undersample.fit(x_training_undersample, y_training_undersample)
    ada_tuned_undersample_scores = model_performance_classification_sklearn(ada_tuned_undersample, x_training_undersample, y_training_undersample)
    ada_tuned_undersample_validation_scores = model_performance_classification_sklearn(ada_tuned_undersample, x_validation_data, y_validation_data)

    # XGBoost Original
    xgb_boost_params = {
        'n_estimators':np.arange(50, 110, 25),
        'scale_pos_weight':[1, 2, 5],
        'learning_rate':[0.01, 0.1, 0.05],
        'gamma':[1, 3],
        'subsample':[0.7, 0.9]
    }
    xgb = XGBClassifier(random_state=SEED)
    randomized_cv = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=xgb_boost_params,
        n_iter=PARAM_DISTR_CNT,
        n_jobs=MAX_PROC_THREADS,
        scoring=scorer,
        cv=CV_FOLDS,
        random_state=SEED
    )
    randomized_cv.fit(x_training_data, y_training_data)
    print("Best parameters are {} with CV score={}:".format(randomized_cv.best_params_, randomized_cv.best_score_))

    xgb_tuned = XGBClassifier(
        random_state=SEED,
        n_estimators=UNTUNED_ESTIMATOR_CNT,
        learning_rate=TUNED_LEARNING_RATE,
        subsample=SUB_SAMPLE_SIZE,
        max_features=FEATURES_SPLIT_PCT,
        gamma=MIN_TREE_SPLIT
    )
    xgb_tuned.fit(x_training_data, y_training_data)
    xgb_tuned_scores = model_performance_classification_sklearn(xgb_tuned, x_training_data, y_training_data)
    xgb_tuned_validation_scores = model_performance_classification_sklearn(xgb_tuned, x_validation_data, y_validation_data)

    # XGBoost Oversampled
    xgb_oversample = XGBClassifier(random_state=SEED)
    randomized_cv = RandomizedSearchCV(
        estimator=xgb_oversample,
        param_distributions=xgb_boost_params,
        n_iter=PARAM_DISTR_CNT,
        n_jobs=MAX_PROC_THREADS,
        scoring=scorer,
        cv=CV_FOLDS,
        random_state=SEED
    )
    randomized_cv.fit(x_training_oversample, y_training_oversample)
    print("Best parameters are {} with CV score={}:".format(randomized_cv.best_params_, randomized_cv.best_score_))

    xgb_tuned_oversample = XGBClassifier(
        random_state=SEED,
        n_estimators=UNTUNED_ESTIMATOR_CNT,
        learning_rate=TUNED_LEARNING_RATE,
        subsample=SUB_SAMPLE_SIZE,
        max_features=FEATURES_SPLIT_PCT,
        gamma=MIN_TREE_SPLIT
    )
    xgb_tuned_oversample.fit(x_training_oversample, y_training_oversample)
    xgb_tuned_oversample_scores = model_performance_classification_sklearn(xgb_tuned_oversample, x_training_oversample, y_training_oversample)
    xgb_tuned_oversample_validation_scores = model_performance_classification_sklearn(xgb_tuned_oversample, x_validation_data, y_validation_data)

    # XGBoost Undersampled
    xgb_undersample = XGBClassifier(random_state=SEED)
    randomized_cv = RandomizedSearchCV(
        estimator=xgb_undersample,
        param_distributions=xgb_boost_params,
        n_iter=PARAM_DISTR_CNT,
        n_jobs=MAX_PROC_THREADS,
        scoring=scorer,
        cv=CV_FOLDS,
        random_state=SEED
    )
    randomized_cv.fit(x_training_undersample, y_training_undersample)
    print("Best parameters are {} with CV score={}:".format(randomized_cv.best_params_, randomized_cv.best_score_))

    xgb_tuned_undersample = XGBClassifier(
        random_state=SEED,
        n_estimators=UNTUNED_ESTIMATOR_CNT,
        learning_rate=TUNED_LEARNING_RATE,
        subsample=SUB_SAMPLE_SIZE,
        max_features=FEATURES_SPLIT_PCT,
        gamma=MIN_TREE_SPLIT
    )
    xgb_tuned_undersample.fit(x_training_undersample, y_training_undersample)
    xgb_tuned_undersample_scores = model_performance_classification_sklearn(xgb_tuned_undersample, x_training_undersample, y_training_undersample)
    xgb_tuned_undersample_validation_scores = model_performance_classification_sklearn(xgb_tuned_undersample, x_validation_data, y_validation_data)

    # Training Comparison
    training_models = pd.concat([
        gbc_tuned_scores.T,
        gbc_tuned_oversample_scores.T,
        gbc_tuned_undersample_scores.T,
        ada_tuned_scores.T,
        ada_tuned_oversample_scores.T,
        ada_tuned_undersample_scores.T,
        xgb_tuned_scores.T,
        xgb_tuned_oversample_scores.T,
        xgb_tuned_undersample_scores.T,
    ], axis=1)

    training_models.columns = [
        'Gradient Boosting Original',
        'Gradient Boosting Oversampled',
        'Gradient Boosting Undersampled',
        'AdaBoost Original',
        'AdaBoost Oversampled',
        'AdaBoost Undersampled',
        'XGBoost Original',
        'XGBoost Oversampled',
        'XGBoost Undersampled'
    ]
    print(training_models)

    # Validation Comparison
    validation_models = pd.concat([
        gbc_tuned_validation_scores.T,
        gbc_tuned_oversample_validation_scores.T,
        gbc_tuned_undersample_validation_scores.T,
        ada_tuned_validation_scores.T,
        ada_tuned_oversample_validation_scores.T,
        ada_tuned_undersample_validation_scores.T,
        xgb_tuned_validation_scores.T,
        xgb_tuned_oversample_validation_scores.T,
        xgb_tuned_undersample_validation_scores.T,
    ], axis=1)

    validation_models.columns = [
        'Gradient Boosting Original Value',
        'Gradient Boosting Oversampled Value',
        'Gradient Boosting Undersampled Value',
        'AdaBoost Original Value',
        'AdaBoost Oversampled Value',
        'AdaBoost Undersampled Value',
        'XGBoost Original Value',
        'XGBoost Oversampled Value',
        'XGBoost Undersampled Value'
    ]
    print(validation_models)

    # Test set final performance
    xgb_undersample_scores_model = model_performance_classification_sklearn(xgb_tuned_undersample, x_testing_data, y_testing_data)
    xgb_oversample_scores_model = model_performance_classification_sklearn(xgb_tuned_oversample, x_testing_data, y_testing_data)
    xgb_tuned_scores_model = model_performance_classification_sklearn(xgb_tuned, x_testing_data, y_testing_data)

    xgb_models = [
        xgb_tuned_undersample,
        xgb_tuned_oversample,
        xgb_tuned
    ]

    xgb_comparison_models = pd.concat([
        xgb_undersample_scores_model.T,
        xgb_oversample_scores_model.T,
        xgb_tuned_scores_model.T],
        axis=1
    )

    xgb_comparison_models.columns = [
        'XGBoost Undersampled',
        'XGBoost Oversampled',
        'XGBoost Tuned'
    ]
    print(xgb_comparison_models)

    # Final model (highest score)
    top_model = pick_top_model(xgb_comparison_models, xgb_models)
    model_performance_classification_sklearn(top_model, x_testing_data, y_testing_data)
    plot_confusion_matrix(top_model, x_testing_data, y_testing_data)

if __name__ == "__main__":
    main()
