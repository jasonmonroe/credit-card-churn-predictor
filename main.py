# src/main.py
import argparse
import gc
import pandas as pd
import numpy as np

from src.config import *
from src.eda import run_eda
from src import utils
from src import preprocessing
from src.preprocessing import (
    split_seeder_data,
    impute_missing_values,
    encode_data
)

from src.modeling import (
    run_model_performance,
    oversample_data,
    undersample_data,
    pick_top_model,
    model_performance_classification_sklearn,
    plot_confusion_matrix,
    build_models
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import ParameterGrid

from src.utils import show_banner


def get_df(seed_data=False):

    # Create a copy of the data (that will be used later)
    #df = data.copy()
    df = preprocessing.load_data(seed_data)
    df = preprocessing.clean_data(df)
    
    return df

def run_eda_pipeline(seed_data=False):
    
    df = get_df(seed_data)

    # Run EDA
    run_eda(df)

    print('EDA complete.')


def tune_and_evaluate(estimator, params, x_train, y_train, x_val, y_val, scorer):
    """
    Helper function to perform RandomizedSearchCV, fit the best model, and calculate scores.
    """
    # Calculate total parameter space size
    total_params = len(ParameterGrid(params))
    n_iter = min(PARAM_DISTR_CNT, total_params)

    randomized_cv = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=n_iter,
        n_jobs=MAX_PROC_THREADS,
        scoring=scorer,
        cv=CV_FOLDS,
        random_state=SEED
    )
    randomized_cv.fit(x_train, y_train)
    print("Best parameters are {} with CV score={}:".format(randomized_cv.best_params_, randomized_cv.best_score_))

    # Re-instantiate or use best_estimator_ directly. 
    # Using best_estimator_ is safer as it contains the fitted model with best params.
    best_model = randomized_cv.best_estimator_
    
    # If you specifically wanted to re-fit on x_train (though best_estimator_ is already refit on the passed x_train)
    # best_model.fit(x_train, y_train) 

    train_scores = model_performance_classification_sklearn(best_model, x_train, y_train)
    val_scores = model_performance_classification_sklearn(best_model, x_val, y_val)

    return best_model, train_scores, val_scores


def main(seed_data=False):
    # Load and clean data
    df = get_df(seed_data)

    # Split Data
    x_training_data, y_training_data, x_validation_data, y_validation_data, x_testing_data, y_testing_data = split_seeder_data(df)

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
    # models = build_model()
    orig_models = build_models()
    run_model_performance(orig_models, x_training_data, y_training_data, x_training_data, y_training_data, 'Original Data Model', 'Training Performance')
    run_model_performance(orig_models, x_training_data, y_training_data, x_validation_data, y_validation_data, 'Original Data Model', 'Validation Performance')
    run_model_performance(orig_models, None, None, x_training_data, y_training_data, 'Original Data Model', 'Classification by Model(s)', True)

    # Oversampled Data
    x_training_oversample, y_training_oversample = oversample_data(x_training_data, y_training_data)
    oversample_models = build_models()
    run_model_performance(oversample_models, x_training_oversample, y_training_oversample, x_training_oversample, y_training_oversample, 'Oversampled Data Models', 'Training Performance')
    run_model_performance(oversample_models, x_training_oversample, y_training_oversample, x_validation_data, y_validation_data, 'Oversampled Data Models', 'Validation Performance')
    run_model_performance(oversample_models, None, None, x_training_oversample, y_training_oversample, 'Oversampled Data Models', 'Classification by Model(s)', True)

    # Undersampled Data
    x_training_undersample, y_training_undersample = undersample_data(x_training_data, y_training_data)
    undersample_models = build_models()
    run_model_performance(undersample_models, x_training_undersample, y_training_undersample, x_training_undersample, y_training_undersample, 'Undersampled Data Models', 'Training Performance')
    run_model_performance(undersample_models, x_training_undersample, y_training_undersample, x_validation_data, y_validation_data, 'Undersampled Data Models', 'Validation Performance')
    run_model_performance(undersample_models, None, None, x_training_undersample, y_training_undersample, 'Undersampled Data Models', 'Classification by Model(s)', True)

    # --- Hyperparameter Tuning ---
    scorer = make_scorer(precision_score, zero_division=0)

    # Define the datasets to iterate over
    datasets = {
        'Original': (x_training_data, y_training_data),
        'Oversampled': (x_training_oversample, y_training_oversample),
        'Undersampled': (x_training_undersample, y_training_undersample)
    }

    # Define models and their configs
    # Format: (Name, Estimator, Params)
    models_config = [
        ('Gradient Boosting', GradientBoostingClassifier(random_state=SEED), GB_PARAMS),
        ('AdaBoost', AdaBoostClassifier(random_state=SEED), ADA_BOOST_PARAMS),
        ('XGBoost', XGBClassifier(random_state=SEED), XGB_BOOST_PARAMS)
    ]

    # Storage for results
    training_results_list = []
    validation_results_list = []
    column_names = []
    
    # Specific storage for XGBoost comparison later
    xgb_models_storage = {} 

    for model_name, estimator, params in models_config:
        for data_name, (x_train, y_train) in datasets.items():
            full_name = f"{model_name} {data_name}"
            show_banner(f"Tuning {full_name}")

            best_model, train_score, val_score = tune_and_evaluate(
                estimator, params, x_train, y_train, x_validation_data, y_validation_data, scorer
            )

            # Store scores for the big comparison table
            training_results_list.append(train_score.T)
            validation_results_list.append(val_score.T)
            column_names.append(full_name)

            # Store XGBoost models specifically for the final step
            if model_name == 'XGBoost':
                xgb_models_storage[data_name] = best_model

    # --- Comparison of Models --- #

    # Training Comparison
    training_models = pd.concat(training_results_list, axis=1)
    training_models.columns = column_names
    # Rename columns to match original output format if strictly necessary, 
    # but the generated names "Gradient Boosting Original", etc. are already correct.
    
    # Adjust column names to match the specific "Value" suffix used in original code for validation
    val_cols = [f"{name} Value" for name in column_names]
    
    print("\n--- Training Comparison ---")
    print(training_models)

    # Validation Comparison
    validation_models = pd.concat(validation_results_list, axis=1)
    validation_models.columns = val_cols
    
    print("\n--- Validation Comparison ---")
    print(validation_models)

    # Test Final Performance (XGBoost specific)
    # Retrieve the specific models we stored
    xgb_tuned_undersample = xgb_models_storage['Undersampled']
    xgb_tuned_oversample = xgb_models_storage['Oversampled']
    # Assuming "Original" is the "Tuned" one in the final comparison context
    xgb_tuned = xgb_models_storage['Original'] 

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

    # Final model (the highest score)
    show_banner('Final Model w/ Plot Confusion Matrix')
    top_model = pick_top_model(xgb_comparison_models, xgb_models)
    model_performance_classification_sklearn(top_model, x_testing_data, y_testing_data)
    plot_confusion_matrix(top_model, x_testing_data, y_testing_data)    


# --- Main --- #
if __name__ == '__main__':
    main_start_time = utils.start_timer()
    run_id = utils.get_run_id()
    print(f'\n# --- {run_id} | START PROGRAM --- #')

    # --- Check arguments ---
    parser = argparse.ArgumentParser(description='Credit Card Churn Predictor')
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'eda'],
        help='Execution mode: train (default) or eda'
    )
    parser.add_argument(
        '--seed',
        action='store_true',
        help='Generate seed data and merge with sample data'
    )

    args = parser.parse_args()

    if args.mode == 'eda':
        run_eda_pipeline(args.seed)
    else:
        main(args.seed)

    gc.collect()

    print('\n')
    utils.show_timer(main_start_time)
    print(f'\n#--- {run_id} | END PROGRAM ---#')
