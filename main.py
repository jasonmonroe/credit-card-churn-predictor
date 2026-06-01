# src/main.py
import argparse
import gc
import pandas as pd

from src.config import (
    ADA_BOOST_PARAMS,
    GB_PARAMS,
    SEED,
    XGB_BOOST_PARAMS,
    UNTUNED_LEARNING_RATE
)
from src.eda import run_eda, plot_confusion_matrix
from src import utils
from src.preprocessing import (
    get_df,
    process_data
)
from src.modeling import (
    run_model_performance,
    oversample_data,
    undersample_data,
    pick_best_model,
    model_performance_classification_sklearn,
    build_models,
    tune_and_evaluate,
    gradient_boosting_model,
    xgboost_model,
    ada_boost_model
)

from sklearn.metrics import make_scorer, precision_score
from src.utils import show_banner

def run_eda_pipeline(seed_data=False):
    
    df = get_df(seed_data)

    # Run EDA
    run_eda(df)

    print('--- EDA complete ---')

def run_main_pipeline(seed_data=False):

    print('+-----------------------------+')
    print('+ CREDIT CARD CHURN PREDICTOR +')
    print('+-----------------------------+')

    # Load and clean data
    df = get_df(seed_data)

    x_training_data, y_training_data, x_validation_data, y_validation_data, x_testing_data, y_testing_data = process_data(df)

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
        ('Gradient Boosting', gradient_boosting_model(), GB_PARAMS),
        ('AdaBoost', ada_boost_model(), ADA_BOOST_PARAMS),
        ('XGBoost', xgboost_model(), XGB_BOOST_PARAMS)
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
    top_model = pick_best_model(xgb_comparison_models, xgb_models)
    model_performance_classification_sklearn(top_model, x_testing_data, y_testing_data)
    plot_confusion_matrix(top_model, x_testing_data, y_testing_data)    


# --- Main --- #
if __name__ == '__main__':
    main_start_time = utils.start_timer()
    run_id = utils.get_run_id()
    print(f'\n{run_id} | START PROGRAM')

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
        run_main_pipeline(args.seed)

    gc.collect()

    print('\n')
    utils.show_timer(main_start_time)
    print(f'\n{run_id} | END PROGRAM')
