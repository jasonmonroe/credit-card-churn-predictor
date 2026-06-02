# src/main.py
import argparse
import gc
import pandas as pd

from models.ada_boost import AdaBoostModel
from models.bagging import BaggingModel
from models.gradient_boosting import GradientBoostingModel
from models.model_evaluator import ModelEvaluator
from models.random_forest import RandomForestModel
from models.xg_boost import XGBoostModel

from sklearn.metrics import make_scorer, precision_score

from src.config import (
    ADA_BOOST_PARAMS,
    GB_PARAMS,
    SEED,
    XGB_BOOST_PARAMS,
    UNTUNED_LEARNING_RATE, DF_TYPES
)
from src.data_handler import DataHandler
from src.eda import run_eda, plot_confusion_matrix
from src.utils import (get_run_id, start_timer, show_timer, show_banner)
from src.preprocessing import (
    get_df,
    process_data
)
from src.modeling import (
    run_model_performance,
    oversample_data,
    undersample_data,
    pick_best_model,
    #model_performance_classification_sklearn,
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

    data_handler = DataHandler(seed_data)
    df = data_handler.data

    """
    
    """
    # Create base model to get the over and undersampled data that will be used for all the models.
    model_eval = ModelEvaluator(df)
    sampled = model_eval.get_sampled()

    # Merge the sampled data into the dataset
    df = {**df, **sampled}


    """
    Build Models with original, oversampled and undersampled data.
    Models include: Bagging Classifier, Random Forest Classifier, ADA Boost Classifier, Gradient Boosting Classifier, 
    XG Boost Classifier.
    Build each model with original, oversampled and undersampled data.
    """

    # Bagging Model
    bagging_model = BaggingModel(df)
    bagging_model.run()
    #bagging_model.run_orig()
    #bagging_model.run_oversampled()
    #bagging_model.run_undersampled()

    # Random Forest Classifier
    rf_model = RandomForestModel(df)
    rf_model.run()
    #rf_model.run_orig()
    #rf_model.run_oversampled(x_os, y_os)
    #rf_model.run_undersampled(x_us, y_us)

    # ADA Boost Classifier
    ada_boost_model = AdaBoostModel(df)
    ada_boost_model.run()
    #ada_boost_model.run_orig()
    #ada_boost_model.run_oversampled(x_os, y_os)
    #ada_boost_model.run_undersampled(x_us, y_us)

    # Gradient Boosting Classifier
    gradient_boost_model = GradientBoostingModel(df)
    gradient_boost_model.run()
    #gradient_boost_model.run_orig()
    #gradient_boost_model.run_oversampled(x_os, y_os)
    #gradient_boost_model.run_undersampled(x_us, y_us)

    # XG Boost Classifier
    xg_boost_model = XGBoostModel(df)
    xg_boost_model.run()
    #xg_boost_model.run_orig()
    #xg_boost_model.run_oversampled(x_os, y_os)
    #xg_boost_model.run_undersampled(x_us, y_us)

    # --- Hyperparameter Tuning ---
    scorer = make_scorer(precision_score, zero_division=0)

    # Define the datasets to iterate over
    #xg_boost = {}
    gradient_boost_results = gradient_boost_model.get_results(scorer)
    ada_boost_results = ada_boost_model.get_results(scorer)
    xg_boost_results = xg_boost_model.get_results(scorer)

    print('--- Training Comparison ---')
    comp_columns = [gradient_boost_results['titles'], ada_boost_results['titles'], xg_boost_results['titles']]
    training_models = pd.concat([gradient_boost_results['train'], ada_boost_results['train'], xg_boost_results['train']], axis=1)
    training_models.columns = comp_columns
    print(training_models)
    #print(gradient_boost_results['train'])
    #print(ada_boost_results['train'])
    #print(xg_boost_results['train'])
    #train_models = pd.concat(train_results, axis=1)
    #training_models.columns =

    print('--- Validation Comparison ---')
    val_models = pd.concat([gradient_boost_results['val'], ada_boost_results['val'], xg_boost_results['val']], axis=1)
    val_models.columns = comp_columns
    print(val_models)
    #val_models = pd.concat(val_results, axis=1)
    
    
    # Final Test Performance (XGB Boost Specific)
    xg_boost_perfs = xg_boost_model.get_perf()
    
    # Create the comparison dataframe horizontally to match previous tables
    xg_boost_comps = pd.concat([xg_boost_perfs[xgb_type] for xgb_type in DF_TYPES], axis=1)
    xg_boost_comps.columns = [name.capitalize() for name in DF_TYPES]
    print(xg_boost_comps)
    
    # Pick the best model performance
    show_banner('⚠️ Final Model w/ Plot Confusion Matrix')
    best_model = xg_boost_model.show_best(xg_boost_comps)
    
    
    #for xgb_type in xg_boost_model.best_estimator:
        
        
    #xgb_best_estimator = xg_boost_model.best_estimator
    

    # Store results
    training_results_list = []
    validation_results_list = []
    column_names = []






    # --- End of Program --- #


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
    main_start_time = start_timer()
    run_id = get_run_id()
    print(f'\n----- ⏱️ START RUN ID: {run_id} ⏱️ -----')

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
    show_timer(main_start_time)

    print(f'\n----- ⏱️ END RUN ID: {run_id} ⏱️ -----')
