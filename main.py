# src/main.py

import argparse
import gc
import sys

import pandas as pd

from sklearn.metrics import make_scorer, precision_score

from models.ada_boost import AdaBoostModel
from models.bagging import BaggingModel
from models.gradient_boosting import GradientBoostingModel
from models.model_evaluator import ModelEvaluator
from models.random_forest import RandomForestModel
from models.xg_boost import XGBoostModel

from src.config import DATASET_TYPES, OUTPUT_FILE
from src.data_handler import DataHandler
from src.eda import run_eda
from src.utils import get_run_id, start_timer, show_timer, show_title_banner, get_time, show_banner


def run_eda_pipeline(seed_data=False):

    data_handler = DataHandler(seed_data)
    df = data_handler.data

    # Run EDA
    run_eda(df)

    print('--- EDA complete ---')

def run_main_pipeline(seed_data=False):
    print(show_title_banner())

    data_handler = DataHandler(seed_data)
    df = data_handler.get()

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
    Each model has it's own original, oversampled and undersampled recall scores 
    """

    # Bagging Model
    bagging_model = BaggingModel(df)
    bagging_model.run()

    # Random Forest Classifier
    rf_model = RandomForestModel(df)
    rf_model.run()

    # ADA Boost Classifier
    ada_boost_model = AdaBoostModel(df)
    ada_boost_model.run()

    # Gradient Boosting Classifier
    gradient_boosting_model = GradientBoostingModel(df)
    gradient_boosting_model.run()

    # XG Boost Classifier
    xg_boost_model = XGBoostModel(df)
    xg_boost_model.run()

    # --- Hyperparameter Tuning ---
    scorer = make_scorer(precision_score, zero_division=0)

    # Get training, validation results from each model for original, oversampled and undersampled data.
    bagging_results = bagging_model.get_results(scorer)
    random_forest_results = rf_model.get_results(scorer)
    gradient_boosting_results = gradient_boosting_model.get_results(scorer)
    ada_boost_results = ada_boost_model.get_results(scorer)
    xg_boost_results = xg_boost_model.get_results(scorer)

    # Collect all results into a list to process dynamically
    # This list allows us to iterate once and handle all comparison tables
    orig_model_results = [gradient_boosting_results, ada_boost_results, xg_boost_results]
    model_results = [
        bagging_results,
        random_forest_results,
        ada_boost_results,
        gradient_boosting_results,
        ada_boost_results,
        xg_boost_results
    ]

    """
    --- Format Results for Comparison ---
    Take the training and validation data for each model that used original, oversampled and undersampled data and 
    create a matrix for viewing.
    """
    model_eval.print_comparisons(model_results)

    # --- Pick the best XGBoost Model --- #
    # Final Test Performance (XGB Boost Specific)
    xg_boost_perfs = xg_boost_model.get_perf()
    
    # Create the comparison dataframe horizontally to match previous tables
    # Transpose (.T) each set so metrics become rows and sampling types become single columns
    xg_boost_comps = pd.concat([xg_boost_perfs[xgb_type].T for xgb_type in DATASET_TYPES], axis=1)
    xg_boost_comps.columns = [name.capitalize() for name in DATASET_TYPES]

    print('\n*** XGB Boost Comparisons ***')
    print(xg_boost_comps)
    
    # ⚠ Pick the best model performance
    show_banner('📊 Final Model w/ Plot Confusion Matrix 📊')
    xg_boost_model.show_best(xg_boost_comps)

    # --- End of Program --- #

if __name__ == '__main__':
    main_start_time = start_timer()
    run_id = get_run_id()
    print(f'\n----- ⏱️ START RUN ID: {run_id} ⏱️ -----\n')

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

    # Write the run time at the end of file
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f'\n\n--------------- ⏱️ Run ID: {run_id} | Total Execution Time: {get_time(main_start_time)} ⏱️ -----------------')

    print(f'\n----- ⏱️ END RUN ID: {run_id} ⏱️ -----')
