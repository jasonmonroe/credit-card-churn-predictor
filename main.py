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

from src.config import DF_TYPES
from src.data_handler import DataHandler
from src.eda import run_eda
from src.utils import get_run_id, start_timer, show_timer
#from src.preprocessing import (
#    get_df,
#    process_data
#)
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

    data_handler = DataHandler(seed_data)
    df = data_handler.data

    # Run EDA
    run_eda(df)

    print('--- EDA complete ---')

def run_main_pipeline(seed_data=False):

    print('+-----------------------------+')
    print('| CREDIT CARD CHURN PREDICTOR |')
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
    
    # ⚠ Pick the best model performance
    show_banner('---️ 📊 Final Model w/ Plot Confusion Matrix 📊 ---')
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

    print(f'\n----- ⏱️ END RUN ID: {run_id} ⏱️ -----')
