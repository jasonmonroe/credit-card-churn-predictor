# src/main.py

import argparse
import gc
import pandas as pd

from sklearn.metrics import make_scorer, precision_score

from models.ada_boost import AdaBoostModel
from models.bagging import BaggingModel
from models.gradient_boosting import GradientBoostingModel
from models.model_evaluator import ModelEvaluator
from models.random_forest import RandomForestModel
from models.xg_boost import XGBoostModel

from src.config import DF_TYPES, OUTPUT_FILE
from src.data_handler import DataHandler
from src.eda import run_eda
from src.utils import get_run_id, start_timer, show_timer
from src.utils import get_time, show_banner


def run_eda_pipeline(seed_data=False):

    data_handler = DataHandler(seed_data)
    df = data_handler.data

    # Run EDA
    run_eda(df)

    print('--- EDA complete ---')

def run_main_pipeline(seed_data=False):

    print('+-----------------------------------+')
    print('| 💳 CREDIT CARD CHURN PREDICTOR 💳 |')
    print('+-----------------------------------+')

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
    model_results = [bagging_results, random_forest_results, ada_boost_results, gradient_boosting_results, ada_boost_results, xg_boost_results]

    
    print('DEBUG: --- model_results ----\n')
    print(model_results)
    print('DEBUG: --- model_results ----\n')

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
    xg_boost_comps = pd.concat([xg_boost_perfs[xgb_type].T for xgb_type in DF_TYPES], axis=1)
    xg_boost_comps.columns = [name.capitalize() for name in DF_TYPES]

    print('*** XGB Boost Comparisons ***')
    print(xg_boost_comps)
    
    # ⚠ Pick the best model performance
    show_banner('📊 Final Model w/ Plot Confusion Matrix 📊')
    xg_boost_model.show_best(xg_boost_comps)

    # --- End of Program --- #


"""
def run_debug(seed_data=False):
    import pandas as pd
    import numpy as np

    print('\n### DEBUG OUTPUT ####')

    # -------------------------------------------------------------
    # HARDCODED PIPELINE DATA (Updated from your new terminal dump)
    # -------------------------------------------------------------

    # 1. Bagging Data Blocks
    bagging_results = {
        'train': pd.DataFrame([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], columns=['Accuracy', 'Recall', 'Precision', 'F1'], index=[0, 0, 0]),
        'val': pd.DataFrame([[0.9605, 0.836066, 0.910714, 0.871795], [0.9605, 0.836066, 0.910714, 0.871795], [0.9605, 0.836066, 0.910714, 0.871795]], columns=['Accuracy', 'Recall', 'Precision', 'F1'], index=[0, 0, 0]),
        'titles': ['Bagging Original', 'Bagging Oversampled', 'Bagging Undersampled']
    }

    # 2. Random Forest Data Blocks
    random_forest_results = {
        'train': pd.DataFrame([[0.913516, 0.527656, 0.889053, 0.662259], [0.913516, 0.527656, 0.889053, 0.662259], [0.913516, 0.527656, 0.889053, 0.662259]], columns=['Accuracy', 'Recall', 'Precision', 'F1'], index=[0, 0, 0]),
        'val': pd.DataFrame([[0.899934, 0.45082, 0.859375, 0.591398], [0.899934, 0.45082, 0.859375, 0.591398], [0.899934, 0.45082, 0.859375, 0.591398]], columns=['Accuracy', 'Recall', 'Precision', 'F1'], index=[0, 0, 0]),
        'titles': ['Random Forest Classifier Original', 'Random Forest Classifier Oversampled', 'Random Forest Classifier Undersampled']
    }

    # 3. ADA Boost Data Blocks
    ada_boost_results = {
        'train': pd.DataFrame([[0.96741, 0.843723, 0.947732, 0.892708], [0.96741, 0.843723, 0.947732, 0.892708], [0.96741, 0.843723, 0.947732, 0.892708]], columns=['Accuracy', 'Recall', 'Precision', 'F1'], index=[0, 0, 0]),
        'val': pd.DataFrame([[0.949967, 0.758197, 0.915842, 0.829596], [0.949967, 0.758197, 0.915842, 0.829596], [0.949967, 0.758197, 0.915842, 0.829596]], columns=['Accuracy', 'Recall', 'Precision', 'F1'], index=[0, 0, 0]),
        'titles': ['ADA Boost Classifier Original', 'ADA Boost Classifier Oversampled', 'ADA Boost Classifier Undersampled']
    }

    # 4. Gradient Boosting Data Blocks
    gradient_boosting_results = {
        'train': pd.DataFrame([[0.975875, 0.889377, 0.957467, 0.922167], [0.975875, 0.889377, 0.957467, 0.922167], [0.975875, 0.889377, 0.957467, 0.922167]], columns=['Accuracy', 'Recall', 'Precision', 'F1'], index=[0, 0, 0]),
        'val': pd.DataFrame([[0.9605, 0.807377, 0.938095, 0.867841], [0.9605, 0.807377, 0.938095, 0.867841], [0.9605, 0.807377, 0.938095, 0.867841]], columns=['Accuracy', 'Recall', 'Precision', 'F1'], index=[0, 0, 0]),
        'titles': ['Gradient Boosting Classifier Original', 'Gradient Boosting Classifier Oversampled', 'Gradient Boosting Classifier Undersampled']
    }

    # 5. XGBoost Data Blocks
    xg_boost_results = {
        'train': pd.DataFrame([[0.881631, 0.271291, 0.971698, 0.424159], [0.881631, 0.271291, 0.971698, 0.424159], [0.881631, 0.271291, 0.971698, 0.424159]], columns=['Accuracy', 'Recall', 'Precision', 'F1'], index=[0, 0, 0]),
        'val': pd.DataFrame([[0.875576, 0.241803, 0.936508, 0.384365], [0.875576, 0.241803, 0.936508, 0.384365], [0.875576, 0.241803, 0.936508, 0.384365]], columns=['Accuracy', 'Recall', 'Precision', 'F1'], index=[0, 0, 0]),
        'titles': ['XG Boost Classifier Original', 'XG Boost Classifier Oversampled', 'XG Boost Classifier Undersampled'],
        'best_estimator': {'original': 'XGBClassifier(...)', 'oversampled': 'XGBClassifier(...)', 'undersampled': 'XGBClassifier(...)'}
    }

    # Re-assembling model_results array exactly like your terminal execution state
    model_results = [
        bagging_results,
        random_forest_results,
        ada_boost_results,
        gradient_boosting_results,
        ada_boost_results,
        xg_boost_results
    ]

    # -------------------------------------------------------------
    # THE BREAKING PIPELINE RUN
    # -------------------------------------------------------------

    # Generates the target flat labels list
    flat_titles = [title for res in model_results for title in res['titles']]
    #print(flat_titles)

    print('\n--- ⚙️ Training Comparison ⚙️ ---')
    try:

        training_chart = bagging_results['train'].T
        print("--- Training Chart ---")
        print(training_chart)

        print('###############')
        # 1. Take the original DataFrame (which has rows 0, 0, 0)
        df_train = bagging_results['train'].copy()

        # 2. Force overwrite the row labels [0, 0, 0] with your list of titles
        df_train.index = bagging_results['titles']

        # 3. Transpose (.T) to turn the titles into columns and the metrics into rows
        training_chart = df_train.T

        print("--- Training Chart (Inline Fix) ---")
        print(training_chart)


        # This triggers your exact KeyError immediately!
        #training_models = pd.concat([res['train'].loc[res['titles']].T for res in model_results], axis=1)
        #training_models.columns = flat_titles
        #print(training_models)
    except KeyError as e:
        print(f"🚨 CAUGHT EXPECTED EXCEPTION:\nKeyError: {e}")

    print('\n--- 💡 THE PRODUCTION REFACTOR FIX ---')
    # Fix logic: Reassign indices cleanly inside the looping block so .loc maps perfectly

    train_columns_fixed = []
    val_columns_fixed = []
    resolved_flat_titles = []

    for res in model_results:
        # 1. Process Training Blocks
        df_train = res['train'].copy()
        df_train.index = res['titles']  # Overwrites the broken [0, 0, 0] with real strings
        train_columns_fixed.append(df_train.loc[res['titles']].T)

        # 2. Process Validation Blocks
        df_val = res['val'].copy()
        df_val.index = res['titles']
        val_columns_fixed.append(df_val.loc[res['titles']].T)

        # 3. Collect titles explicitly to avoid length mismatches from duplicated entry items
        resolved_flat_titles.extend(res['titles'])

    # Horizontal Concatenation
    training_models_fixed = pd.concat(train_columns_fixed, axis=1)
    training_models_fixed.columns = resolved_flat_titles
    print("TRAINING COMPARISON GRID:")
    print(training_models_fixed)

    print('\n--- Validation Comparison ---')
    val_models_fixed = pd.concat(val_columns_fixed, axis=1)
    val_models_fixed.columns = [f"{t} Value" for t in resolved_flat_titles]
    print("VALIDATION COMPARISON GRID:")
    print(val_models_fixed)

    print('!!! PRINT COMPARISONS !!!')
    #best_estimator = # Pure Python dictionary fixture for testing your XGBoost selection logic
    best_estimator = {
        'original': {
            'base_score': None,
            'booster': None,
            'callbacks': None,
            'colsample_bylevel': None,
            'colsample_bynode': None,
            'colsample_bytree': None,
            'device': None,
            'early_stopping_rounds': None,
            'enable_categorical': False,
            'eval_metric': None,
            'feature_types': None,
            'feature_weights': None,
            'gamma': 3,
            'grow_policy': None,
            'importance_type': None,
            'interaction_constraints': None,
            'learning_rate': 0.01,
            'max_bin': None,
            'max_cat_threshold': None,
            'max_cat_to_onehot': None,
            'max_delta_step': None,
            'max_depth': 3,
            'max_leaves': None,
            'min_child_weight': None,
            'missing': None,
            'monotone_constraints': None,
            'multi_strategy': None,
            'n_estimators': 75,
            'n_jobs': None,
            'num_parallel_tree': None,
            'objective': 'binary:logistic',
            'random_state': 42,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'sampling_method': None,
            'scale_pos_weight': 1,
            'subsample': 0.7,
            'tree_method': None,
            'validate_parameters': None,
            'verbosity': None
        },
        'oversampled': {
            'base_score': None,
            'booster': None,
            'callbacks': None,
            'colsample_bylevel': None,
            'colsample_bynode': None,
            'colsample_bytree': None,
            'device': None,
            'early_stopping_rounds': None,
            'enable_categorical': False,
            'eval_metric': None,
            'feature_types': None,
            'feature_weights': None,
            'gamma': 3,
            'grow_policy': None,
            'importance_type': None,
            'interaction_constraints': None,
            'learning_rate': 0.01,
            'max_bin': None,
            'max_cat_threshold': None,
            'max_cat_to_onehot': None,
            'max_delta_step': None,
            'max_depth': 3,
            'max_leaves': None,
            'min_child_weight': None,
            'missing': None,
            'monotone_constraints': None,
            'multi_strategy': None,
            'n_estimators': 75,
            'n_jobs': None,
            'num_parallel_tree': None,
            'objective': 'binary:logistic',
            'random_state': 42,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'sampling_method': None,
            'scale_pos_weight': 1,
            'subsample': 0.7,
            'tree_method': None,
            'validate_parameters': None,
            'verbosity': None
        },
        'undersampled': {
            'base_score': None,
            'booster': None,
            'callbacks': None,
            'colsample_bylevel': None,
            'colsample_bynode': None,
            'colsample_bytree': None,
            'device': None,
            'early_stopping_rounds': None,
            'enable_categorical': False,
            'eval_metric': None,
            'feature_types': None,
            'feature_weights': None,
            'gamma': 3,
            'grow_policy': None,
            'importance_type': None,
            'interaction_constraints': None,
            'learning_rate': 0.01,
            'max_bin': None,
            'max_cat_threshold': None,
            'max_cat_to_onehot': None,
            'max_delta_step': None,
            'max_depth': 3,
            'max_leaves': None,
            'min_child_weight': None,
            'missing': None,
            'monotone_constraints': None,
            'multi_strategy': None,
            'n_estimators': 75,
            'n_jobs': None,
            'num_parallel_tree': None,
            'objective': 'binary:logistic',
            'random_state': 42,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'sampling_method': None,
            'scale_pos_weight': 1,
            'subsample': 0.7,
            'tree_method': None,
            'validate_parameters': None,
            'verbosity': None
        }
    }
    model_eval = ModelEvaluator({})
    model_eval.print_comparisons(model_results)
    xg_boost_model = XGBoostModel({})
    xg_boost_model.best_estimator = best_estimator

    xg_boost_perfs = xg_boost_model.get_perf()

    # Create the comparison dataframe horizontally to match previous tables
    # Transpose (.T) each set so metrics become rows and sampling types become single columns
    xg_boost_comps = pd.concat([xg_boost_perfs[xgb_type].T for xgb_type in DF_TYPES], axis=1)
    xg_boost_comps.columns = [name.capitalize() for name in DF_TYPES]
    print('* xg_boost_comps *')
    print(xg_boost_comps)

    # ⚠ Pick the best model performance
    show_banner('📊 Final Model w/ Plot Confusion Matrix 📊')

    xg_boost_model.show_best(xg_boost_comps)
"""

# --- DEBUG --- #

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
        #run_debug(args.seed)

    gc.collect()

    print('\n')
    show_timer(main_start_time)

    # Write the run time at the end of file
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f'\n\n----- ⏱️ Run ID: {run_id} | Total Execution Time: {get_time(main_start_time)} ⏱️ -----')

    print(f'\n----- ⏱️ END RUN ID: {run_id} ⏱️ -----')
