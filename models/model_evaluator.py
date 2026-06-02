# models/model_evaluator.py

from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from pandas import DataFrame
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from xgboost import XGBClassifier

from src.config import CV_FOLDS, DF_TYPES, MAX_PROC_THREADS, OUTPUT_FILE, PARAM_DISTR_CNT, SEED
from src.utils import show_timer, start_timer


class ModelEvaluator:
    def __init__(self, dataset: dict):
        self.title = ''
        self.model = None
        self.params = {}
        self.perf = pd.DataFrame()
        self.sampled = {}

        # Scores
        self.orig = {'train': 0.0, 'val': 0.0}
        self.oversample = {'train': 0.0, 'val': 0.0}
        self.undersample = {'train': 0.0, 'val': 0.0}

        self.x_train = pd.DataFrame()
        self.y_train = pd.Series()
        self.x_val = pd.DataFrame()
        self.y_val = pd.Series()
        self.x_test = pd.DataFrame()
        self.y_test = pd.Series()

        # Sampled data
        self.x_over = pd.DataFrame()
        self.y_over = pd.Series()
        self.x_under = pd.DataFrame()
        self.y_under = pd.Series()

        self._set_attrs(dataset)

    def _set_attrs(self, dataset):
        for key, value in dataset.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_sampled(self) -> dict:
        x_over, y_over = self._get_oversampled()
        x_under, y_under = self._get_undersampled()

        return {
            'x_over': x_over,
            'y_over': y_over,
            'x_under': x_under,
            'y_under': y_under
        }

    def _get_oversampled(self):
        # Synthetic Minority Over Sampling Technique
        sm = SMOTE(
            sampling_strategy=1,
            k_neighbors=5,
            random_state=SEED
        )

        return sm.fit_resample(self.x_train, self.y_train)

    def _get_undersampled(self):
        # Random under sampler for under sampling the data
        rus = RandomUnderSampler(sampling_strategy=1, random_state=SEED)
        return rus.fit_resample(self.x_train, self.y_train)

    def run(self):

        # Get original, oversamples and undersampled datasets
        print(f'\n--- Running {self.title} Model Performances ---')

        self.run_orig()
        self.run_oversampled()
        self.run_undersampled()

    def run_orig(self):
        print(f'\n* {self.title} Original Data *')
        self.orig['train'] = self._fit_model(self.x_train, self.y_train, self.x_train, self.y_train)
        self.orig['val'] = self._fit_model(self.x_train, self.y_train, self.x_val, self.y_val)
        print(self.orig)
        self.show_classification_model_perf(self.x_train, self.y_train)

    def run_oversampled(self):
        print(f'\n* {self.title} Oversampled Data *')

        self.oversample['train'] = self._fit_model(self.x_over, self.y_over, self.x_over, self.y_over)
        self.oversample['val'] = self._fit_model(self.x_over, self.y_over, self.x_val, self.y_val)
        print(f'Training: {self.oversample["train"]}')
        print(f'Validation: {self.oversample["val"]}')
        self.show_classification_model_perf(self.x_over, self.y_over)

    def run_undersampled(self):
        print(f'\n* {self.title} Undersampled Data *')

        self.undersample['train'] = self._fit_model(self.x_under, self.y_under, self.x_under, self.y_under)
        self.undersample['val'] = self._fit_model(self.x_under, self.y_under, self.x_val, self.y_val)
        print(f'Training: {self.undersample["train"]}')
        print(f'Validation: {self.undersample["val"]}')
        self.show_classification_model_perf(self.x_under, self.y_under)

    def show_classification_model_perf(self, x, y):
        print('\n** 🖥️ Model Classification 🖥️ **')
        self.perf = self._get_model_perf(self.model, x, y)
        print(self.perf)

    def _fit_model(self, x_fit, y_fit, x, y) -> float:
        self.model.fit(x_fit, y_fit)
        score = recall_score(y, self.model.predict(x))

        return score

    def _tune(self, scorer) -> BaseEstimator:
        """
        Helper function to perform RandomizedSearchCV, fit the best model, and calculate scores.
        
        Returns:
            BaseEstimator: The best fitted model found during search.
        """
        total_params = len(ParameterGrid(self.params))
        n_iter = min(PARAM_DISTR_CNT, total_params)

        randomized_cv = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.params,
            n_iter=n_iter,
            n_jobs=MAX_PROC_THREADS,
            scoring=scorer,
            cv=CV_FOLDS,
            random_state=SEED
        )

        # Fit model
        randomized_cv.fit(self.x_train, self.y_train)

        print(f'💡 CV Score: {randomized_cv.best_score_}')
        print('✅ Best parameters are: ')
        for key, value in randomized_cv.best_params_.items():
            print(f"\t{key}: {value}")

        return randomized_cv.best_estimator_

    def get_results(self, scorer) -> dict[str, Any]:
        model_titles, train_results, val_results = [], [], []

        xgb = {'title': ''}
        for df_type in DF_TYPES:
            title = f'{self.title} {df_type.title()}'
            print(f'\n🔧 --- Tuning {title} Data ---')

            start_time = start_timer()
            tuned_model = self._tune(scorer)
            show_timer(start_time)

            train_perf = self._get_model_perf(tuned_model, self.x_train, self.y_train)
            val_perf = self._get_model_perf(tuned_model, self.x_val, self.y_val)

            train_results.append(train_perf)
            val_results.append(val_perf)
            model_titles.append(f'{title}')

            if isinstance(self.model, XGBClassifier):
                print('💡 DEBUG: instance is XGBClassifier')
                xgb['title'] += title+' '
                xgb[df_type] = tuned_model
            else:
                print(f'❌ DEBUG: {type(self.model).__name__} is not XGBClassifier')

        # --- debug ---
        import pprint

        clean_mock_dump = {}

        for strategy, model_obj in xgb.items():
            # 1. Pull the raw parameter dictionary out of the estimator object
            params = model_obj.get_params()

            # 2. Fix the non-serializable objects (like NumPy types or float('nan'))
            for key, val in list(params.items()):
                # Convert np.int64 or np.float64 to native Python int/float
                if hasattr(val, 'item'):
                    params[key] = val.item()
                # Convert true float NaN to a safe string or None for easy testing
                elif isinstance(val, float) and str(val) == 'nan':
                    params[key] = None

            clean_mock_dump[strategy] = params

        # Print the sanitized dictionary layout
        print("CLEAN_HARDCODED_ESTIMATORS = ")
        pprint.pprint(clean_mock_dump, indent=4, width=120)

        # --- debug ---
        # Concatenate the list of DataFrames into a single DataFrame for each set
        # We use axis=0 to stack 'Original', 'Oversampled', and 'Undersampled' vertically
        df_train = pd.concat(train_results, axis=0) if train_results else pd.DataFrame()
        df_val = pd.concat(val_results, axis=0) if val_results else pd.DataFrame()

        # Explicitly use Any to allow DataFrames, lists, and dicts (xgb)
        results: dict[str, Any] = {
            'train': df_train,
            'val': df_val,
            'titles': model_titles,

            # Note: Only append best_estimator if xgb is populated and matches its model!
            'best_estimator': xgb if isinstance(self.model, XGBClassifier) else None
        }

        return results

    def _format_results(self, results: list) -> tuple[DataFrame, DataFrame]:
        """
        Helper function to format the results dictionary into a more structured format.
        This is optional and can be adjusted based on how you want to present the results.
        Take the training and validation data for each model that used original, oversampled and undersampled data and
        create a matrix for viewing.
        Reassign indices cleanly inside the looping block so .loc maps perfectly
        """
        train_cols, val_cols, title_cols = [],[],[]

        for result in results:
            train_cols.append(self._flatten(result, 'train'))
            val_cols.append(self._flatten(result, 'val'))
            title_cols.extend(result['titles'])

        # Combine all (model) training and validation data into one
        training_models = pd.concat(train_cols, axis=1)
        training_models.columns = title_cols

        val_models = pd.concat(val_cols, axis=1)
        #val_models.columns = title_cols # @todo - why are we adding value below?
        val_models.columns = [f"{t} Value" for t in title_cols]

        return training_models, val_models

    @staticmethod
    def _flatten(result, key: str):
        df_train = result[key].copy()
        df_train.index = result['titles']
        return df_train.loc[result['titles']].T

    def print_comparisons(self, results: list) -> None:

        # Format Results
        training_models, val_models = self._format_results(results)

        # Show Training and Validation Comparison for each model
        print('\n--- ⚙️ Model Training Comparisons ⚙️ ---')
        df_train_long = training_models.T
        print(df_train_long.to_string())

        print('\n--- ☑️️ Model Validation Comparisons ☑️️ ---')
        df_val_long = val_models.T
        print(df_val_long.to_string())

        proj_title_str = ''
        proj_title_str += '\t\t\t\t\t\t+-----------------------------------+\n\t\t\t\t\t\t|'
        proj_title_str += ' 💳️ CREDIT CARD CHURN PREDICTOR 💳️ '
        proj_title_str += '|\n\t\t\t\t\t\t+-----------------------------------+'

        #proj_title_str = '\t\t\t\t\t\t+-----------------------------------+\n\t\t\t\t\t\t| ⚙️ CREDIT CARD CHURN PREDICTOR ⚙️ |\n\t\t\t\t\t\t+-----------------------------------+'

        # Print Results to file
        with open(OUTPUT_FILE, 'w') as f:
            f.write(proj_title_str)
            f.write("\n\n")
            f.write('----------------------- ⚙️ Model Training Comparisons ⚙️ ------------------------\n')
            f.write(df_train_long.to_string())
            f.write('\n----------------------------------------------------------------------------------')
            f.write("\n\n")
            f.write('------------------------- ⚙️ Model Validation Comparisons ⚙️ --------------------------\n')
            f.write(df_val_long.to_string())
            f.write('\n----------------------------------------------------------------------------------------')


    @staticmethod
    def _get_model_perf(model, predictors, target) -> pd.DataFrame:
        """
        Function to compute different metrics to check classification model performance

        model: classifier
        predictors: independent variables
        target: dependent variable
        """

        # Predicting using the independent variables of the tuned model
        pred = model.predict(predictors)

        acc = accuracy_score(target, pred) # to compute Accuracy
        recall = recall_score(target, pred) # to compute Recall
        precision = precision_score(target, pred) # to compute Precision
        f1 = f1_score(target, pred) # to compute F1-score

        # Creating a dataframe of metrics
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
