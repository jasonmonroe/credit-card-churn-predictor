# models/model_evaluator.py

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from notebooks.credit_card_churn_predictor_notebook import show_timer
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from src.config import SEED, PARAM_DISTR_CNT, MAX_PROC_THREADS, CV_FOLDS, DF_TYPES
from src.utils import start_timer
from xgboost import XGBClassifier


class ModelEvaluator():
    def __init__(self, dataset: dict):
        self.title = ''
        self.model = None
        self.params = {}
        self.perf = []
        self.sampled = {}

        #self.dataset = dataset

        # Scores
        self.orig = {'train': 0.0, 'val': 0.0}
        self.oversample = {'train': 0.0, 'val': 0.0}
        self.undersample = {'train': 0.0, 'val': 0.0}

        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []

        # Sampled data
        self.x_over = []
        self.y_over = []
        self.x_under = []
        self.y_under = []

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
        print(f'\n-- {self.title} Original Data --')
        self.orig['train'] = self._fit_model(self.x_train, self.y_train, self.x_train, self.y_train)
        self.orig['val'] = self._fit_model(self.x_train, self.y_train, self.x_val, self.y_val)

        self.show_classification_model_perf(self.x_train, self.y_train)

    def run_oversampled(self):
        print(f'\n-- {self.title} Oversampled Data --')

        self.oversample['train'] = self._fit_model(self.x_over, self.y_over, self.x_over, self.y_over)
        self.oversample['val'] = self._fit_model(self.x_over, self.y_over, self.x_val, self.y_val)
        self.show_classification_model_perf(self.x_over, self.y_over)

    def run_undersampled(self):
        print(f'\n-- {self.title} Undersampled Data --')
        #x_us, y_us = self._get_undersampled()

        self.undersample['train'] = self._fit_model(self.x_under, self.y_under, self.x_under, self.y_under)
        self.undersample['val'] = self._fit_model(self.x_under, self.y_under, self.x_val, self.y_val)
        self.show_classification_model_perf(self.x_under, self.y_under)

    def show_classification_model_perf(self, x, y):
        print('- Model Classification -')
        self.perf = self._get_model_perf(self.model, x, y)
        print(self.perf)

    def _fit_model(self, x_fit, y_fit, x, y) -> float:
        self.model.fit(x_fit, y_fit)
        score = recall_score(y, self.model.predict(x))
        print(f'{self.title}: {score}')

        return score

    def _tune(self, scorer):
        """
        Helper function to perform RandomizedSearchCV, fit the best model, and calculate scores.
        Calculate total parameter space size
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

        #print(f'Best parameters are {randomized_cv.best_params_} with CV score={randomized_cv.best_score_}:')
        print(f'CV Score: {randomized_cv.best_score_}')
        print('Best parameter are: ')
        for key, value in randomized_cv.best_params_.items():
            print(f"\t{key}: {value}")

        # Re-instantiate or use best_estimator_ directly.
        # Using best_estimator_ is safer as it contains the fitted model with best params.
        return randomized_cv.best_estimator_

    def get_results(self, scorer) -> dict:
        model_titles = []
        train_results = []
        val_results = []

        xgb = {}
        for df_type in DF_TYPES:
            title = f'{self.title } {df_type.title()}'
            print(f'\n--- Tuning {title} Data ---')

            start_time = start_timer()
            tuned_model = self._tune(scorer)
            show_timer(start_time)
            train_perf = self._get_train_perf(tuned_model)
            val_perf = self._get_val_perf(tuned_model)

            train_results.append(train_perf)
            val_results.append(val_perf)
            model_titles.append(f'{title}')

            if isinstance(self.model, XGBClassifier):
                xgb[df_type] = tuned_model

        results =  {
            'train': train_results,
            'val': val_results,
            'titles': model_titles
        }

        # If we're utilizing the XGBClassifier Model then return that as well
        if isinstance(self.model, XGBClassifier):
            print('DEBUG: storing xgb in results as best_estimator!')
            results['best_estimator'] = xgb

        return results

    def _get_train_perf(self, model):
        return self._get_model_perf(model, self.x_train, self.y_train)

    def _get_val_perf(self, model):
        return self._get_model_perf(model, self.x_val, self.y_val)

    # Defining a function to compute different metrics to check performance of a classification model built using sklearn
    def _get_model_perf(self, model, predictors, target) -> pd.DataFrame:
        """
        Function to compute different metrics to check classification model performance

        model: classifier
        predictors: independent variables
        target: dependent variable
        """

        # predicting using the independent variables of the tuned model
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
