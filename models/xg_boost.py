# models/xg_boost.py

import pandas as pd

from xgboost import XGBClassifier

from models.model_evaluator import ModelEvaluator
from src.config import (
    UNTUNED_ESTIMATOR_CNT,
    NODE_XGBOOST_CNT,
    DATASET_TYPES,
    UNTUNED_LEARNING_RATE,
    SEED
)
from src.utils import show_banner

class XGBoostModel(ModelEvaluator):
    def __init__(self, dataset: dict):
        super().__init__(dataset)

        self.title = 'XG Boost Classifier'
        self.model = self._create()
        self.params = self._get_search_cv_params()
        self.perf = pd.DataFrame()
        self.best_estimator = None

    def _create(self) -> XGBClassifier:
        # Used as a baseline
        return XGBClassifier(
            n_estimators=UNTUNED_ESTIMATOR_CNT,
            max_depth=NODE_XGBOOST_CNT,
            learning_rate= UNTUNED_LEARNING_RATE,
            reg_alpha=0.3,
            reg_lambda=0.3,
            random_state=SEED,
            scale_pos_weight=5.0
        )

    # Params for Randomized Search CV
    @staticmethod
    def _get_search_cv_params() -> dict:
        return {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.01, 0.03, 0.05],
            'max_depth': [3, 4],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.5, 0.6, 0.7],
            'reg_alpha': [1.0, 2.0, 5.0],
            'reg_lambda': [2.0, 5.0, 10.0],
            'scale_pos_weight': [1.0, 5.0]  # Only leave this here
        }

    def get_results(self, scorer) -> dict:
        results = super().get_results(scorer)

        # Now, store XGBoost models specifically for the final step
        # Need to store tuned_models here since the object is XGBClassifier.
        self.best_estimator = results['best_estimator']

        return results

    def get_perf(self) -> dict:
        """
        Gets best performance of the best estimators for XGBoost Performance
        :return:
        """

        perf = {}
        if self.best_estimator is not None:
            # Only iterate over valid sampling types to avoid metadata like 'title'.
            for xgb_type in DATASET_TYPES:
                model = self.best_estimator.get(xgb_type)
                if model is not None and not isinstance(model, dict):
                    perf[xgb_type] = self._get_model_perf(model, self.x_test, self.y_test)

        return perf

    def get_best(self, comp_models: pd.DataFrame):
        """
        xgb_models: pd.DataFrame

        Compares the three XGBoost models and returns the best one.
        """
        f1_scores = []

        # Get F1 Scores
        for model in comp_models.columns:
            f1_scores.append(comp_models[model]['F1'])

        # Get index and variable of the best F1 score
        best_model_index = f1_scores.index(max(f1_scores))
        best_model_title = comp_models.columns[best_model_index]

        # Extract the variant keyword (e.g., "Original", "Oversampled", "Undersampled") and match your lowercase
        # dictionary keys
        col_header = best_model_title.split()[-1].lower()

        # Look up using 'original', 'oversampled', or 'undersampled' instead of a numeric 0.
        best_model = self.best_estimator[col_header]

        show_banner('🏆 --- BEST XG BOOST MODEL --- 🏆', best_model_title)
        print(comp_models[best_model_title])

        return best_model

    def show_best(self, comp_models: pd.DataFrame) -> None:
        best_model = self.get_best(comp_models)
        best_perf = self._get_model_perf(best_model, self.x_test, self.y_test)

        print('\n# --- Best Model Performance --- #')
        print(best_perf)

        # Move import here to break circular dependency with src.eda
        from src.eda import plot_confusion_matrix

        print('\nShowing Plot Confusion Matrix of Best Model...')
        plot_confusion_matrix(best_model, self.x_test, self.y_test, 'Best XG Boost Model')
