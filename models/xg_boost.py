# models/xg_boost.py

import numpy as np
import pandas as pd
from src.eda import plot_confusion_matrix
from src.utils import show_banner
from xgboost import XGBClassifier

from models.model_evaluator import ModelEvaluator
from src.config import (
    UNTUNED_ESTIMATOR_CNT,
    NODE_XGBOOST_CNT,
    UNTUNED_LEARNING_RATE,
    SEED
)


class XGBoostModel(ModelEvaluator):
    def __init__(self, dataset: dict):
        super().__init__(dataset)

        self.title = 'XG Boost Classifier'
        self.model = self._create()
        self.params = self.get_params()
        self.perf = []
        self.best_estimator = None

    def _create(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=UNTUNED_ESTIMATOR_CNT,
            max_depth=NODE_XGBOOST_CNT,
            learning_rate= UNTUNED_LEARNING_RATE,
            reg_alpha=0.3,
            reg_lambda=0.3,
            random_state=SEED
        )

    def get_params(self) -> dict:
        return {
            'n_estimators': np.arange(50, 110, 25),
            'scale_pos_weight': [1, 2, 5],
            'learning_rate': [0.01, 0.1, 0.05],
            'gamma': [1, 3, 5],
            'subsample': [0.7, 0.9]
        }

    def get_results(self, scorer) -> dict:
        results = super().get_results(scorer)

        # Now, store XGBoost models specifically for the final step
        # Need to store tuned_models here since the object is XGBClassifier.
        self.best_estimator = results['best_estimator']
        #self.best_estimator = best_estimator

        return results

    def get_perf(self) -> dict:
        """
        Gets best performance of the best estimators for XGBoost Performance
        :return:
        """

        perf = {}
        if self.best_estimator is not None:
            for xgb_type in self.best_estimator:
                perf[xgb_type] = self._get_model_perf(self.best_estimator[xgb_type], self.x_test, self.y_test)

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
        best_model = self.best_estimator[best_model_index]

        show_banner('--- 🏆 BEST XG BOOST MODEL 🏆--- ', best_model_title)
        print(comp_models[best_model_title])
        print(f'best_model type: {type(best_model)}')

        return best_model

    def show_best(self, comp_models: pd.DataFrame) -> None:
        best_model = self.get_best(comp_models)
        best_perf = self._get_model_perf(best_model, self.x_test, self.y_test)

        print('--- Best Model Performance ---')
        print(best_perf)

        plot_confusion_matrix(best_model, self.x_test, self.y_test)


