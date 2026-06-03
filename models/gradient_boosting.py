# models/gradient_boosting.py

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from models.model_evaluator import ModelEvaluator
from src.config import SEED, UNTUNED_LEARNING_RATE


class GradientBoostingModel(ModelEvaluator):
    def __init__(self, dataset: dict):
        super().__init__(dataset)
        self.title = 'Gradient Boosting Classifier'
        self.model = self._create()
        self.params = self._get_search_cv_params()
        self.perf = pd.DataFrame()

    def _create(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            random_state=SEED,
            learning_rate=UNTUNED_LEARNING_RATE
        )

    @staticmethod
    def _get_search_cv_params() -> dict:
        return {
            'n_estimators': [100, 150],
            'learning_rate': [0.01, 0.05],
            'max_depth': [3, 4],
            'subsample': [0.6, 0.7, 0.8],
            'min_samples_split': [15, 20, 30],
            'min_samples_leaf': [10, 15, 20]
        }
