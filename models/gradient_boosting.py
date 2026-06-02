# models/gradient_boosting.py

import numpy as np

from models.model_evaluator import ModelEvaluator
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from src.config import SEED, UNTUNED_LEARNING_RATE


class GradientBoostingModel(ModelEvaluator):
    def __init__(self, dataset: dict):
        super().__init__(dataset)
        self.title = 'Gradient Boosting Classifier'
        self.model = self._create()
        self.params = self._get_search_cv_params()
        self.perf = []

    def _create(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            random_state=SEED,
            learning_rate=UNTUNED_LEARNING_RATE
        )

    def _get_search_cv_params(self) -> dict:
        return {
            "init": [
                AdaBoostClassifier(learning_rate=UNTUNED_LEARNING_RATE, random_state=SEED),
                DecisionTreeClassifier(ccp_alpha=0.0, random_state=SEED)
            ],
            "n_estimators": np.arange(50, 110, 25),
            "learning_rate": [0.01, 0.1, 0.05],
            "subsample": [0.7, 0.9],
            "max_features": [0.5, 0.7, 1],
        }
