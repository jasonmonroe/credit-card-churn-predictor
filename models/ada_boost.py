# models/ada_boost.py
import numpy as np

from models.model_evaluator import ModelEvaluator
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from src.config import SEED, UNTUNED_LEARNING_RATE


class AdaBoostModel(ModelEvaluator):
    def __init__(self, dataset: dict):
        super().__init__(dataset)
        self.title = 'ADA Boost Classifier'
        self.model = self._create()
        self.params = self._get_search_cv_params()
        self.perf = []

    def _create(self) -> AdaBoostClassifier:
        return AdaBoostClassifier(
            random_state=SEED,
            learning_rate=UNTUNED_LEARNING_RATE
        )

    def _get_search_cv_params(self) -> dict:
        return {
            "n_estimators": np.arange(50, 110, 25),
            "learning_rate": [0.01, 0.1, 0.05],
            "estimator": [
                DecisionTreeClassifier(max_depth=2, ccp_alpha=0.0, random_state=SEED),
                DecisionTreeClassifier(max_depth=2, ccp_alpha=0.01, random_state=SEED),
                DecisionTreeClassifier(max_depth=3, ccp_alpha=0.0, random_state=SEED),
                DecisionTreeClassifier(max_depth=3, ccp_alpha=0.01, random_state=SEED),
            ],
        }