# models/random_forest.py

import numpy as np

from models.model_evaluator import ModelEvaluator
from sklearn.ensemble import RandomForestClassifier
from src.config import (
    NODE_RFC_CNT,
    UNTUNED_ESTIMATOR_CNT,
    SEED
)


class RandomForestModel(ModelEvaluator):
    def __init__(self, dataset: dict):
        super().__init__(dataset)

        self.title = 'Random Forest Classifier'
        self.params = self._params()
        self.model = self._create()
        self.perf = []

    def _create(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            max_depth=NODE_RFC_CNT,
            n_estimators=UNTUNED_ESTIMATOR_CNT,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=SEED
        )

    def _params(self) -> dict:
        return {
            "n_estimators": [50,110,25],
            "min_samples_leaf": np.arange(1, 4),
            "max_features": [np.arange(0.3, 0.6, 0.1),'sqrt'],
            "max_samples": np.arange(0.4, 0.7, 0.1)
        }
