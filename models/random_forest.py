# models/random_forest.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from models.model_evaluator import ModelEvaluator
from src.config import (
    NODE_RFC_CNT, SEED, UNTUNED_ESTIMATOR_CNT
)


class RandomForestModel(ModelEvaluator):
    def __init__(self, dataset: dict):
        super().__init__(dataset)

        self.title = 'Random Forest Classifier'
        self.params = self._get_search_cv_params()
        self.model = self._create()
        self.perf = pd.DataFrame()

    def _create(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            max_depth=NODE_RFC_CNT,
            n_estimators=UNTUNED_ESTIMATOR_CNT,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=SEED
        )

    def _get_search_cv_params(self) -> dict:
        return {
            # Using np.arange creates a list of candidates to sample from
            "n_estimators": np.arange(50, 125, 25),
            "min_samples_leaf": np.arange(1, 5),
            # To mix floats and strings, convert the array to a list and extend it
            "max_features": np.arange(0.3, 0.6, 0.1).tolist() + ['sqrt'],
            "max_samples": np.arange(0.4, 0.7, 0.1)
        }
