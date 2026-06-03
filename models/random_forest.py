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

    @staticmethod
    def _get_search_cv_params() -> dict:
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10, 12],       # Cap how deep the trees can grow
            'min_samples_split': [5, 10, 15],  # Higher minimum required data points to make a split
            'min_samples_leaf': [3, 5, 10],    # Forces leaves to hold more samples, smoothing out noise
            'max_features': ['sqrt', 'log2']   # Built-in fix for your earlier array error flag
        }
