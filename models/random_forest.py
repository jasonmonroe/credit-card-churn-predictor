# models/random_forest.py
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