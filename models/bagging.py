# models/bagging.py
from models.model_evaluator import ModelEvaluator
from sklearn.ensemble import BaggingClassifier
from src.config import SEED, BASE_ESTIMATOR_CNT


class BaggingModel(ModelEvaluator):
    def __init__(self, dataset: dict):
        super().__init__(dataset)

        self.title = 'Bagging'
        self.params = self._params()
        self.model = self._create()

    def _create(self) -> BaggingClassifier:
        return BaggingClassifier(
            random_state=SEED,
            n_estimators=BASE_ESTIMATOR_CNT
        )

    def _params(self) -> dict:
        return {
            'max_samples': [0.8,0.9,1],
            'max_features': [0.7,0.8,0.9],
            'n_estimators' : [30,50,70],
        }
