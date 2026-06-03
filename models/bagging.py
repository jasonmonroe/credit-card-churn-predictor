# models/bagging.py

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from models.model_evaluator import ModelEvaluator
from src.config import SEED, BASE_ESTIMATOR_CNT


class BaggingModel(ModelEvaluator):
    def __init__(self, dataset: dict):
        super().__init__(dataset)

        self.title = 'Bagging'
        self.params = self._get_search_cv_params()
        self.model = self._create()

    def _create(self) -> BaggingClassifier:
        return BaggingClassifier(
            random_state=SEED,
            n_estimators=BASE_ESTIMATOR_CNT
        )

    @staticmethod
    def _get_search_cv_params() -> dict:
        return {
            'max_samples': [0.5, 0.7, 0.9],
            'max_features': [0.5, 0.7, 0.9],
            'n_estimators' : [30, 50, 70],
            'estimator': [
                DecisionTreeClassifier(max_depth=5, ccp_alpha=0.0, random_state=SEED),
                DecisionTreeClassifier(max_depth=10, ccp_alpha=0.01, random_state=SEED),
                None # <--- Default unlimited depth
            ]
        }
