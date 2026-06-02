# models/bagging.py
from models.model_evaluator import ModelEvaluator
from sklearn.ensemble import BaggingClassifier
from src.config import SEED, BASE_ESTIMATOR_CNT


class BaggingModel(ModelEvaluator):
    def __init__(self, dataset: dict):
        super().__init__(dataset)

        self.title = 'Bagging'
        self.model = self._create()
        #self.perf = []



    def _create(self) -> BaggingClassifier:
        return BaggingClassifier(
            random_state=SEED,
            n_estimators=BASE_ESTIMATOR_CNT
        )


