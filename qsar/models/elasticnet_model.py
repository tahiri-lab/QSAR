import pandas as pd
from optuna import Trial
from sklearn.linear_model import ElasticNet

from qsar.models.model import Model
from qsar.utils import utils, extractor


class ElasticnetModel(Model):
    def __init__(self):
        super().__init__()
        self.model = ElasticNet()

    def optimize_hyperparameters(self, trial: Trial, df: pd.DataFrame) -> float:
        alpha = trial.suggest_float('alpha', 1e-10, 1e10, log=True)

        l1_ratio = trial.suggest_float('l1_ratio', 1e-10, 1, log=True)

        self.model = ElasticNet(max_iter=100000, alpha=alpha, l1_ratio=l1_ratio, random_state=0)

        estimator = utils.Utils(df)
        return estimator.cross_value_score(self.model)
