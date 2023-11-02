import pandas as pd
from optuna import Trial
from sklearn.linear_model import ElasticNet

from qsar.models.model import Model
from qsar.utils.cross_validator import CrossValidator


class ElasticnetModel(Model):
    def __init__(self, max_iter: int = Model.DEFAULT_MAX_ITER, random_state: int = Model.DEFAULT_RANDOM_STATE,
                 params=None):
        """
           Initialize the ElasticNet model.

           Parameters:
           - max_iter: Maximum number of iterations for convergence.
           - random_state: Seed for reproducibility.
        """
        super().__init__()
        self.model = ElasticNet(max_iter=max_iter, random_state=random_state)
        self.params = params

    def optimize_hyperparameters(self, trial: Trial, df: pd.DataFrame) -> float:
        """
        Optimize the hyperparameters of the ElasticNet model using the given trial and data.

        Parameters:
        - trial: Optuna trial for hyperparameter optimization.
        - df: Data for cross-validation.

        Returns:
        - Cross-validation score.
        """
        self.params = {
            "alpha": trial.suggest_float("alpha", 1e-10, 1e10, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 1e-10, 1, log=True),
        }

        self.model.set_params(**self.params)

        estimator = CrossValidator(df)
        return estimator.cross_value_score(self.model)
