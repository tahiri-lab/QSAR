import pandas as pd
from optuna import Trial
from sklearn.linear_model import Ridge

from qsar.models.baseline_model import Model
from qsar.utils.cross_validator import CrossValidator


class RidgeModel(Model):
    def __init__(self, max_iter: int = Model.DEFAULT_MAX_ITER, random_state: int = Model.DEFAULT_RANDOM_STATE,
                 params=None):
        """
           Initialize the Ridge model.

           Parameters:
           - max_iter: Maximum number of iterations for convergence.
           - random_state: Seed for reproducibility.
        """
        super().__init__()
        self.model = Ridge(max_iter=max_iter, random_state=random_state)
        self.params = params

    def optimize_hyperparameters(self, trial: Trial, df: pd.DataFrame) -> float:
        """
        Optimize the hyperparameters of the Ridge model using the given trial and data.

        Parameters:
        - trial: Optuna trial for hyperparameter optimization.
        - df: Data for cross-validation.

        Returns:
        - Cross-validation score.
        """
        self.params = {
            "alpha": trial.suggest_float('alpha', 1e-10, 1e10, log=True),
            "solver": trial.suggest_categorical('solver',
                                                ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]),
        }

        self.model.set_params(**self.params)

        estimator = CrossValidator(df)
        return estimator.cross_value_score(self.model)
