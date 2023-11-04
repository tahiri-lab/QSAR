import pandas as pd
from optuna import Trial

from qsar.models.model import Model
from sklearn.linear_model import Lasso

from qsar.utils.cross_validator import CrossValidator


class LassoModel(Model):
    def __init__(self, max_iter: int = Model.DEFAULT_MAX_ITER, random_state: int = Model.DEFAULT_RANDOM_STATE,
                 params=None):
        """
           Initialize the Lasso model.

           Parameters:
           - max_iter: Maximum number of iterations for convergence.
           - random_state: Seed for reproducibility.
        """
        super().__init__()
        self.model = Lasso(max_iter=max_iter, random_state=random_state)
        self.params = params

    def optimize_hyperparameters(self, trial: Trial, df: pd.DataFrame) -> float:
        """
        Optimize the hyperparameters of the Lasso model using the given trial and data.

        Parameters:
        - trial: Optuna trial for hyperparameter optimization.
        - df: Data for cross-validation.

        Returns:
        - Cross-validation score.
        """
        self.params = {
            "alpha": trial.suggest_float("alpha", 1e-10, 1e10, log=True),
            "tol": trial.suggest_float("tol", 1e-10, 1e-2, log=False),
            "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
        }
        
        self.model.set_params(**self.params)

        estimator = CrossValidator(df)
        return estimator.cross_value_score(self.model)
