import pandas as pd
from optuna import Trial
from catboost import CatBoostRegressor

from qsar.models.baseline_model import BaselineModel
from qsar.utils.cross_validator import CrossValidator


class CatboostModel(BaselineModel):
    def __init__(self, max_iter: int = BaselineModel.DEFAULT_MAX_ITER, random_state: int = BaselineModel.DEFAULT_RANDOM_STATE,
                 params=None):
        """
           Initialize the XGBoost model.

           Parameters:
           - max_iter: Maximum number of iterations for convergence.
           - random_state: Seed for reproducibility.
        """
        super().__init__()
        self.model = CatBoostRegressor(random_state=random_state)
        self.params = params

    def optimize_hyperparameters(self, trial: Trial, df: pd.DataFrame) -> float:
        """
        Optimize the hyperparameters of the XGBoost model using the given trial and data.

        Parameters:
        - trial: Optuna trial for hyperparameter optimization.
        - df: Data for cross-validation.

        Returns:
        - Cross-validation score.
        """
        self.params = {
            "objective": trial.suggest_categorical("objective", ["RMSE", "MAE", "Poisson", "Quantile", "MAPE"]),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        }

        self.model.set_params(**self.params)

        estimator = CrossValidator(df)
        return estimator.cross_value_score(self.model)
