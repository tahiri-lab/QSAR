import pandas as pd
from optuna import Trial
from xgboost import XGBRegressor

from qsar.models.model import Model
from qsar.utils.cross_validator import CrossValidator


class XGBoostModel(Model):
    def __init__(self, max_iter: int = Model.DEFAULT_MAX_ITER, random_state: int = Model.DEFAULT_RANDOM_STATE,
                 params=None):
        """
           Initialize the XGBoost model.

           Parameters:
           - max_iter: Maximum number of iterations for convergence.
           - random_state: Seed for reproducibility.
        """
        super().__init__()
        self.model = XGBRegressor(max_iter=max_iter, random_state=random_state)
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
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        }

        self.model.set_params(**self.params)

        estimator = CrossValidator(df)
        return estimator.cross_value_score(self.model)
