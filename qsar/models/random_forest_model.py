import pandas as pd
from optuna import Trial
from sklearn.ensemble import RandomForestRegressor

from qsar.models.model import Model
from qsar.utils.cross_validator import CrossValidator


class RandomForestModel(Model):
    def __init__(self, random_state: int = Model.DEFAULT_RANDOM_STATE,
                 params=None):
        """
           Initialize the ElasticNet model.

           Parameters:
           - random_state: Seed for reproducibility.
        """
        super().__init__()
        self.model = RandomForestRegressor(random_state=random_state)
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
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
        }

        self.model.set_params(**self.params)

        estimator = CrossValidator(df)
        return estimator.cross_value_score(self.model)
