import pandas as pd
from catboost import CatBoostRegressor
from optuna import Trial

from qsar.models.baseline_model import BaselineModel
from qsar.utils.cross_validator import CrossValidator


class CatboostModel(BaselineModel):
    """
    A class used to represent a CatboostModel which inherits from the Model class. This class is specific for handling
    CatBoost Regressor models.
    """

    def __init__(
            self,
            max_iter: int = BaselineModel.DEFAULT_MAX_ITER,
            random_state: int = BaselineModel.DEFAULT_RANDOM_STATE,
            params=None,
    ):
        """
        Initialize the CatboostModel with optional maximum iterations, random state, and model parameters.

        :param max_iter: The maximum number of iterations for the model. Defaults to Model.DEFAULT_MAX_ITER.
        :type max_iter: int, optional
        :param random_state: The random state for the model. Defaults to Model.DEFAULT_RANDOM_STATE.
        :type random_state: int, optional
        :param params: The parameters for the CatBoost Regressor model. Defaults to None.
        :type params: dict, optional
        """
        super().__init__()
        self.model = CatBoostRegressor(random_state=random_state)
        self.params = params
        self.max_iter = max_iter

    def optimize_hyperparameters(self, trial: Trial, df: pd.DataFrame) -> float:
        """
        Optimizes the hyperparameters of the CatBoost Regressor model using a trial from Optuna.

        :param trial: the trial for hyperparameter optimization
        :type trial: Trial
        :param df: the dataframe used for training the model
        :type df: pd.DataFrame
        :return: the cross validation score of the model
        :rtype: float
        """
        self.params = {
            "objective": trial.suggest_categorical(
                "objective", ["RMSE", "MAE", "Poisson", "Quantile", "MAPE"]
            ),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1, log=True),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["Ordered", "Plain"]
            ),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
        }
        self.model.set_params(**self.params)

        estimator = CrossValidator(df)
        return estimator.cross_value_score(self.model)
