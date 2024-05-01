import pandas as pd
from optuna import Trial
from xgboost import XGBRegressor

from qsar.models.baseline_model import BaselineModel
from qsar.utils.cross_validator import CrossValidator


class XgboostModel(BaselineModel):
    """
    A class used to represent a XGBoostModel, inheriting from the Model class. This class specifically handles the
    XGBoost Regressor from the xgboost library.
    """

    def __init__(
            self,
            max_iter: int = BaselineModel.DEFAULT_MAX_ITER,
            random_state: int = BaselineModel.DEFAULT_RANDOM_STATE,
            params=None,
    ):
        """
        Initialize the XGBoostModel with optional maximum iterations, random state, and model parameters.

        :param max_iter: the maximum number of iterations for the model, defaults to Model.DEFAULT_MAX_ITER
        :type max_iter: int, optional
        :param random_state: the random state for the model, defaults to Model.DEFAULT_RANDOM_STATE
        :type random_state: int, optional
        :param params: the parameters for the model, defaults to None
        :type params: dict, optional
        """
        super().__init__()
        self.model = XGBRegressor(random_state=random_state, n_estimators=100)
        self.params = params

    def optimize_hyperparameters(self, trial: Trial, df: pd.DataFrame) -> float:
        """
        Optimizes the hyperparameters of the XGBoost Regressor model using a trial from Optuna.

        :param trial: the trial for hyperparameter optimization
        :type trial: Trial
        :param df: the dataframe used for training the model
        :type df: pd.DataFrame
        :return: the cross validation score of the model
        :rtype: float
        """

        self.params = {
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.01, 1.0),
            "subsample": trial.suggest_float("subsample", 0.01, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0),
        }

        self.model.set_params(**self.params)

        estimator = CrossValidator(df)
        return estimator.cross_value_score(self.model)
