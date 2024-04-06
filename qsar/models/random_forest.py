import pandas as pd
from optuna import Trial
from sklearn.ensemble import RandomForestRegressor

from qsar.models.baseline_model import BaselineModel
from qsar.utils.cross_validator import CrossValidator


class RandomForestModel(BaselineModel):
    """
    A class used to represent a RandomForestModel, inheriting from the Model class. This class specifically handles the
    RandomForest Regressor from the sklearn library.
    """

    def __init__(
            self, random_state: int = BaselineModel.DEFAULT_RANDOM_STATE, params=None
    ):
        """
        Initialize the RandomForestModel with an optional random state and model parameters.

        :param max_iter: the maximum number of iterations for the model, defaults to Model.DEFAULT_MAX_ITER
        :type max_iter: int, optional
        :param random_state: the random state for the model, defaults to Model.DEFAULT_RANDOM_STATE
        :type random_state: int, optional
        :param params: the parameters for the model, defaults to None
        :type params: dict, optional
        """
        super().__init__()
        self.model = RandomForestRegressor(random_state=random_state)
        self.params = params

    def optimize_hyperparameters(self, trial: Trial, df: pd.DataFrame) -> float:
        """
        Optimizes the hyperparameters of the RandomForest Regressor model.

        :param trial: the trial for hyperparameter optimization
        :type trial: Trial
        :param df: the dataframe used for training the model
        :type df: pd.DataFrame
        :return: the cross validation score of the model
        :rtype: float
        """

        self.params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 100),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 150),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
        }

        self.model.set_params(**self.params)

        estimator = CrossValidator(df)
        return estimator.cross_value_score(self.model)
