import pandas as pd
from optuna import Trial
from sklearn.linear_model import ElasticNet

from qsar.models.baseline_model import BaselineModel
from qsar.utils.cross_validator import CrossValidator


class ElasticnetModel(BaselineModel):
    """
    A class used to represent an ElasticNet model, inheriting from the Model class. This class specifically deals with
    the ElasticNet Regressor from the sklearn library.
    """

    def __init__(
        self,
        max_iter: int = BaselineModel.DEFAULT_MAX_ITER,
        random_state: int = BaselineModel.DEFAULT_RANDOM_STATE,
        params=None,
    ):
        """
        Initialize the ElasticnetModel with optional maximum iterations, random state, and model parameters.

        :param max_iter: The maximum number of iterations for the model. Defaults to Model.DEFAULT_MAX_ITER.
        :type max_iter: int, optional
        :param random_state: The random state for the model. Defaults to Model.DEFAULT_RANDOM_STATE.
        :type random_state: int, optional
        :param params: The parameters for the ElasticNet Regressor model. Defaults to None.
        :type params: dict, optional
        """
        super().__init__()
        self.model = ElasticNet(max_iter=max_iter, random_state=random_state)
        self.params = params

    def optimize_hyperparameters(self, trial: Trial, df: pd.DataFrame) -> float:
        """
        Optimizes the hyperparameters of the ElasticNet Regressor model using a trial from Optuna.

        :param trial: The trial instance for hyperparameter optimization.
        :type trial: optuna.Trial
        :param df: The DataFrame used for training the model.
        :type df: pd.DataFrame
        :return: The cross-validation score of the model.
        :rtype: float
        """
        self.params = {
            "alpha": trial.suggest_float("alpha", 1e-5, 2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 1e-2, 1, log=True),
        }

        self.model.set_params(**self.params)

        estimator = CrossValidator(df)
        return estimator.cross_value_score(self.model)
