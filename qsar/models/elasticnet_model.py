import pandas as pd
from optuna import Trial
from sklearn.linear_model import ElasticNet

from qsar.models.model import Model
from qsar.utils.cross_validator import CrossValidator


class ElasticnetModel(Model):
    def __init__(self, max_iter: int = Model.DEFAULT_MAX_ITER, random_state: int = Model.DEFAULT_RANDOM_STATE,
                 params=None):
        """
        A class used to represent an ElasticNet.

        :param max_iter: the maximum number of iterations for the model, defaults to Model.DEFAULT_MAX_ITER
        :type max_iter: int, optional
        :param random_state: the random state for the model, defaults to Model.DEFAULT_RANDOM_STATE
        :type random_state: int, optional
        :param params: the parameters for the model, defaults to None
        :type params: dict, optional
        """
        super().__init__()
        self.model = ElasticNet(max_iter=max_iter, random_state=random_state)
        self.params = params

    def optimize_hyperparameters(self, trial: Trial, df: pd.DataFrame) -> float:
        """
        Optimizes the hyperparameters of the ElasticNet Regressor model.

        :param trial: the trial for hyperparameter optimization
        :type trial: Trial
        :param df: the dataframe used for training the model
        :type df: pd.DataFrame
        :return: the cross validation score of the model
        :rtype: float
        """
        self.params = {
            "alpha": trial.suggest_float("alpha", 1e-10, 1e10, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 1e-10, 1, log=True),
        }

        self.model.set_params(**self.params)

        estimator = CrossValidator(df)
        return estimator.cross_value_score(self.model)
