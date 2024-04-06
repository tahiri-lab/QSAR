import pandas as pd
from optuna import Trial
from sklearn.linear_model import Ridge

from qsar.models.baseline_model import BaselineModel
from qsar.utils.cross_validator import CrossValidator


class RidgeModel(BaselineModel):
    """
    A class used to represent a RidgeModel, inheriting from the Model class. This class specifically handles the Ridge
    Regressor from the sklearn library.
    """

    def __init__(
            self,
            max_iter: int = BaselineModel.DEFAULT_MAX_ITER,
            random_state: int = BaselineModel.DEFAULT_RANDOM_STATE,
            params=None,
    ):
        """
        Initialize the RidgeModel with optional maximum iterations, random state, and model parameters.

        :param max_iter: the maximum number of iterations for the model, defaults to Model.DEFAULT_MAX_ITER
        :type max_iter: int, optional
        :param random_state: the random state for the model, defaults to Model.DEFAULT_RANDOM_STATE
        :type random_state: int, optional
        :param params: the parameters for the model, defaults to None
        :type params: dict, optional
        """
        super().__init__()
        self.model = Ridge(max_iter=max_iter, random_state=random_state)
        self.params = params
        if self.params is not None:
            self.model.set_params(**self.params)

    def optimize_hyperparameters(self, trial: Trial, df: pd.DataFrame) -> float:
        """
        Optimizes the hyperparameters of the Ridge Regressor model using a trial from Optuna.

        :param trial: the trial for hyperparameter optimization
        :type trial: Trial
        :param df: the dataframe used for training the model
        :type df: pd.DataFrame
        :return: the cross validation score of the model
        :rtype: float
        """

        self.params = {
            "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
            "solver": trial.suggest_categorical(
                "solver",
                ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            ),
        }

        self.model.set_params(**self.params)

        estimator = CrossValidator(df)
        return estimator.cross_value_score(self.model)
