"""
The BaselineModel leverages the Optuna library for hyperparameter optimization, facilitating an automated and
efficient search for the best model parameters based on the provided data. This approach allows for the rapid
development and comparison of different QSAR modeling strategies, enhancing the flexibility and robustness of
the modeling process.
"""

from abc import ABC, abstractmethod

import pandas as pd
from optuna import Trial


class BaselineModel(ABC):
    """
    An abstract base class used to represent a QSAR model. It serves as a template for all QSAR models and uses the
    Optuna library for hyperparameter optimization. Child classes should implement the specific model logic and
    hyperparameter optimization.
    """

    DEFAULT_MAX_ITER = 100000
    DEFAULT_RANDOM_STATE = 0

    def __init__(self):
        """
        Initializes the Model object with default parameters.
        """
        self.model = None
        self.params = None

    @abstractmethod
    def optimize_hyperparameters(self, trial: Trial, df: pd.DataFrame) -> float:
        """
        An abstract method that should be overridden in child classes to optimize the hyperparameters of the model.

        :param trial: The trial instance for hyperparameter optimization.
        :type trial: optuna.Trial
        :param df: The DataFrame used for training the model.
        :type df: pd.DataFrame
        :return: The cross-validation score of the model.
        :rtype: float
        """

    def set_hyperparameters(self, **kwargs):
        """
        Sets the hyperparameters of the model.

        :param kwargs: the hyperparameters to set
        :type kwargs: dict
        """
        self.model.set_params(**kwargs)

    def fit(self, x_train, y_train):
        """
        Fits the model to the training data.

        :param x_train: the training data
        :type x_train: pd.DataFrame or np.ndarray
        :param y_train: the target values for the training data
        :type y_train: pd.Series or np.ndarray
        """
        self.model.fit(x_train, y_train)

    def predict(self, x):
        """
        Predicts the target values for the given data.

        :param x: the data to predict the target values for
        :type x: pd.DataFrame or np.ndarray
        :return: the predicted target values
        :rtype: np.ndarray
        """
        return self.model.predict(x)
