from abc import ABC, abstractmethod

import pandas as pd
from optuna import Trial


class Model(ABC):
    DEFAULT_MAX_ITER = 100000
    DEFAULT_RANDOM_STATE = 0

    def __init__(self):
        """
        Abstract class for all models.
        """
        self.model = None

    @abstractmethod
    def optimize_hyperparameters(self, trial: Trial, df: pd.DataFrame) -> float:
        """
        Optimize the hyperparameters of the model.
        Parameters
        ----------
        trial : optuna.Trial object to be used for the optimization.
        df : pd.DataFrame to be used for the optimization.

        Returns
        -------
        float cross value score of the model.
        """
        pass

    def set_hyperparameters(self, **kwargs):
        """
        Set the hyperparameters of the model.
        Parameters
        ----------
        **kwargs : hyperparameters to be set.

        Returns
        -------
        None
        """
        self.model.set_params(**kwargs)

    def fit(self, X_train, y_train):
        """
        Fit the model with the given data.
        Parameters
        ----------
        X_train : pd.DataFrame of the train data.
        y_train : pd.DataFrame of the train data.

        Returns
        -------
        None
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict the given data.
        Parameters
        ----------
        X : pd.DataFrame of the data to be predicted.

        Returns
        -------
        pd.DataFrame of the predicted data.
        """
        return self.model.predict(X)
