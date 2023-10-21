from abc import ABC, abstractmethod

import pandas as pd
from optuna import Trial


class Model(ABC):
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
