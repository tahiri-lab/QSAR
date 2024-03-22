"""
The class encapsulates all necessary components for conducting an extensive search over a predefined hyperparameter
space, handling the iterative trial-and-error process automatically and recording the outcomes for analysis. The
optimization process aims either to maximize or minimize a given metric, depending on the user's requirements and the
nature of the QSAR problem being addressed.
"""

import optuna

from qsar.models.baseline_model import BaselineModel


class HyperParameterOptimizer:
    """
    A class for optimizing hyperparameters of a given model using Optuna.

    This optimizer handles the process of finding the best hyperparameters for a model, using the Optuna library for
    efficient optimization. It is designed to work with models that inherit from the 'Model' class and implement an
    'optimize_hyperparameters' method.

    :ivar DEFAULT_TRIALS: Default number of trials for optimization (class attribute).
    :vartype DEFAULT_TRIALS: int
    :ivar model: Model instance to be optimized.
    :vartype model: Model
    :ivar data: Training data on which the optimization is performed.
    :ivar trials: Number of optimization trials.
    :vartype trials: int
    :ivar direction: Direction for optimization, either 'maximize' or 'minimize'.
    :vartype direction: str
    """

    DEFAULT_TRIALS = 1000

    def __init__(
        self, model: BaselineModel, data, trials=DEFAULT_TRIALS, direction="maximize"
    ):
        """
        Initialize the HyperParameterOptimizer.

        :param model: The model to be optimized. This model should inherit from 'Model' and should implement
        'optimize_hyperparameters'.
        :type model: Model
        :param data: The training data.
        :type data: Varies (specify the expected data type)
        :param trials: Number of optimization trials. Defaults to 1000.
        :type trials: int
        :param direction: Direction for the optimization ('maximize' or 'minimize'). Defaults to 'maximize'.
        :type direction: str
        """
        self.model = model
        self.data = data
        self.trials = trials
        self.direction = direction

    def _objective(self, trial) -> float:
        """
        Objective function for the optimization process.

        :param trial: A trial instance of the optuna optimization process.
        :type trial: optuna.trial.Trial
        :returns: The evaluation metric value for a set of hyperparameters.
        :rtype: float
        """
        return self.model.optimize_hyperparameters(trial, self.data)

    def optimize(self) -> optuna.study.Study:
        """
        Perform hyperparameter optimization.

        :returns: The optuna study object containing the results of the optimization.
        :rtype: optuna.study.Study
        """
        study = optuna.create_study(direction=self.direction)
        study.optimize(
            self._objective, n_trials=self.trials, n_jobs=-1, show_progress_bar=True
        )

        return study
