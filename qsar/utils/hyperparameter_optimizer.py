import optuna

from qsar.models.model import Model


class HyperParameterOptimizer:
    DEFAULT_TRIALS = 1000

    def __init__(self, model: Model, data, trials=DEFAULT_TRIALS, direction='maximize'):
        """
        Initialize the HyperParameterOptimizer.

        Parameters:
        - model: The model to be optimized. Its class should inherit from 'Model' and should implement
        'optimize_hyperparameters'.
        - data: The training data.
        - trials: Number of optimization trials.
        - direction: Direction for the optimization ('maximize' or 'minimize').
        """
        self.model = model
        self.data = data
        self.trials = trials
        self.direction = direction

    def _objective(self, trial) -> float:
        return self.model.optimize_hyperparameters(trial, self.data)

    def optimize(self) -> optuna.study.Study:
        """
        Perform hyperparameter optimization.

        Returns:
        - study: The optuna study object containing the results of the optimization.
        """
        study = optuna.create_study(direction=self.direction)
        study.optimize(self._objective, n_trials=self.trials, n_jobs=-1, show_progress_bar=True)

        return study
