import math
from typing import Tuple

import pandas as pd
from matplotlib import pyplot as plt

from qsar.models.model import Model


class Visualizer:
    """
    A class to visualize various aspects of QSAR models.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the Visualizer with optional figure size.

        Parameters:
        - figsize (tuple): Tuple indicating figure width and height.
        """
        self.figsize = figsize

    def plot_folds(self, df: pd.DataFrame, y: str, n_folds: int):
        """
        Plot the distribution of data across different folds.

        Parameters:
        - df (pd.DataFrame): The dataframe containing the data.
        - y (str): The target column name.
        - n_folds (int): Number of folds.
        """
        fig, axs = plt.subplots(1, n_folds, sharex=True, sharey=True, figsize=self.figsize)
        for i, ax in enumerate(axs):
            ax.hist(df[df.Fold == i][y], bins=10, density=True)
            ax.set_title(f"Fold-{i}")
            if i == 0:
                ax.set_ylabel("Frequency")
            if i == math.ceil(n_folds / 2):
                ax.set_xlabel(y)
        plt.tight_layout()
        plt.show()

    def display_score(self, model: Model, R2: float, CV: float, custom_cv: float, Q2: float):
        """
        Display the score of the model in a table format.

        Parameters:
        - model (Model): The model to be evaluated.
        - R2 (float): R squared score.
        - CV (float): Cross-validation score.
        - custom_cv (float): Custom cross-validation score.
        - Q2 (float): Q squared score.
        """
        # Create a new figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Data for the table
        columns = ["Metric", "Score"]
        rows = ["R2", "CV", "Custom CV", "Q2"]
        data = [[rows[i], score] for i, score in enumerate([R2, CV, custom_cv, Q2])]

        # Remove axes
        ax.axis('tight')
        ax.axis('off')

        # Create a table and set its title
        table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(columns))))
        table.scale(1, 1.5)
        ax.set_title(f"Scores for {type(model).__name__}", fontsize=16)

        plt.show()

    def display_graph(self, model: Model, y_train: pd.DataFrame, y_test: pd.DataFrame,
                      y_pred_train: pd.DataFrame, y_pred_test: pd.DataFrame):
        """
        Display a scatter plot of true vs. predicted values for training and test sets.

        Parameters:
        - model (Model): The model used for prediction.
        - y_train (pd.DataFrame): True values for the training set.
        - y_test (pd.DataFrame): True values for the test set.
        - y_pred_train (pd.DataFrame): Predicted values for the training set.
        - y_pred_test (pd.DataFrame): Predicted values for the test set.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(y_train, y_pred_train, c="blue", label="Train", alpha=0.7)
        ax.scatter(y_test, y_pred_test, c="orange", label="Test", alpha=0.7)

        ax.plot(
            [min(y_train) - 1, max(y_train) + 1],
            [min(y_train) - 1, max(y_train) + 1],
            c="black",
        )
        ax.set_xlim((min(y_train) - 2, max(y_train) + 2))
        ax.set_ylim((min(y_train) - 2, max(y_train) + 2))
        ax.set_title(type(model).__name__)
        ax.set_ylabel("True target", fontsize=14)
        ax.set_xlabel("Predicted target", fontsize=14)
        plt.legend(loc="upper right")
        plt.show()

