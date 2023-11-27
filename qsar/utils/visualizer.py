import math
from typing import Tuple, List

import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem

from rdkit.Chem import Draw


class Visualizer:
    """
    A class to visualize various aspects of QSAR models.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the Visualizer with optional figure size.

        :param figsize: Tuple indicating figure width and height.
        :type figsize: Tuple[int, int]
        """
        self.figsize = figsize

    def display_cv_folds(self, df: pd.DataFrame, y: str, n_folds: int):
        """
        Plot the distribution of data across different cross-validation folds.

        :param df: The DataFrame containing the data.
        :type df: pd.DataFrame
        :param y: The target column name.
        :type y: str
        :param n_folds: Number of folds.
        :type n_folds: int
        """
        fig, axs = plt.subplots(
            1, n_folds, sharex=True, sharey=True, figsize=self.figsize
        )
        for i, ax in enumerate(axs):
            ax.hist(df[df.Fold == i][y], bins=10, density=True)
            ax.set_title(f"Fold-{i}")
            if i == 0:
                ax.set_ylabel("Frequency")
            if i == math.ceil(n_folds / 2):
                ax.set_xlabel(y)
        plt.tight_layout()
        plt.show()

    def display_model_performance(
            self, model_name: str, R2: float, CV: float, custom_cv: float, Q2: float
    ):
        """
        Display the scores of the model in a table format.

        :param model_name: The name of the model to be evaluated.
        :type model_name: str
        :param R2: R squared score.
        :type R2: float
        :param CV: Cross-validation score.
        :type CV: float
        :param custom_cv: Custom cross-validation score.
        :type custom_cv: float
        :param Q2: Q squared score.
        :type Q2: float
        """
        # Create a new figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Data for the table
        columns = ["Metric", "Score"]
        rows = ["R2", "CV", "Custom CV", "Q2"]
        data = [[rows[i], score] for i, score in enumerate([R2, CV, custom_cv, Q2])]

        # Remove axes
        ax.axis("tight")
        ax.axis("off")

        # Create a table and set its title
        table = ax.table(
            cellText=data, colLabels=columns, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(columns))))
        table.scale(1, 1.5)
        ax.set_title(f"Scores for {model_name}", fontsize=16)

        plt.show()

    def display_graph(
            self,
            model_name: str,
            y_train: pd.DataFrame,
            y_test: pd.DataFrame,
            y_pred_train: pd.DataFrame,
            y_pred_test: pd.DataFrame,
    ):
        """
        Display a scatter plot of true vs. predicted values for training and test sets.

        :param model_name: The name of the model used for prediction.
        :type model_name: str
        :param y_train: True values for the training set.
        :type y_train: pd.DataFrame
        :param y_test: True values for the test set.
        :type y_test: pd.DataFrame
        :param y_pred_train: Predicted values for the training set.
        :type y_pred_train: pd.DataFrame
        :param y_pred_test: Predicted values for the test set.
        :type y_pred_test: pd.DataFrame
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(y_train, y_pred_train, c="blue", label="Train", alpha=0.7)
        ax.scatter(y_test, y_pred_test, c="orange", label="Test", alpha=0.7)
        y_train_min = float(y_train.min().iloc[0])
        y_train_max = float(y_train.max().iloc[0])

        ax.plot(
            [y_train_min - 1, y_train_max + 1],
            [y_train_min - 1, y_train_max + 1],
            c="black",
        )
        ax.set_xlim(y_train_min - 2, y_train_max + 2)
        ax.set_ylim(y_train_min - 2, y_train_max + 2)
        ax.set_title(model_name)
        ax.set_ylabel("True target", fontsize=14)
        ax.set_xlabel("Predicted target", fontsize=14)
        plt.legend(loc="upper right")
        plt.show()

    @staticmethod
    def display_atom_count_distribution(atom_counts):
        """
        Plot the distribution of atom counts in a dataset.

        :param atom_counts: List or array of atom counts.
        :type atom_counts: List[int] or similar
        """
        plt.hist(atom_counts, bins=30)
        plt.xlabel('Number of Atoms')
        plt.ylabel('Frequency')
        plt.title('Distribution of Atom Counts in Dataset')
        plt.show()

    @staticmethod
    def draw_generated_molecules(molecules: List[Chem.Mol]):
        """
        Draw the generated molecules.

        :param molecules: List of molecules to be visualized.
        :type molecules: List[Chem.Mol]
        """
        img = Draw.MolsToGridImage(molecules[0:100], molsPerRow=5, subImgSize=(250, 250), maxMols=100, legends=None,
                                   returnPNG=False)
        img.show()
