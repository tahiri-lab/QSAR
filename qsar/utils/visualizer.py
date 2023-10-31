import math

import pandas as pd
from matplotlib import pyplot as plt

from qsar.models.model import Model


class Visualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_folds(self, df: pd.DataFrame, y: str, n_folds: int):

        fig, axs = plt.subplots(
            1, n_folds, sharex=True, sharey=True, figsize=(10, 4)
        )
        for i, ax in enumerate(axs):
            ax.hist(df[df.Fold == i][y], bins=10, density=True, label=f"Fold-{i}")
            if i == 0:
                ax.set_ylabel("Frequency")
            if i == math.ceil(n_folds / 2):
                ax.set_xlabel(y)
            ax.legend(frameon=False, handlelength=0)
        plt.tight_layout()
        plt.show()

    def display_score(self, model, R2, CV, custom_cv, Q2):
        print(
            "===== ",
            type(model).__name__,
            " =====",
            "\n\tR2\t\t\t:\t",
            R2,
            "\n\tCV\t\t\t:\t",
            CV,
            "\n\tCustom CV\t:\t",
            custom_cv,
            "\n\tQ2\t\t\t:\t",
            Q2,
        )

    def display_graph(self, model: Model, y_train: pd.DataFrame, y_test: pd.DataFrame, y_pred_train: pd.DataFrame,
                      y_pred_test: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_train, y_pred_train, c="blue", label="Train", alpha=0.7)
        ax.scatter(y_test, y_pred_test, c="orange", label="Test", alpha=0.7)
        ax.plot(
            [min(y_train) - 1, max(y_train) + 1],
            [min(y_train) - 1, max(y_train) + 1],
            c="black",
        )
        plt.xlim((min(y_train) - 2, max(y_train) + 2))
        plt.ylim((min(y_train) - 2, max(y_train) + 2))
        plt.title(type(model).__name__)
        plt.legend(loc="upper right")
        ax.set_ylabel("True target", fontsize=14)
        ax.set_xlabel("Predicted target", fontsize=14)
        plt.show()
