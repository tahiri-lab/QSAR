import math
from typing import Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem

from rdkit.Chem import Draw
from PIL import Image
from sklearn.cluster import KMeans

from matplotlib import patches


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

    def display_true_vs_predicted(
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

    def display_elbow(self, df: pd.DataFrame, max_num_clusters: int = 15) -> None:
        """
        Displays the elbow curve for the given dataframe and its associated Within-Cluster Sum of Square

        Parameters
        ----------
        df: pd.DataFrame
            A correlation dataframe
        max_num_clusters (default = 15): int
            The maximum number of clusters wanted

        Returns
        ---------
        None
        """
        corr_feat_mtx: np.ndarray = df.to_numpy()

        wcss: list = []
        max_num_clusters = max_num_clusters

        for i in range(1, max_num_clusters):
            kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
            kmeans.fit(corr_feat_mtx)
            wcss.append(kmeans.inertia_)

        plt.plot(range(1, max_num_clusters), wcss)
        plt.title("Elbow method")
        plt.xlabel("Number of clusters")
        plt.ylabel("WCSS")
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
    def draw_generated_molecules(molecules: List[Chem.Mol]) -> Image:
        """
        Draw the generated molecules.

        :param molecules: List of molecules to be visualized.
        :type molecules: List[Chem.Mol]
        """
        img = Draw.MolsToGridImage(molecules[0:100], molsPerRow=5, subImgSize=(250, 250), maxMols=100, legends=None,
                                   returnPNG=False)
        img.show()
        return img

    def display_data_cluster(self, df_corr: pd.DataFrame, n_clusters: int = 8,
                             n_init: str = 500, max_iter: int = 1000) -> None:
        # https://www.kaggle.com/code/ignacioalorre/clustering-features-based-on-correlation-and-tags/notebook
        """
        Displays the correlated features in a clusterized graph

        Parameters
        ----------
        df_corr: pd.Dataframe
            A correlation dataframe
        n_clusters (default = 8): int
            The number of clusters kmean does
        n_init (default = 500): int
            number of time the KMeans algorithm is run with different centroid
        max_iter (default = 1000): int
            maximum number of iterations for a single run

        Returns
        ---------
        None
        """
        feat_names = df_corr.columns
        corr_feat_mtx: np.ndarray = df_corr.to_numpy()

        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=1000, n_init=500, random_state=0)
        corr_feat_labels = kmeans.fit_predict(corr_feat_mtx)

        print(len(corr_feat_labels))

        # Preparing a dataframe to collect some cluster stats
        # Contains the clusters and what features they group together
        corr_feat_clust_df = pd.DataFrame(np.c_[feat_names, corr_feat_labels])
        corr_feat_clust_df.columns = ["feature", "cluster"]
        corr_feat_clust_df["feat_list"] = corr_feat_clust_df.groupby(["cluster"]).transform(lambda x: ", ".join(x))
        corr_feat_clust_df = corr_feat_clust_df.groupby(["cluster", "feat_list"]).size().reset_index(name="feat_count")

        # Transforming our data with the KMean model
        # Contains the feature their distance inside the cluster and their distance normalized
        corr_node_dist = kmeans.transform(df_corr)
        corr_clust_dist = np.c_[feat_names, np.round(corr_node_dist.min(axis=1), 3),
        np.round(corr_node_dist.min(axis=1) / np.max(corr_node_dist.min(axis=1)), 3),
        corr_feat_labels]
        corr_clust_dist_df = pd.DataFrame(corr_clust_dist)
        corr_clust_dist_df.columns = ["feature", "dist_corr", "dist_corr_norm", "cluster_corr"]

        # Method to group together in correlation matrix features with same labels
        def clustering_corr_matrix(corr_matrix: pd.DataFrame, clustered_features: list):
            npm: np.ndarray = corr_matrix.to_numpy()
            # Creates an numpy array filled with zeros
            npm_zero: np.ndarray = np.zeros(shape=(len(npm), len(npm)))
            n: int = 0
            for i in clustered_features:
                m: int = 0
                for j in clustered_features:
                    npm_zero[n, m] = npm[i - 1, j - 1]
                    m += 1
                n += 1
            return npm_zero

        # Preprocessing the correlation matrix before starting the clustering based on labels
        def processing_clustered_corr_matrix(feat_labels: np.ndarray, corr_matrix: pd.DataFrame):
            lst_lab = list(feat_labels)
            lst_feat = corr_matrix.columns

            lab_feat_map = {i: lst_lab[i] for i in range(len(lst_lab))}
            lab_feat_map_sorted = {k: v for k, v in sorted(lab_feat_map.items(), key=lambda item: item[1])}

            clustered_features = list(map(int, lab_feat_map_sorted.keys()))
            print(len(clustered_features))
            return clustering_corr_matrix(corr_matrix, clustered_features)

        def plot_clustered_matrix(clust_mtx: np.ndarray, feat_clust_list: np.ndarray) -> None:
            plt.figure()

            fig, ax = plt.subplots(1)
            im = ax.imshow(clust_mtx, interpolation="nearest")

            corner: int = 0
            for s in feat_clust_list:
                rect = patches.Rectangle((float(corner), float(corner)), float(s), float(s), angle=0.0, linewidth=2,
                                         edgecolor='r', facecolor="none")
                ax.add_patch(rect)
                corner += s
                ax.add_patch(rect)

            fig.colorbar(im)
            plt.title("Clustered feature by correlation")
            plt.show()

        clust_mtx = processing_clustered_corr_matrix(corr_feat_labels, df_corr)
        plot_clustered_matrix(clust_mtx, corr_feat_clust_df["feat_count"].to_numpy())
