"""
The FeatureSelector is intended to be used in a stepwise manner, starting with data normalization, followed by the
removal of low-variance features, handling of multicollinearity, and the elimination of highly correlated features.
It supports various strategies for feature selection, allowing users to customize the preprocessing pipeline according
to their QSAR modeling needs.

The class also provides methods for visualizing data clusters and determining the optimal number of clusters for
analysis, enhancing the interpretability and analysis of QSAR datasets. This tool is essential for researchers and
scientists working in the field of QSAR modeling, offering a systematic approach to feature selection and dataset
preparation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.display_functions import display
from matplotlib import patches
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn import feature_selection, preprocessing
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


class FeatureSelector:
    """
    Implements methods for feature selection within QSAR modeling.

    The recommended order of use is normalization followed by feature selection.

    :param df: DataFrame with continuous data describing the observations and features.
    :type df: pd.DataFrame
    :param y: Name of the column for the dependent variable. Defaults to 'Log_MP_RATIO'.
    :type y: str, optional
    :param cols_to_ignore: List of column names to be ignored during processing.
    :type cols_to_ignore: list, optional
    """

    def __init__(self, df: pd.DataFrame, y: str = "Log_MP_RATIO", cols_to_ignore=None):
        """
        Initializes the FeatureSelector class.

        :param df: DataFrame with only continuous data describing the observations and features.
        :type df: pd.DataFrame
        :param y: Name of the column of the dependent variable. Defaults to 'Log_MP_RATIO'.
        :type y: str, optional
        :param cols_to_ignore: List of column names to ignore. Defaults to an empty list.
        :type cols_to_ignore: list, optional
        """
        if cols_to_ignore is None:
            cols_to_ignore = []
        self.df: pd.DataFrame = df
        self.y: str = y
        self.cols_to_ignore: list = cols_to_ignore

    def scale_data(
        self: "FeatureSelector",
        y: str = "Log_MP_RATIO",
        verbose: bool = False,
        inplace=False,
    ) -> pd.DataFrame:
        # It seems data should be normalized since not every data has a gaussian distribution
        """
        Normalizes the data within the DataFrame.

        :param y: The dependent variable (ignored in feature scaling). Defaults to 'Log_MP_RATIO'.
        :type y: str, optional
        :param verbose: If True, displays text to help visualize changes. Defaults to False.
        :type verbose: bool, optional
        :param inplace: If True, replaces the internal DataFrame with the normalized one. Defaults to False.
        :type inplace: bool, optional
        :return: The normalized DataFrame.
        :rtype: pd.DataFrame
        """
        if verbose:
            print("===== Before normalization ===== ")
            print(self.df.describe())

        df_to_normalize: pd.DataFrame = self.df.drop(columns=[y])

        df_normalized = preprocessing.normalize(df_to_normalize, axis=0)

        df_normalized = pd.concat(
            [
                pd.DataFrame(df_normalized, columns=self.df.columns.drop(y)),
                self.df["Log_MP_RATIO"],
            ],
            axis=1,
        )
        if verbose:
            print("===== DESCRIPTION =====")
            print(df_normalized.describe())
        if inplace:
            self.df = df_normalized
        return df_normalized

    def remove_low_variance(
        self: "FeatureSelector",
        y: str = "",
        variance_threshold: float = 0,
        cols_to_ignore=None,
        verbose: bool = False,
        inplace: bool = False,
    ) -> tuple[pd.DataFrame, list]:
        """
        Removes features from the DataFrame with variance below the specified threshold.

        :param df: DataFrame with continuous data describing observations and features.
        :type df: pd.DataFrame
        :param y: The dependent variable which will be ignored in the feature removal process.
                 Defaults to an empty string.
        :type y: str, optional
        :param variance_threshold: The threshold for variance below which features will be removed.
        :type variance_threshold: float
        :param cols_to_ignore: List of columns to be ignored during processing. Defaults to an empty list.
        :type cols_to_ignore: list, optional
        :param verbose: If True, displays descriptive text to help visualize changes. Defaults to False.
        :type verbose: bool, optional
        :param inplace: If True, updates the attribute `df` of the FeatureSelector object with the resultant DataFrame.
                       Defaults to False.
        :type inplace: bool, optional
        :return: A tuple containing the DataFrame with low variance features removed and a list of the removed column
                names.
        :rtype: Tuple[pd.DataFrame, list]
        """
        if cols_to_ignore is None:
            cols_to_ignore = []
        if not y:
            y = self.y

        if not cols_to_ignore:
            cols_to_ignore = self.cols_to_ignore.copy()

        if verbose:
            print("===== INITIAL SHAPE =====")
            print(self.df.shape)

        cols_to_ignore.append(y)
        df_clone: pd.DataFrame = self.df.copy()
        df_clone = df_clone.drop(columns=cols_to_ignore, axis=1)

        # Computes the mean of the variance of each column and deduces the
        # value to delete that will be below the percentage given by the user
        computed_treshold: int = df_clone.var(axis=1).mean() * variance_threshold
        # computed_treshold = variance_threshold

        vt: feature_selection.VarianceThreshold = feature_selection.VarianceThreshold(
            threshold=computed_treshold
        )
        high_variance = vt.fit_transform(df_clone)
        if verbose:
            print("===== CLEANED SHAPE =====")
            print(high_variance.shape)

        deleted_features: list = [
            column
            for column in df_clone
            if column not in df_clone.columns[vt.get_support()]
        ]
        if verbose:
            print("===== DELETED FEATURES ======")
            print(deleted_features)

        cleaned_df: pd.DataFrame = df_clone[
            df_clone.columns[vt.get_support(indices=True)]
        ].copy()

        cleaned_df[cols_to_ignore] = self.df[cols_to_ignore]
        if verbose:
            print("===== DF CLONE FINAL =====")
            print(cleaned_df.shape)

        if inplace:
            self.df = cleaned_df.copy()

        return cleaned_df, deleted_features

    def get_correlation_to_y(
        self: "FeatureSelector",
        df: pd.DataFrame = None,
        y: str = "",
        cols_to_ignore=None,
        method: str = "kendall",
    ) -> pd.DataFrame:
        """
        Calculates a correlation score for each feature in relation to the specified dependent variable.

        :param df: DataFrame with continuous data describing observations and features.
        :type df: pd.DataFrame.
        :param y: The dependent variable to compare for correlation. Defaults to an empty string.
        :type y: str, optional
        :param cols_to_ignore: List of columns where the function should not be executed. Defaults to an empty list.
        :type cols_to_ignore: list, optional
        :param method: Method used to calculate correlation. Options include "pearson", "kendall", or "spearman".
                      Defaults to "kendall".
        :type method: str, optional
        :return: A Series object with the correlation score for each feature relative to the y variable.
        :rtype: pd.Series
        """
        if cols_to_ignore is None:
            cols_to_ignore = []
        if df is None:
            df = self.df.copy()

        if not y:
            y = self.y

        if not cols_to_ignore:
            cols_to_ignore = self.cols_to_ignore.copy()

        df_corr_y = df.drop(columns=cols_to_ignore)

        df_corr_y = df_corr_y.corrwith(df_corr_y[y], method=method)
        return pd.DataFrame(df_corr_y).T

    def get_correlation(
        self,
        df: pd.DataFrame = None,
        y: str = "",
        cols_to_ignore=None,
        method: str = "kendall",
    ) -> pd.DataFrame:
        # https://datascience.stackexchange.com/a/64261
        """
        Calculates a correlation score for all features within a DataFrame.

        :param df: DataFrame with only continuous data describing the observations and features.
                  If None, uses the internal DataFrame. Defaults to None.
        :type df: pd.DataFrame, optional.
        :param y: Name of the column for the dependent variable to ignore in the collinearity test.
                 Defaults to an empty string, indicating no dependent variable.
        :type y: str, optional.
        :param cols_to_ignore: List of column names to ignore during correlation computation. Defaults to an empty list.
        :type cols_to_ignore: list, optional.
        :param method: Method used to calculate the correlation between the features.
                      Options are "pearson", "kendall", or "spearman". Defaults to "kendall".
        :type method: str, optional.
        :return: A DataFrame containing the correlation coefficients of the features.
        :rtype: pd.DataFrame.
        """
        if cols_to_ignore is None:
            cols_to_ignore = []
        if df is None:
            df = self.df.copy()

        if not y:
            y = self.y

        if not cols_to_ignore:
            cols_to_ignore = self.cols_to_ignore.copy()

        cols_to_ignore.append(y)

        df_corr: pd.DataFrame = df.drop(columns=cols_to_ignore)
        df_corr = df_corr.corr(method)
        return df_corr.dropna(how="all", axis=1).dropna(how="all")

    def remove_highly_correlated(
        self: "FeatureSelector",
        df_correlation: pd.DataFrame = None,
        df_corr_y: pd.DataFrame = None,
        df: pd.DataFrame = None,
        threshold: float = 0.9,
        verbose: bool = False,
        inplace=False,
        graph: bool = False,
    ) -> pd.DataFrame:
        # used: https://stackoverflow.com/a/61938339
        """
        Removes all the features from the DataFrame that have a correlation coefficient above the specified threshold.

        :param df_correlation: A DataFrame representing the correlation matrix.
                              Defaults to None, which will use the internal DataFrame's correlation matrix.
        :type df_correlation: pd.DataFrame, optional
        :param df_corr_y: A DataFrame of correlation values correlating all features to the dependent variable.
                         Defaults to None, which will calculate it from the internal DataFrame.
        :type df_corr_y: pd.DataFrame, optional
        :param df: A DataFrame with only continuous data describing the observations and features.
                  Defaults to None, which will use the internal DataFrame.
        :type df: pd.DataFrame, optional
        :param threshold: The threshold where features correlated beyond should be dropped. Defaults to 0.9.
        :type threshold: float, optional
        :param verbose: If True, displays additional information during processing. Defaults to False.
        :type verbose: bool, optional
        :param inplace: If True, replaces the internal DataFrame with the result. Defaults to False.
        :type inplace: bool, optional
        :param graph: If True, draws a heatmap of all dropped features and their respective correlation to each other.
                     Defaults to False.
        :type graph: bool, optional
        :return: A DataFrame with highly correlated features removed.
        :rtype: pd.DataFrame
        """
        if df is None:
            df = self.df.copy()

        # Get the correlation matrix
        if df_correlation is None:
            df_correlation = self.get_correlation()

        if verbose:
            print("===== CORRELATION MATRIX OF ALL FEATURES =====")
            print(df_correlation.shape)

        if df_corr_y is None:
            df_corr_y = self.get_correlation_to_y()

        if verbose:
            print("===== CORRELATION MATRIX TO Y =====")
            print(df_corr_y.shape)

        iters: range = range(len(df_correlation.columns) - 1)
        drop_cols: list = []  # cols to be dropped in the real dataframe

        for i in iters:  # cols
            for j in range(i + 1):  # rows
                # i + 1: jumps the first col
                # j: limits itself to i (if i = 0 then lim(j) = 0)
                # it basically is a triangle growing to the right where for each each x col it does x - 1 rows
                #   A   B   C   D
                # A     ✓   ✓   ✓
                # B         ✓   ✓
                # C             ✓
                # D
                item = df_correlation.iloc[j : (j + 1), (i + 1) : (i + 2)]
                col: str = item.columns
                row: int = item.index
                value: int = abs(item.values)

                # if correlation exceeds the threshold
                if value >= threshold:
                    if verbose:
                        print(
                            "col: ",
                            col.values[0],
                            " | ",
                            "row: ",
                            row.values[0],
                            " = ",
                            round(value[0][0], 2),
                        )
                    # Check which feature is more correlated to y in the table
                    col_feature: str = abs(df_corr_y[col.values[0]])
                    row_feature: str = abs(df_corr_y[row.values[0]])

                    feature_to_drop: str = ""

                    # Select the feature to drop, the lowest correlation to y is selected
                    feature_to_drop = (
                        col.values[0]
                        if (col_feature < row_feature).all()
                        else row.values[0]
                    )
                    if feature_to_drop not in drop_cols:
                        drop_cols.append(feature_to_drop)
                        if verbose:
                            print("DROPPED: ", feature_to_drop)
                    else:
                        if verbose:
                            print("ALREADY DROPPED: ", feature_to_drop)

        drops: set = set(drop_cols)
        if verbose:
            print("num of cols to drop: ", len(drops))
        dropped_df: pd.DataFrame = df.drop(columns=drops).copy()

        if graph:
            removed_feat = df.columns.difference(dropped_df.columns)
            df_display = 1 - df_correlation.loc[removed_feat, removed_feat]
            linkage = hierarchy.linkage(
                distance.squareform(df_display), method="average"
            )
            print(linkage)
            g = sns.clustermap(df_display, row_linkage=linkage, col_linkage=linkage)

            mask = np.tril(np.ones_like(df_display))
            values = g.ax_heatmap.collections[0].get_array().reshape(df_display.shape)
            new_values = np.ma.array(values, mask=mask)
            g.ax_heatmap.collections[0].set_array(new_values)
            display(g)

        if inplace:
            self.df = dropped_df.copy()

        return dropped_df

    # WARNING: This function only works if the number of observations is higher or close to the number of features
    # https://stats.stackexchange.com/a/583502
    def remove_multicollinearity(
        self: "FeatureSelector", df: pd.DataFrame = None, y: str = "Log_MP_RATIO"
    ):
        """
        Removes multicollinearity from the DataFrame.

        :param df: DataFrame with continuous data describing the observations and features. Defaults to None.
        :type df: pd.DataFrame, optional
        :param y: Name of the column for the dependent variable. Defaults to 'Log_MP_RATIO'.
        :type y: str, optional
        """
        if df is None:
            df = self.df.copy()

        df = df.loc[:, df.columns != y]

        # This is needed to do correctly the VIF
        df = add_constant(df)

        df_vif: pd.DataFrame = pd.DataFrame()
        df_vif["VIF_factor"] = [
            variance_inflation_factor(df.values, i) for i in range(df.shape[1])
        ]
        df_vif["features"] = df.columns

        print(df_vif)

    def transform(self: "FeatureSelector") -> pd.DataFrame:
        """
        Removes low variance and highly correlated features from the DataFrame stored in the FeatureSelector object.

        :return: A DataFrame with low variance and highly correlated features removed.
        :rtype: pd.DataFrame
        """
        self.remove_low_variance(inplace=True)
        self.remove_highly_correlated(inplace=True)
        return self.df


def display_data_cluster(
    df_corr: pd.DataFrame,
    n_clusters: int = 8,
) -> None:
    # https://www.kaggle.com/code/ignacioalorre/clustering-features-based-on-correlation-and-tags/notebook
    """
    Displays the correlated features in a clusterized heatmap graph.

    :param df_corr: Correlation DataFrame to be clustered.
    :type df_corr: pd.DataFrame
    :param n_clusters: Number of clusters to form. Defaults to 8.
    :type n_clusters: int, optional
    :param n_init: Number of time the k-means algorithm will run with different centroid seeds. Defaults to 500.
    :type n_init: int, optional
    :param max_iter: Maximum number of iterations of the k-means algorithm for a single run. Defaults to 1000.
    :type max_iter: int, optional
    """
    feat_names = df_corr.columns
    corr_feat_mtx: np.ndarray = df_corr.to_numpy()

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=1000,
        n_init=500,
        random_state=0,
    )
    corr_feat_labels = kmeans.fit_predict(corr_feat_mtx)

    print(len(corr_feat_labels))

    # Preparing a dataframe to collect some cluster stats
    # Contains the clusters and what features they group together
    corr_feat_clust_df = pd.DataFrame(np.c_[feat_names, corr_feat_labels])
    corr_feat_clust_df.columns = ["feature", "cluster"]
    corr_feat_clust_df["feat_list"] = corr_feat_clust_df.groupby(["cluster"]).transform(
        lambda x: ", ".join(x)
    )
    corr_feat_clust_df = (
        corr_feat_clust_df.groupby(["cluster", "feat_list"])
        .size()
        .reset_index(name="feat_count")
    )

    # Transforming our data with the KMean model
    # Contains the feature their distance inside the cluster and their distance normalized
    corr_node_dist = kmeans.transform(df_corr)
    corr_clust_dist = np.c_[
        feat_names,
        np.round(corr_node_dist.min(axis=1), 3),
        np.round(corr_node_dist.min(axis=1) / np.max(corr_node_dist.min(axis=1)), 3),
        corr_feat_labels,
    ]
    corr_clust_dist_df = pd.DataFrame(corr_clust_dist)
    corr_clust_dist_df.columns = [
        "feature",
        "dist_corr",
        "dist_corr_norm",
        "cluster_corr",
    ]

    # Method to group together in correlation matrix features with same labels
    def clustering_corr_matrix(corr_matrix: pd.DataFrame, clustered_features: list):
        """
        Groups together features with the same labels in the correlation matrix based on the clustered features.

        :param corr_matrix: Correlation matrix to be clustered based on labels.
        :type corr_matrix: pd.DataFrame
        :param clustered_features: List of clustered features to group together.
        :type clustered_features: list
        :return: A numpy array of the clustered correlation matrix based on labels.
        :rtype: np.ndarray
        """
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
    def processing_clustered_corr_matrix(
        feat_labels: np.ndarray, corr_matrix: pd.DataFrame
    ):
        """
        Processes the correlation matrix based on the clustered features.

        :param feat_labels: Array of feature labels to be clustered.
        :type feat_labels: np.ndarray
        :param corr_matrix: Correlation matrix to be clustered.
        :type corr_matrix: pd.DataFrame
        :return: A numpy array of the clustered correlation matrix.
        :rtype: np.ndarray
        """
        lst_lab = list(feat_labels)
        # lst_feat = corr_matrix.columns

        lab_feat_map = {i: lst_lab[i] for i in range(len(lst_lab))}
        lab_feat_map_sorted = dict(
            sorted(lab_feat_map.items(), key=lambda item: item[1])
        )

        clustered_features = list(map(int, lab_feat_map_sorted.keys()))
        print(len(clustered_features))
        return clustering_corr_matrix(corr_matrix, clustered_features)

    def plot_clustered_matrix(
        clust_mtx: np.ndarray, feat_clust_list: np.ndarray
    ) -> None:
        """
        Plots the clustered matrix based on the correlation matrix.

        :param clust_mtx: Clustered matrix based on the correlation matrix.
        :type clust_mtx: np.ndarray
        :param feat_clust_list: List of clustered features.
        :type feat_clust_list: np.ndarray
        """
        plt.figure()

        fig, ax = plt.subplots(1)
        im = ax.imshow(clust_mtx, interpolation="nearest")

        corner: int = 0
        for s in feat_clust_list:
            rect = patches.Rectangle(
                (float(corner), float(corner)),
                float(s),
                float(s),
                angle=0.0,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            corner += s
            ax.add_patch(rect)

        fig.colorbar(im)
        plt.title("Clustered feature by correlation")
        plt.show()

    clust_mtx = processing_clustered_corr_matrix(corr_feat_labels, df_corr)
    plot_clustered_matrix(clust_mtx, corr_feat_clust_df["feat_count"].to_numpy())


def display_elbow(df: pd.DataFrame, max_num_clusters: int = 15) -> None:
    """
    Displays the elbow curve for the given dataframe and its associated Within-Cluster Sum of Square (WCSS).

    :param df: A correlation dataframe to determine the optimal number of clusters for k-means clustering.
    :type df: pd.DataFrame
    :param max_num_clusters: The maximum number of clusters to evaluate for the elbow curve. Defaults to 15.
    :type max_num_clusters: int, optional
    """
    corr_feat_mtx: np.ndarray = df.to_numpy()

    wcss: list = []

    for i in range(1, max_num_clusters):
        kmeans = KMeans(
            n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0
        )
        kmeans.fit(corr_feat_mtx)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, max_num_clusters), wcss)
    plt.title("Elbow method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()
