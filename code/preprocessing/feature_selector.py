"""
Feature selector class
"""


import pandas as pd
import numpy as np
import seaborn as sns

from scipy.spatial import distance
from scipy.cluster import hierarchy

from sklearn import preprocessing, feature_selection
from sklearn.cluster import KMeans

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from matplotlib import patches
import matplotlib.pyplot as plt


class FeatureSelector:
    """
    A class made implementing methods for feature selection
    The recommended order of use is: Normalization --> Feature selection

    :param df: A datafram with only continuous data describing the observations and features
    :type df: pd.Dataframe
    """
    def __init__(self, df: pd.DataFrame, y: str = "Log_MP_RATIO", cols_to_ignore: list = []):
        """
        Init function of the FeatureSelector class

        :param df: A dataframe with only continuous data describing the observations and features
        :type df: pd.Dataframe
        :param y: A string that corresponds to the name of the column of the dependent variable
        :type y: str
        :param cols_to_ignore: A list of string that corresponds to columns name to ignore
        :type: list
        """
        self.df: pd.DataFrame = df
        self.y: str = y
        self.cols_to_ignore: list = cols_to_ignore

        
    def scale_data(self: "FeatureSelector", y: str = "Log_MP_RATIO", verbose: bool=False, 
                   inplace=False) -> pd.DataFrame:
        # It seems data should be normalized since not every data has a gaussian distribution
        """
        Normalize the data of in the CSV
        Even though on linear regression it isn't necessarily needed but it can help for data interpretation

        Parameters
        ----------
        y: str
            The depedent variable (it will be ignored in the scaling of features)
        verbose (default=False): bool
            If set to True displays some text to help visualize the changes
        inplace (default=False): bool
            If set to True replaces the attribute df of the FeatureSelector object with the normalized dataframe

        Returns
        ----------
        pd.DataFrame
            The same dataframe as input but with normalized values
        """
        if verbose:
            print("===== Before normalization ===== ")
            print(self.df.describe())

        df_to_normalize: pd.DataFrame = self.df.drop(columns=[y])
        
        df_normalized = preprocessing.normalize(df_to_normalize, axis=0)

        df_normalized = pd.concat([pd.DataFrame(df_normalized, columns=self.df.columns.drop(y)), self.df["Log_MP_RATIO"]], axis=1)
        if verbose:
            print("===== DESCRIPTION =====")
            print(df_normalized.describe())
        if inplace:
            self.df = df_normalized
        return df_normalized


    # TODO: check avec Nadia "0.01 would mean dropping the column where 99% of the values are similar."
    def remove_low_variance(self: "FeatureSelector", y: str = "", variance_threshold: float = 0,
                            cols_to_ignore: list = [], verbose: bool = False, 
                            inplace: bool = False) -> tuple[pd.DataFrame, list]:
        """
        Remove features with a variance level below the threshold

        Parameters
        ----------
        df: pd.DataFrame
            A dataframe with only continuous data describing the observations and features
        y (default = ""): str
            The depedent variable (it will be ignored in the removal of features)
        variance_threshold: int
            The threshold at which features below should deleted
        cols_to_ignore (default = []): list
            Cols where the function should not be executed
        verbose (default=False): bool
            If set to True displays some text to help visualize the changes
        inplace (default=False): bool
            If set to True replaces the attribute df of the FeatureSelector object with the removed
            low variance feature dataframe

        Returns
        ----------
        pd.Dataframe
            The same dataframe as input but the features with low variance are removed
        list
            A list of all the deleted columns
        """
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
        #computed_treshold = variance_threshold   

        Vt: feature_selection.VarianceThreshold = feature_selection.VarianceThreshold(threshold=computed_treshold)
        high_variance = Vt.fit_transform(df_clone)
        if verbose:
            print("===== CLEANED SHAPE =====")
            print(high_variance.shape)

        deleted_features: list = [column for column in df_clone if column not in df_clone.columns[Vt.get_support()]]
        if verbose:
            print("===== DELETED FEATURES ======")
            print(deleted_features)

        cleaned_df: pd.DataFrame = df_clone[df_clone.columns[Vt.get_support(indices=True)]].copy()

        cleaned_df[cols_to_ignore] = self.df[cols_to_ignore]
        if verbose:
            print("===== DF CLONE FINAL =====")
            print(cleaned_df.shape)

        if inplace:
            self.df = cleaned_df.copy()

        return cleaned_df, deleted_features
    
    def get_correlation_to_y(self: "FeatureSelector", df: pd.DataFrame = None, y: str = "",
                             cols_to_ignore: list = [], method: str = "kendall") -> pd.DataFrame:
        """
        Calculates a correlation score of all the features depending on the y variable

        Parameters
        ----------
        df: pd.DataFrame
            A dataframe with only continuous data describing the observations and features
        y (default = ""): str
            The dependent variable we want to compare for the correlation
        cols_to_ignore (default = []): list
            Cols where the function should not be executed
        method (default = "kendall")
            Method to use to calculate correlation

        Returns
        ----------
        pd.Series
            A Series object with the score of correlation for each feature on the y variable
        """
        if df is None:
            df = self.df.copy()

        if not y:
            y = self.y

        if not cols_to_ignore:
            cols_to_ignore = self.cols_to_ignore.copy()

        df_corr_y = df.drop(columns=cols_to_ignore)
        
        df_corr_y = df_corr_y.corrwith(df_corr_y[y], method=method)
        return df_corr_y

    def get_correlation(self, df: pd.DataFrame = None, y: str = "", 
                        cols_to_ignore: list = [], method: str = "kendall") -> pd.DataFrame:
        #https://datascience.stackexchange.com/a/64261
        """
        Calculates a correlation score of all the features

        Parameters
        ----------
        df (default = None): pd.DataFrame
            A dataframe with only continuous data describing the observations and features
        y (default = ''): str
            Dependent variable to ignore in the colinearity test
        cols_to_ignore (default = []): list
            Columns to ignore if needed
        method (default = "kendall"): str
            Method to use to calculate the correlation between the features ("pearson", "kendall" or "spearman")

        Returns
        ----------
        pd.DataFrame
            A dataframe with the score of correlation for each feature on the y variable
        """
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


    def remove_highly_correlated(self: "FeatureSelector", df_correlation: pd.DataFrame = None, 
                                 df_corr_y: pd.DataFrame = None, df: pd.DataFrame = None, 
                                 threshold: float = 0.9, verbose: bool = False, 
                                 inplace = False, graph: bool = False) -> pd.DataFrame:
        #used: https://stackoverflow.com/a/61938339
        """
        Removes all the features that have correlation above the threshold

        Parameters
        ----------
        df_correlation (default = None): pd.DataFrame
            A dataframe representing the correlation matrix 
        df_corr_y (default = None): pd.Dataframe
            A dataframe of correlation correlating all the features to the dependant variable
        df (default = None): pd.DataFrame
            A dataframe with only continuous data describing the observations and features
            If the default value is given it will use the dataframe contained in the FeatureSelector object
        threshold (default = 0.9): float
            The treshold where features correlated beyond should be dropped
        verbose (default = False): bool
            If set to True displays some informations
        inplace (default = False): bool
            If set to True replaces the attribute df of the FeatureSelector object with the normalized dataframe
        graph (default = False): bool
            If true draws a heatmap of all the dropped features and their respective correlation to each other
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
        drop_cols: list = [] #cols to be dropped in the real dataframe


        for i in iters: # cols
            for j in range(i+1): # rows
                # i + 1: jumps the first col
                # j: limits itself to i (if i = 0 then lim(j) = 0)
                # it basically is a triangle growing to the right where for each each x col it does x - 1 rows 
                #   A   B   C   D
                # A     ✓   ✓   ✓
                # B         ✓   ✓
                # C             ✓
                # D 
                item = df_correlation.iloc[j:(j+1), (i+1):(i+2)] 
                col: str = item.columns
                row: int = item.index
                value: int = abs(item.values)
                
                # if correlation exceeds the threshold
                if value >= threshold:
                    #TODO: change this to drop either by either the lowest variance or the more correlated to y
                    if verbose:
                        print("col: ", col.values[0], " | ", "row: ", row.values[0], " = ", round(value[0][0], 2))
                    # Check which feature is more correlated to y in the table
                    col_feature: str = abs(df_corr_y[col.values[0]])
                    row_feature: str = abs(df_corr_y[row.values[0]])
                    
                    feature_to_drop: str = ""

                    # Select the feature to drop, the lowest correlation to y is selected
                    feature_to_drop = col.values[0] if col_feature < row_feature else row.values[0]
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
            linkage = hierarchy.linkage(distance.squareform(df_display), method="average")
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
    def remove_multicollinearity(self: "FeatureSelector", df: pd.DataFrame = None, y: str = "Log_MP_RATIO"):
        if df is None:
            df = self.df.copy()

        df = df.loc[:, df.columns != y]
        
        # This is needed to do correctly the VIF
        df = add_constant(df)

        df_vif: pd.DataFrame = pd.DataFrame()
        df_vif["VIF_factor"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        df_vif["features"] = df.columns
        
        print(df_vif)


    # TODO: if time remains add all the options to fully custom this function
    def transform(self: "FeatureSelector") -> pd.DataFrame:
        """
        Removes low variance and highly correlated features

        Returns
        ----------
        pd.Datarame
        """
        self.remove_low_variance(inplace=True)
        self.remove_highly_correlated(inplace=True)
        return self.df


def display_data_cluster(df_corr: pd.DataFrame, n_clusters: int = 8, 
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


    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter = 1000, n_init=500, random_state = 0)
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
                            np.round(corr_node_dist.min(axis=1)/np.max(corr_node_dist.min(axis=1)), 3),
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
                npm_zero[n, m] = npm[i-1, j-1]
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


def display_elbow(df: pd.DataFrame, max_num_clusters: int = 15) -> None:
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
