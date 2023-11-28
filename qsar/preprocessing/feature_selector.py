"""
Feature selector class
"""

import pandas as pd
import numpy as np
import seaborn as sns

from scipy.spatial import distance
from scipy.cluster import hierarchy

from sklearn import preprocessing, feature_selection

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from tensorboard.notebook import display


class FeatureSelector:
    """
    A class made implementing methods for feature selection
    The recommended order of use is: Normalization --> Feature selection

    :param df: A datafram with only continuous data describing the observations and features
    :type df: pd.Dataframe
    """

    def __init__(self, df: pd.DataFrame, target: str = "Log_MP_RATIO", cols_to_ignore=None):
        """
        Init function of the FeatureSelector class

        :param df: A dataframe with only continuous data describing the observations and features
        :type df: pd.Dataframe
        :param target: A string that corresponds to the name of the column of the dependent variable
        :type target: str
        :param cols_to_ignore: A list of string that corresponds to columns name to ignore
        :type: list
        """
        if cols_to_ignore is None:
            cols_to_ignore = []
        self.df: pd.DataFrame = df
        self.y: str = target
        self.cols_to_ignore: list = cols_to_ignore

    def scale_data(self: "FeatureSelector", y: str = "Log_MP_RATIO", verbose: bool = False,
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

        df_normalized = pd.concat(
            [pd.DataFrame(df_normalized, columns=self.df.columns.drop(y)), self.df["Log_MP_RATIO"]], axis=1)
        if verbose:
            print("===== DESCRIPTION =====")
            print(df_normalized.describe())
        if inplace:
            self.df = df_normalized
        return df_normalized

    # TODO: check avec Nadia "0.01 would mean dropping the column where 99% of the values are similar."
    def remove_low_variance(self: "FeatureSelector", y: str = "", variance_threshold: float = 0,
                            cols_to_ignore=None, verbose: bool = False,
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

        Args:
            cols_to_ignore:
            cols_to_ignore:
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
                             cols_to_ignore=None, method: str = "kendall") -> pd.DataFrame:
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

    def get_correlation(self, df: pd.DataFrame = None, y: str = "",
                        cols_to_ignore=None, method: str = "kendall") -> pd.DataFrame:
        # https://datascience.stackexchange.com/a/64261
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

    def remove_highly_correlated(self: "FeatureSelector", df_correlation: pd.DataFrame = None,
                                 df_corr_y: pd.DataFrame = None, df: pd.DataFrame = None,
                                 threshold: float = 0.9, verbose: bool = False,
                                 inplace=False, graph: bool = False) -> pd.DataFrame:
        # used: https://stackoverflow.com/a/61938339
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
                item = df_correlation.iloc[j:(j + 1), (i + 1):(i + 2)]
                col: str = item.columns
                row: int = item.index
                value: int = abs(item.values)

                # if correlation exceeds the threshold
                if value >= threshold:
                    # TODO: change this to drop either by either the lowest variance or the more correlated to y
                    if verbose:
                        print("col: ", col.values[0], " | ", "row: ", row.values[0], " = ", round(value[0][0], 2))
                    # Check which feature is more correlated to y in the table
                    col_feature: str = abs(df_corr_y[col.values[0]])
                    row_feature: str = abs(df_corr_y[row.values[0]])

                    feature_to_drop: str = ""

                    # Select the feature to drop, the lowest correlation to y is selected
                    feature_to_drop = col.values[0] if (col_feature < row_feature).all() else row.values[0]
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
