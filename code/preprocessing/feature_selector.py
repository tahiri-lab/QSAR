"""
TITLE
"""


import pandas as pd
import numpy as np

from sklearn import preprocessing, feature_selection
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


class FeatureSelector:
    """
    A class made implementing methods for feature selection
    The recommended order of use is: Normalization --> Feature selection

    :param df: A datafram with only continuous data describing the observations and features
    :type df: pd.Dataframe
    """
    def __init__(self, df: pd.DataFrame):
        """
        Init function of the FeatureSelector class

        :param df: A datafram with only continuous data describing the observations and features
        :type df: pd.Dataframe
        """
        self.df = df

        
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
    def remove_low_variance(self: "FeatureSelector", y: str = "Log_MP_RATIO", variance_threshold: float = 0.05, 
                            verbose: bool = False, inplace: bool = False) -> tuple[pd.DataFrame, list]:
        """
        Remove features with a variance level below the threshold

        Parameters
        ----------
        df: pd.DataFrame
            A dataframe with only continuous data describing the observations and features
        y: str
            The depedent variable (it will be ignored in the removel of features)
        variance_threshold: int
            The threshold at which features below should deleted
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
        df_clone: pd.DataFrame = self.df.loc[:, self.df.columns != y].copy()
        
        # Computes the mean of the variance of each column and deduces the 
        # value to delete that will be below the percentage given by the user
        computed_treshold: int = self.df.var(axis=1).mean() * variance_threshold
        #computed_treshold = variance_threshold   

        if verbose:
            print("===== INITIAL SHAPE =====")
            print(df_clone.shape)
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

        cleaned_df[y] = self.df[y]
        if verbose:
            print("===== DF CLONE FINAL =====")
            print(cleaned_df.shape)

        if inplace:
            self.df = cleaned_df.copy()

        return cleaned_df, deleted_features
    
    def get_correlation_to_y(self: "FeatureSelector", df: pd.DataFrame = None, y: str = "Log_MP_RATIO") -> pd.DataFrame:
        """
        Calculates a correlation score of all the features depending on the y variable

        Parameters
        ----------
        df: pd.DataFrame
            A dataframe with only continuous data describing the observations and features
        y: str
            The dependent variable we want to compare for the correlation

        Returns
        ----------
        pd.DataFrame
            A dataframe with the score of correlation for each feature on the y variable
        """
        if df is None:
            df = self.df.copy()

        print(df.loc[:, df.columns != y].shape)
        print(df[y].shape)
        X = df.loc[:, df.columns != y]
        y = df[y]

        mi = feature_selection.mutual_info_regression(X, y)
        mi /= np.max(mi)
        mi = mi.reshape(-1, len(mi))
        print(mi)
        df_mi: pd.DataFrame = pd.DataFrame(mi, columns=X.columns)
        return df_mi

    def get_correlation(self, df: pd.DataFrame = None, y: str = "Log_MP_RATIO", method: str = "kendall") -> pd.DataFrame:
        #https://datascience.stackexchange.com/a/64261
        """
        Calculates a correlation score of all the features

        Parameters
        ----------
        df: pd.DataFrame
            A dataframe with only continuous data describing the observations and features
        y: str
            Dependent variable to ignore in the colinearity test
        method: str
            Method to use to calculate the correlation between the features ("pearson", "kendall" or "spearman")

        Returns
        ----------
        pd.DataFrame
            A dataframe with the score of correlation for each feature on the y variable
        """
        if df is None:
            df = self.df.copy()

        # Here we create a new dataframe made up from the original dataframe columns + Features as the first column
        # The Features column holds all the names of the features to make a nxn table of correlation
        # df: pd.DataFrame = df.loc[:, df.columns != y]
        # columns_names: list = df.columns
        # df_result: pd.DataFrame = pd.DataFrame(index=columns_names, columns=columns_names)

        # counter: int = 0
        # list_rows: list = []
        # for column in df:
        # mi = feature_selection.mutual_info_regression(df, df[column])
        # mi /= np.max(mi)
        # df_result[column] = mi

        #    mi = feature_selection.mutual_info_regression(df.iloc[:, counter:], df[column])

        #    mi /= np.max(mi)
        #    mi[0:counter] = np.NaN
        #    list_rows.append(mi.tolist())
        # df_result[column] = mi
        #    counter = counter + 1
        #    print(counter)

        # print(len(list_rows[0]))
        # print(len(list_rows[200]))
        # print(df_result.head)
        df = df.loc[:, df.columns != y]
        df_corr: pd.DataFrame = df.corr(method)
        return df_corr.dropna(how="all", axis=1).dropna(how="all")

    def remove_highly_correlated(self: "FeatureSelector", df_correlation: pd.DataFrame, 
                                 df: pd.DataFrame = None, threshold: float = 0.9,
                                 verbose: bool = False, inplace = False) -> pd.DataFrame:
        #used: https://stackoverflow.com/a/61938339
        """
        Removes all the features that have correlation above the threshold

        Parameters
        ----------
        df_correlation: pd.DataFrame
            A dataframe representing the correlation matrix 
        df (default = None): pd.DataFrame
            A dataframe with only continuous data describing the observations and features
            If the default value is given it will use the dataframe contained in the FeatureSelector object
        threshold (default = 0.9): float
            The treshold where features correlated beyond should be dropped
        verbose (default = False): bool
            If set to True displays some informations
        inplace (default = False): bool
            If set to True replaces the attribute df of the FeatureSelector object with the normalized dataframe
        """
        if df is None:
            df = self.df.copy()

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
                    drop_cols.append(col.values[0])

        drops: set = set(drop_cols)
        if verbose:
            print("num of cols to drop: ", len(drops))
        dropped_df: pd.DataFrame = df.drop(columns=drops).copy()
        
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



def open_data(path: str, delimiter: str = ";") -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(path, delimiter=delimiter)
    return df










def find_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to find the correlation between all the features in the dataframe

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe with only continuous data describing the observations and features

    Returns
    ----------
    pd.DataFrame

    """
    return 
    # TODO: Check this https://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html#sphx-glr-auto-examples-feature-selection-plot-f-test-vs-mi-py

# df = open_data("../../Data/full_dataset_test_divprio.csv", ";")
# df_normalized: pd.DataFrame = scale_data(df)
# df_2, l = remove_low_variance(df)
# print(df_2.head())
# print(l)
# mutual_info_reg(df_normalized)
