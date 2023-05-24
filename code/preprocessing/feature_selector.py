"""
TITLE
"""


import pandas as pd
import numpy as np

from sklearn import preprocessing, feature_selection


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

        
    def scale_data(self: "FeatureSelector", verbose: bool=False, inplace=False) -> pd.DataFrame:
        # It seems data should be normalized since not every data has a gaussian distribution
        """
        Normalize the data of in the CSV
        Even though on linear regression it isn't necessarily needed but it can help for data interpretation

        Parameters
        ----------
        df: pd.DataFrame
            A dataframe with only continuous data describing the observations and features
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

        df_to_normalize: pd.DataFrame = self.df.drop(columns=["Log_MP_RATIO"])

        df_normalized = preprocessing.normalize(self.df, axis=0)
        df_normalized = pd.concat([pd.DataFrame(df_normalized, columns=self.df.columns), self.df["Log_MP_RATIO"]], axis=1)
        if verbose:
            print("===== DESCRIPTION =====")
            print(df_normalized.describe())
        if inplace:
            self.df = df_normalized
        return df_normalized


    # TODO: check avec Nadia "0.01 would mean dropping the column where 99% of the values are similar."
    def remove_low_variance(df: pd.DataFrame, y: str = "Log_MP_RATIO", variance_threshold: float = 0.05) -> tuple[
        pd.DataFrame, list]:
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

        Returns
        ----------
        pd.Dataframe
            The same dataframe as input but the features with low variance are removed
        list
            A list of all the deleted columns
        """
        df_clone: pd.DataFrame = df.loc[:, df.columns != y].copy()
        # print("===== INITIAL SHAPE =====")
        print(df_clone.shape)
        Vt: feature_selection.VarianceThreshold = feature_selection.VarianceThreshold(threshold=variance_threshold)
        high_variance = Vt.fit_transform(df_clone)
        # print("===== CLEANED SHAPE =====")
        print(high_variance.shape)

        # print("===== DELETED FEATURES ======")
        deleted_features: list = [column for column in df_clone if column not in df_clone.columns[Vt.get_support()]]
        # print(deleted_features)

        # print("===== HELLO =====")

        df_clone[y] = df[y]

        return df_clone, deleted_features
    
    def correlation_to_y(df: pd.DataFrame, y: str = "Log_MP_RATIO") -> pd.DataFrame:
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

    def mutual_info_reg(df: pd.DataFrame, y: str = "Log_MP_RATIO", method: str = "kendall") -> pd.DataFrame:
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
