"""
The class supports the initialization with a dictionary mapping dataset names to file paths, automatically loads these datasets
into memory, and provides methods for easy retrieval and manipulation of these datasets.
"""

import os
from typing import Dict, Tuple

import pandas as pd


class Extractor:
    """
    Class for cross-validation related functionalities.

    :param paths: Dictionary of {str: str} pairs where the key is the name of the dataframe and the value is the path to
                 the CSV file.
    :type paths: Dict[str, str]

    :ivar dfs: Extracted DataFrames from the paths provided during initialization.
    :vartype dfs: Dict[str, pd.DataFrame]
    """

    def __init__(self, paths: Dict[str, str]):
        """
        Initialize the Extractor instance.

        :param paths: Dictionary of {str: str} pairs where the key is the name of the dataframe and the value is the
                     path to the CSV file.
        :type paths: Dict[str, str]
        """
        self.paths = paths
        self.dfs = self.extract_dfs(self.paths)

    def get_df(self, name: str) -> pd.DataFrame:
        """
        Get a DataFrame by its name.

        :param name: Name of the DataFrame.
        :type name: str
        :returns: The DataFrame associated with the given name.
        :rtype: pd.DataFrame

        :raises KeyError: If the name does not exist in the stored DataFrames.
        """
        if name not in self.dfs.keys():
            raise KeyError(f"{name} deos not exist.")
        return self.dfs[name]

    def extract_dfs(self, paths: Dict[str, str] = None) -> Dict[str, pd.DataFrame]:
        """
        Extracts DataFrames from a dictionary of {name: path} pairs.

        :param paths: Dictionary of {str: str} pairs where the key is the name of the dataframe and the value is the
                     path to the CSV file. If not provided, defaults to the paths provided at initialization.
        :type paths: Dict[str, str], optional
        :returns: Dictionary of {str: pd.DataFrame} pairs where the key is the name of the dataframe and the value is
                 the DataFrame itself.
        :rtype: Dict[str, pd.DataFrame]

        :raises FileNotFoundError: If a path in the dictionary does not exist.
        """
        self.dfs = {}
        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} does not exist.")
            self.dfs[name] = pd.read_csv(path)
        return self.dfs

    def split_x_y(
        self, y_col: str
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Splits the DataFrames into X and y DataFrames based on the specified column.

        :param y_col: Name of the column to be used as the y values.
        :type y_col: str
        :returns: A tuple containing two dictionaries. The first dictionary contains the X DataFrames and the second
                 dictionary contains the y DataFrames, both keyed by the names of the original DataFrames.
        :rtype: Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]].
        """
        x_dfs = {}
        y_dfs = {}
        for name, df in self.dfs.items():
            y_dfs[name] = df[[y_col]]
            x_dfs[name] = df.drop(columns=[y_col])
        return x_dfs, y_dfs
