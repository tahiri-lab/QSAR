from typing import Dict, Tuple
import os

import pandas as pd


class Extractor:
    def __init__(self, paths: Dict[str, str]):
        """
        Parameters
        ----------
        paths : dict of {str: str} pairs where the key is the name of the dataframe and the value is the path to
        the csv file.
        """
        self.paths = paths
        self.dfs = self.extract_dfs(self.paths)

    def get_df(self, name: str) -> pd.DataFrame:
        """
        Get a df by its name
        Parameters
        ----------
        name : str name of the df.

        Returns
        -------
        found dataframe
        """
        if name not in self.dfs.keys():
            raise KeyError(f"{name} deos not exist.")
        return self.dfs[name]

    def extract_dfs(self, paths: Dict[str, str] = None) -> Dict[str, pd.DataFrame]:
        """
        Extracts dataframes from a dictionary of {name: path} pairs.
        Parameters
        ----------
        paths : dict of {str: str} pairs where the key is the name of the dataframe and the value is the path to
        the csv file.

        Returns
        -------
        dict of {str: pd.DataFrame} pairs where the key is the name of the dataframe and the value is the dataframe.

        """
        self.dfs = dict()
        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} does not exist.")
            self.dfs[name] = pd.read_csv(path)
        return self.dfs

    def split_x_y(self, y_col: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Splits the dataframes into X and y dataframes.
        Parameters
        ----------
        y_col : str
            name of the column to be used as the y values.

        Returns
        -------
        tuple of two dicts
            The first dict contains the X dataframes and the second dict contains the y dataframes.
            Both dicts have string keys corresponding to the names of the dataframes.
        """
        x_dfs = dict()
        y_dfs = dict()
        for name, df in self.dfs.items():
            y_dfs[name] = df[y_col]
            x_dfs[name] = df.drop(columns=y_col)
        return x_dfs, y_dfs
