import math

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from qsar.models.model import Model


class CrossValidator:
    """
    Class for cross-validation related functionalities.

    Attributes:
    - df (pd.DataFrame): DataFrame containing the data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize a CrossValidator instance.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the data.
        """
        self.df = df

    def create_cv_folds(
            self,
            df: pd.DataFrame = None,
            y: str = "Log_MP_RATIO",
            n_folds: int = 3,
            n_groups: int = 5,
    ) -> tuple:
        """
        Create cross-validation folds.

        Parameters:
        - df (pd.DataFrame, optional): DataFrame to be used. Default is None.
        - y (str, optional): Target column name. Default is 'Log_MP_RATIO'.
        - n_folds (int, optional): Number of folds. Default is 3.
        - n_groups (int, optional): Number of groups for stratified k-fold. Default is 5.

        Returns:
        - tuple: List of feature sets, list of targets, DataFrame with fold information, target column name, number of folds.
        """
        if df is None:
            df = self.df.copy()

        if n_groups is None:
            skf = KFold(n_splits=n_folds)
            target = df.target
        else:
            skf = StratifiedKFold(n_splits=n_folds)
            df["grp"] = pd.cut(df[y], n_groups, labels=False)
            target = df.grp

        for fold_no, (t, v) in enumerate(skf.split(target, target)):
            df.loc[v, "Fold"] = fold_no

        X_list = []
        y_list = []

        for i in range(n_folds):
            X_list.append(
                df.loc[df["Fold"] == i].copy().drop(columns=["grp", "Fold", y])
            )
            y_list.append(df.loc[df["Fold"] == i][y].copy())
        return X_list, y_list, df, y, n_folds

    def cross_value_score(self, model: Model, df: pd.DataFrame = None) -> float:
        """
        Compute cross-validation score for the given model.

        Parameters:
        - model (Model): The model to be evaluated.
        - df (pd.DataFrame, optional): DataFrame to be used. Default is None.

        Returns:
        - float: Mean cross-validation score.
        """
        if df is None:
            df = self.df.copy()

        X_list, y_list = self.create_cv_folds(df)

        mean_cv_score: list = []
        for i in range(len(X_list)):
            Model_scorer = clone(model)

            X_train = X_list[:-1]
            X_test = X_list[-1]

            y_train = y_list[:-1]
            y_test = y_list[-1]

            X_train = pd.concat(X_train)
            y_train = pd.concat(y_train)

            Model_scorer.fit(X_train, y_train)
            mean_cv_score.append(Model_scorer.score(X_test, y_test))

            X_list.insert(0, X_list.pop())
            y_list.insert(0, y_list.pop())

        return sum(mean_cv_score) / len(mean_cv_score)

    def get_score_data(self, model, X_train, y_train, X_test, y_test):
        """
        Compute various scores for model evaluation.

        Parameters:
        - model (Model): The model to be evaluated.
        - X_train (pd.DataFrame): Training feature set.
        - y_train (pd.DataFrame): Training target set.
        - X_test (pd.DataFrame): Testing feature set.
        - y_test (pd.DataFrame): Testing target set.

        Returns:
        - tuple: R squared score, CV score, custom CV score, Q squared score.
        """
        # Copying all values and models to not change the original one
        X_train = X_train.copy()
        y_train = y_train.copy()
        X_test = X_test.copy()
        y_test = y_test.copy()

        model_scorer = clone(model)

        # Creating a train dataframe for the fold creator
        df_train = pd.concat([X_train.copy(), y_train.copy()], axis=1)
        # TODO: adapt this to get custom values for the rest
        custom_cv = self.cross_value_score(clone(model), df_train)

        model_scorer.fit(X_train, y_train)
        R2 = model_scorer.score(X_train, y_train)
        CV = cross_val_score(model_scorer, X_train, y_train, cv=3, scoring="r2").mean()
        Q2 = model_scorer.score(X_test, y_test)

        return R2, CV, custom_cv, Q2

    def get_predictions(self, model: Model, x_train: pd.DataFrame, x_test: pd.DataFrame,
                        y_train: pd.DataFrame) -> tuple:
        """
        Get predictions using the provided model.

        Parameters:
        - model (Model): The model to be used for prediction.
        - x_train (pd.DataFrame): Training feature set.
        - x_test (pd.DataFrame): Testing feature set.
        - y_train (pd.DataFrame): Training target set.

        Returns:
        - tuple: Predictions on the training set, Predictions on the testing set.
        """
        model_scorer = clone(model)
        model_scorer.fit(x_train, y_train)
        y_pred_train = model_scorer.predict(x_train)
        y_pred_test = model_scorer.predict(x_test)
        return y_pred_train, y_pred_test
