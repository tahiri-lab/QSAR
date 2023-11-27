import pandas as pd
from sklearn import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score


class CrossValidator:
    """
    Class for cross-validation related functionalities.

    :ivar df: DataFrame containing the data.
    :vartype df: pd.DataFrame
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize a CrossValidator instance.

        :param df: DataFrame containing the data.
        :type df: pd.DataFrame
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

        :param df: DataFrame to be used. If not provided, a default will be used.
        :type df: pd.DataFrame, optional
        :param y: Target column name. Defaults to 'Log_MP_RATIO'.
        :type y: str, optional
        :param n_folds: Number of folds. Defaults to 3.
        :type n_folds: int, optional
        :param n_groups: Number of groups for stratified k-fold. Defaults to 5.
        :type n_groups: int, optional
        :returns: A tuple containing a list of feature sets, a list of targets, a DataFrame with fold information, the target column name, and the number of folds.
        :rtype: tuple
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

    def cross_value_score(self, model, df: pd.DataFrame = None) -> float:
        """
        Compute cross-validation score for the given model.

        :param model: The model to be evaluated.
        :type model: Model
        :param df: DataFrame to be used, if not provided, default is used.
        :type df: pd.DataFrame, optional
        :returns: Mean cross-validation score.
        :rtype: float
        """
        if df is None:
            df = self.df.copy()

        X_list, y_list, *_ = self.create_cv_folds(df)

        mean_cv_score: list = []
        for i in range(len(X_list)):
            model_scorer = clone(model)

            X_train = X_list[:-1]
            X_test = X_list[-1]

            y_train = y_list[:-1]
            y_test = y_list[-1]

            X_train = pd.concat(X_train)
            y_train = pd.concat(y_train)

            model_scorer.fit(X_train, y_train)
            mean_cv_score.append(model_scorer.score(X_test, y_test))

            X_list.insert(0, X_list.pop())
            y_list.insert(0, y_list.pop())

        return sum(mean_cv_score) / len(mean_cv_score)

    def evaluate_model_performance(self, model, X_train, y_train, X_test, y_test):
        """
        Compute various scores for model evaluation.

        :param model: The model to be evaluated.
        :type model: Model
        :param X_train: Training feature set.
        :type X_train: pd.DataFrame
        :param y_train: Training target set.
        :type y_train: pd.DataFrame
        :param X_test: Testing feature set.
        :type X_test: pd.DataFrame
        :param y_test: Testing target set.
        :type y_test: pd.DataFrame
        :returns: A tuple containing the R squared score, CV score, custom CV score, and Q squared score.
        :rtype: tuple
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

    @staticmethod
    def get_predictions(
            model,
            x_train: pd.DataFrame,
            y_train: pd.DataFrame,
            x_test: pd.DataFrame,
    ) -> tuple:
        """
        Get predictions using the provided model.

        :param model: The model to be used for prediction.
        :type model: object or model instance
        :param x_train: Training feature set.
        :type x_train: pd.DataFrame
        :param y_train: Training target set.
        :type y_train: pd.DataFrame
        :param x_test: Testing feature set.
        :type x_test: pd.DataFrame
        :returns: A tuple containing predictions on the training set and predictions on the testing set.
        :rtype: tuple
        """
        model_scorer = clone(model)
        model_scorer.fit(x_train, y_train)
        y_pred_train = model_scorer.predict(x_train)
        y_pred_test = model_scorer.predict(x_test)
        return y_pred_train, y_pred_test
