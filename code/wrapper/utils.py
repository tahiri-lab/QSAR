import math

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score


class Utils:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def create_cv_folds(self, df: pd.DataFrame = None, y: str = "Log_MP_RATIO",
                        n_folds: int = 3, n_groups: int = 5,
                        display: bool = False) -> tuple:
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

        if display is True:
            fig, axs = plt.subplots(1, n_folds, sharex=True, sharey=True, figsize=(10, 4))
            for i, ax in enumerate(axs):
                ax.hist(df[df.Fold == i][y], bins=10, density=True, label=f"Fold-{i}")
                if i == 0:
                    ax.set_ylabel("Frequency")
                if i == math.ceil(n_folds / 2):
                    ax.set_xlabel(y)
                ax.legend(frameon=False, handlelength=0)
            plt.tight_layout()
            plt.show()

        X_list = []
        y_list = []

        for i in range(n_folds):
            X_list.append(df.loc[df["Fold"] == i].copy().drop(columns=["grp", "Fold", y]))
            y_list.append(df.loc[df["Fold"] == i][y].copy())
        return X_list, y_list

    def cv_score(self, Model, X_list, y_list) -> int:
        mean_cv_score: list = []
        for i in range(len(X_list)):
            Model_scorer = clone(Model)

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

    def display_score(self: "Utils", Model, X_train, y_train, X_test, y_test):
        # Copying all values and models to not change the original one
        X_train = X_train.copy()
        y_train = y_train.copy()
        X_test = X_test.copy()
        y_test = y_test.copy()

        Model_scorer = clone(Model)

        # Creating a train dataframe for the fold creator
        df_train = pd.concat([X_train.copy(), y_train.copy()], axis=1)
        # TODO: adapt this to get custom values for the rest
        cv_x_train, cv_y_train = self.create_cv_folds(df_train)
        Custom_CV = self.cv_score(clone(Model), cv_x_train, cv_y_train)

        Model_scorer.fit(X_train, y_train)
        R2 = Model_scorer.score(X_train, y_train)
        CV = cross_val_score(Model_scorer, X_train, y_train, cv=3, scoring="r2").mean()
        Q2 = Model_scorer.score(X_test, y_test)

        print("===== ", type(Model).__name__, " =====",
              "\n\tR2\t\t\t:\t", R2,
              "\n\tCV\t\t\t:\t", CV,
              "\n\tCustom CV\t:\t", Custom_CV,
              "\n\tR2\t\t\t:\t", Q2,
              )
