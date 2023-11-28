from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from qsar.preprocessing.feature_selector import FeatureSelector


class LowVarianceRemover(BaseEstimator, TransformerMixin):
    def __init__(self, y, variance_threshold, cols_to_ignore, verbose):
        self.y = y
        self.variance_threshold = variance_threshold
        self.cols_to_ignore = cols_to_ignore
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fs = FeatureSelector(X, self.y, self.cols_to_ignore)
        return fs.remove_low_variance(self.y, self.variance_threshold, self.cols_to_ignore, self.verbose, inplace=True)


class HighCorrelationRemover(BaseEstimator, TransformerMixin):
    def __init__(self, df_correlation, df_corr_y, threshold, verbose):
        self.df_correlation = df_correlation
        self.df_corr_y = df_corr_y
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fs = FeatureSelector(X[0])
        return fs.remove_highly_correlated(self.df_correlation, self.df_corr_y, X[0], self.threshold, self.verbose,
                                           inplace=True)


class PreprocessingPipeline:
    def __init__(self, target="Log_MP_RATIO", variance_threshold=0, cols_to_ignore=None, verbose=False, threshold=0.9):
        if cols_to_ignore is None:
            cols_to_ignore = []
        self.target = target
        self.variance_threshold = variance_threshold
        self.cols_to_ignore = cols_to_ignore
        self.verbose = verbose
        self.threshold = threshold

    def get_pipeline(self):
        pipeline = Pipeline([
            ('low_variance_remover',
             LowVarianceRemover(y=self.target, variance_threshold=self.variance_threshold,
                                cols_to_ignore=self.cols_to_ignore, verbose=self.verbose)),
            ('high_correlation_remover',
             HighCorrelationRemover(df_correlation=None, df_corr_y=None, threshold=self.threshold,
                                    verbose=self.verbose))
        ])
        return pipeline
