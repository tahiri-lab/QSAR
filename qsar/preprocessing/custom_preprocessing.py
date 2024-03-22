"""
The LowVarianceRemover and HighCorrelationRemover are designed as sklearn-compatible transformers,
making them suitable for inclusion in sklearn Pipeline objects. They facilitate the automatic removal of
features based on variance and correlation criteria, simplifying the data preprocessing steps required
for effective QSAR modeling.

The PreprocessingPipeline class combines these individual transformers into a single pipeline,
ensuring a coherent and orderly application of feature selection procedures. This custom pipeline
can be directly integrated with other sklearn processes, offering a versatile tool for QSAR data preparation.
"""


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from qsar.preprocessing.feature_selector import FeatureSelector


class LowVarianceRemover(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove features with low variance from a dataset.
    """
    def __init__(self, y, variance_threshold, cols_to_ignore, verbose):
        """
        Initialize the transformer.

        :param y: target variable name.
        :param variance_threshold: variance threshold to remove features.
        :param cols_to_ignore: columns to ignore.
        :param verbose: verbosity level.
        """
        self.y = y
        self.variance_threshold = variance_threshold
        self.cols_to_ignore = cols_to_ignore
        self.verbose = verbose

    def fit(self, x, y=None):
        """
        Fit the transformer.

        :param x: input features.
        :param y: target variable.
        :return: self.
        """
        return self

    def transform(self, x):
        """
        Transform the input features.

        :param x: input features.
        :return: transformed features.
        """
        fs = FeatureSelector(x, self.y, self.cols_to_ignore)
        return fs.remove_low_variance(
            self.y,
            self.variance_threshold,
            self.cols_to_ignore,
            self.verbose,
            inplace=True,
        )


class HighCorrelationRemover(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove features with high correlation from a dataset.
    """
    def __init__(self, df_correlation, df_corr_y, threshold, verbose):
        """
        Initialize the transformer.

        :param df_correlation: correlation matrix.
        :param df_corr_y: correlation with target variable.
        :param threshold: correlation threshold to remove features.
        :param verbose: verbosity level.
        """
        self.df_correlation = df_correlation
        self.df_corr_y = df_corr_y
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, x, y=None):
        """
        Fit the transformer.

        :param x: input features.
        :param y: target variable.
        :return: self.
        """
        return self

    def transform(self, x):
        """
        Transform the input features.

        :param x: input features.
        :return: transformed features (without highly correlated features).
        """
        fs = FeatureSelector(x[0])
        return fs.remove_highly_correlated(
            self.df_correlation,
            self.df_corr_y,
            x[0],
            self.threshold,
            self.verbose,
            inplace=True,
        )


class PreprocessingPipeline:
    """
    Custom preprocessing pipeline composed of two custom transformers:
    - LowVarianceRemover: removes features with low variance
    - HighCorrelationRemover: removes features with high correlation
    """
    def __init__(
        self,
        target="Log_MP_RATIO",
        variance_threshold=0,
        cols_to_ignore=None,
        verbose=False,
        threshold=0.9,
    ):
        """
        Initialize the pipeline with the given parameters.

        :param target: target variable name. Default is "Log_MP_RATIO".
        :param variance_threshold: variance threshold to remove features. Default is 0.
        :param cols_to_ignore: columns to ignore. Default is None.
        :param verbose: verbosity level. Default is False.
        :param threshold: correlation threshold to remove features. Default is 0.9.
        """
        if cols_to_ignore is None:
            cols_to_ignore = []
        self.target = target
        self.variance_threshold = variance_threshold
        self.cols_to_ignore = cols_to_ignore
        self.verbose = verbose
        self.threshold = threshold

    def get_pipeline(self):
        """
        Get the preprocessing pipeline composed of two custom transformers. Start with LowVarianceRemover and then
        apply HighCorrelationRemover to the output of the first transformer.

        :return: preprocessing pipeline.
        """
        pipeline = Pipeline(
            [
                (
                    "low_variance_remover",
                    LowVarianceRemover(
                        y=self.target,
                        variance_threshold=self.variance_threshold,
                        cols_to_ignore=self.cols_to_ignore,
                        verbose=self.verbose,
                    ),
                ),
                (
                    "high_correlation_remover",
                    HighCorrelationRemover(
                        df_correlation=None,
                        df_corr_y=None,
                        threshold=self.threshold,
                        verbose=self.verbose,
                    ),
                ),
            ]
        )
        return pipeline
