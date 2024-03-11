import numpy as np
import pandas as pd
import pytest

from qsar.preprocessing.feature_selector import FeatureSelector


@pytest.fixture
def sample_data():
    data = {
        "feature1": np.random.rand(10),
        "feature2": np.random.rand(10),
        "feature3": np.random.rand(10),
        "Log_MP_RATIO": np.random.rand(10),
    }
    return pd.DataFrame(data)


def test_init(sample_data):
    fs = FeatureSelector(sample_data)
    assert isinstance(fs, FeatureSelector), "FeatureSelector instance should be created"
    assert fs.y == "Log_MP_RATIO", "Default dependent variable should be 'Log_MP_RATIO'"
    assert fs.cols_to_ignore == [], "Default cols_to_ignore should be an empty list"


def test_scale_data(sample_data):
    fs = FeatureSelector(sample_data)
    normalized_df = fs.scale_data(verbose=False, inplace=False)
    assert (
        "Log_MP_RATIO" in normalized_df.columns
    ), "Dependent variable should not be removed"
    assert (
        normalized_df["feature1"].min() >= 0 and normalized_df["feature1"].max() <= 1
    ), "Data should be normalized"


def test_remove_low_variance(sample_data):
    fs = FeatureSelector(sample_data)
    cleaned_df, deleted_features = fs.remove_low_variance(
        variance_threshold=0.01, verbose=False, inplace=False
    )
    assert isinstance(cleaned_df, pd.DataFrame), "Should return a DataFrame"
    assert isinstance(
        deleted_features, list
    ), "Should return a list of deleted features"


def test_get_correlation_to_y(sample_data):
    fs = FeatureSelector(sample_data)
    correlation_series = fs.get_correlation_to_y()
    assert isinstance(correlation_series, pd.DataFrame), "Should return a DataFrame"


def test_get_correlation(sample_data):
    fs = FeatureSelector(sample_data)
    correlation_matrix = fs.get_correlation()
    assert isinstance(correlation_matrix, pd.DataFrame), "Should return a DataFrame"
    assert correlation_matrix.shape == (
        3,
        3,
    ), "Correlation matrix should not include the dependent variable"


def test_remove_highly_correlated(sample_data):
    fs = FeatureSelector(sample_data)
    cleaned_df = fs.remove_highly_correlated(
        threshold=0.9, verbose=False, inplace=False
    )
    assert isinstance(cleaned_df, pd.DataFrame), "Should return a DataFrame"
