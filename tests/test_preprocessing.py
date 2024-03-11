import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from qsar.preprocessing.custom_preprocessing import (HighCorrelationRemover,
                                                     LowVarianceRemover,
                                                     PreprocessingPipeline)


@pytest.fixture
def sample_data():
    data = {
        "feature1": np.random.rand(15),
        "feature2": np.random.rand(15),
        "feature3": np.random.rand(15),
        "Log_MP_RATIO": np.random.rand(15),
    }
    return pd.DataFrame(data)


def test_low_variance_remover(sample_data):
    transformer = LowVarianceRemover(
        y="Log_MP_RATIO", variance_threshold=0.01, cols_to_ignore=[], verbose=False
    )
    transformed_data = transformer.transform(sample_data)[0]
    assert isinstance(transformed_data, pd.DataFrame), "Should return a DataFrame"


def test_high_correlation_remover(sample_data):
    transformer = HighCorrelationRemover(
        df_correlation=None, df_corr_y=None, threshold=0.9, verbose=False
    )
    transformed_data = transformer.transform([sample_data])
    assert isinstance(transformed_data, pd.DataFrame), "Should return a DataFrame"


def test_preprocessing_pipeline(sample_data):
    pipeline = PreprocessingPipeline(
        target="Log_MP_RATIO",
        variance_threshold=0.01,
        cols_to_ignore=[],
        verbose=False,
        threshold=0.9,
    )
    pipe = pipeline.get_pipeline()
    assert isinstance(pipe, Pipeline), "Should return a sklearn Pipeline instance"
    transformed_data = pipe.fit_transform(sample_data)
    assert isinstance(
        transformed_data, pd.DataFrame
    ), "Pipeline should output a DataFrame"
