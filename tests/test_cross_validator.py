import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from qsar.utils.cross_validator import CrossValidator


@pytest.fixture
def sample_dataframe():
    """Creates a sample DataFrame to be used in tests."""
    data = {
        'feature1': np.random.rand(15),
        'feature2': np.random.rand(15),
        'Log_MP_RATIO': np.random.rand(15)
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_model():
    # Create a mock model with the appropriate predict method behavior
    model = MagicMock()
    model.score.return_value = 0.9
    model.fit.return_value = model

    # Set the predict method to return arrays of appropriate lengths
    # The side_effect should be a function to handle dynamic input sizes
    def predict_side_effect(X):
        return np.random.rand(len(X))

    # Assign this side_effect to the predict method of the mock
    model.predict.side_effect = predict_side_effect

    # Ensure that when the model is cloned, it returns a mock with the same setup
    model.clone.return_value = model

    return model


def test_create_cv_folds(sample_dataframe):
    # Test creating cross-validation folds
    validator = CrossValidator(sample_dataframe)
    X_list, y_list, df, y, n_folds = validator.create_cv_folds()

    assert isinstance(X_list, list), "X_list should be a list"
    assert isinstance(y_list, list), "y_list should be a list"
    assert all(isinstance(x, pd.DataFrame) for x in X_list), "All elements in X_list should be DataFrames"
    assert len(X_list) == n_folds, "X_list length should be equal to n_folds"
    assert 'Fold' in df.columns, "Fold column should be added to the DataFrame"


def test_cross_value_score(sample_dataframe, mock_model):
    validator = CrossValidator(sample_dataframe)
    validator.cross_value_score(mock_model)
    score = mock_model.score()
    assert isinstance(score, float), "Score should be a float"


def test_get_predictions(sample_dataframe, mock_model):
    """Test getting predictions."""
    validator = CrossValidator(sample_dataframe)
    X_train, X_test = sample_dataframe.iloc[:10, :-1], sample_dataframe.iloc[10:, :-1]  # Exclude target variable
    y_train = sample_dataframe['Log_MP_RATIO'][:10]

    validator.get_predictions(mock_model, X_train, y_train, X_test)

    y_pred_train = mock_model.predict(X_train)
    y_pred_test = mock_model.predict(X_test)

    assert len(y_pred_train) == len(X_train), "Length of y_pred_train should match X_train"
    assert len(y_pred_test) == len(X_test), "Length of y_pred_test should match X_test"
