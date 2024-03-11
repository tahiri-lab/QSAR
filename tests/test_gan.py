from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from deepchem.models.optimizers import ExponentialDecay

from qsar.gan.gan_featurizer import QsarGanFeaturizer
from qsar.gan.qsar_gan import QsarGan


@pytest.fixture
def mock_featurizer():
    featurizer = MagicMock(spec=QsarGanFeaturizer)
    featurizer.max_atom_count = 5
    return featurizer


@pytest.fixture
def mock_learning_rate():
    return ExponentialDecay(initial_rate=0.001, decay_rate=0.9, decay_steps=1000)


@pytest.fixture
def mock_features():
    return np.random.rand(10, 5, 5)


def test_qsar_gan_initialization(mock_featurizer, mock_learning_rate):
    model = QsarGan(
        learning_rate=mock_learning_rate,
        featurizer=mock_featurizer,
        edges=5,
        nodes=5,
        embedding_dim=10,
        dropout_rate=0.1,
    )
    assert (
        model.featurizer == mock_featurizer
    ), "Featurizer should be correctly assigned"
    assert model.gan is not None, "GAN model should be instantiated"


@patch('deepchem.models.molgan.BasicMolGANModel.fit_gan')
@patch('deepchem.models.molgan.BasicMolGANModel.predict_gan_generator')
def test_fit_predict(mock_predict_gan, mock_fit_gan, mock_featurizer, mock_features):
    model = QsarGan(
        learning_rate=mock_learning_rate,
        featurizer=mock_featurizer,
        edges=5,
        nodes=5,
        embedding_dim=10,
        dropout_rate=0.1
    )
    mock_predict_gan.return_value = mock_features  # Mock the prediction output
    mock_featurizer.defeaturize.return_value = mock_features  # Mock the defeaturization process
    mock_featurizer.get_unique_smiles.return_value = ['CCC', 'CCO', 'CCN']  # Mock unique smiles generation

    generated_smiles = model.fit_predict(features=mock_features, epochs=1, generator_steps=0.2)
    assert isinstance(generated_smiles, list), "Output should be a list"
    assert len(generated_smiles) > 0, "There should be at least one generated SMILES"
