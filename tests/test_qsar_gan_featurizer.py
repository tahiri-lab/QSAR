import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from qsar.gan.gan_featurizer import QsarGanFeaturizer


@pytest.fixture
def featurizer():
    return QsarGanFeaturizer()


@pytest.fixture
def sample_smiles():
    return pd.DataFrame({"smiles": ["CCO", "N=C=O", "C#N", "[Na]", "C(C)(C)C=C"]})


def test_initialization(featurizer):
    assert featurizer.max_atom_count == 9, "Default max atom count should be 9"


def test_get_atom_count():
    assert (
        QsarGanFeaturizer._get_atom_count("CCO") == 3
    ), "Incorrect atom count for 'CCO'"
    assert (
        QsarGanFeaturizer._get_atom_count("") == 0
    ), "Empty string should return 0 atoms"


def test_determine_atom_count(featurizer, sample_smiles):
    print(sample_smiles)
    print(type(sample_smiles))
    count, df = featurizer.determine_atom_count(sample_smiles)
    print(type(df))
    assert isinstance(count, int), "Atom count should be an integer"
    assert isinstance(df, pd.Series), "Should return a Series"
    assert count > 0, "Atom count should be greater than 0"


def test_filter_smiles(featurizer, sample_smiles):
    filtered = featurizer._filter_smiles(sample_smiles["smiles"], 3)
    assert isinstance(filtered, np.ndarray), "Filtered SMILES should be an ndarray"
    assert len(filtered) <= len(
        sample_smiles
    ), "Filtered SMILES should not exceed original count"


def test_get_unique_smiles():
    unique_smiles = QsarGanFeaturizer.get_unique_smiles(
        np.array([Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCO"), None])
    )
    assert isinstance(unique_smiles, list), "Should return a list"
    assert len(unique_smiles) == 1, "Should only contain unique items"


def test_get_features(featurizer, sample_smiles):
    features = featurizer.get_features(sample_smiles)
    assert isinstance(features, np.ndarray), "Features should be an ndarray"
