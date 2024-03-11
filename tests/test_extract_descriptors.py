import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from qsar.gan.extract_descriptors import DescriptorsExtractor


@pytest.fixture
def sample_molecules():
    smiles_list = ["CCO", "C=C", "CCN", "C#N", "CCOCC"]
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list]


def test_extract_descriptors(sample_molecules):
    df = DescriptorsExtractor.extract_descriptors(sample_molecules)
    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert df.shape[0] == len(
        sample_molecules
    ), "DataFrame should have one row per molecule"
    assert df.shape[1] == len(
        Descriptors.descList
    ), "DataFrame should have one column per descriptor"

    descriptor_names = [desc[0] for desc in Descriptors.descList]
    for name in descriptor_names:
        assert name in df.columns, f"Descriptor {name} should be in the DataFrame"

    with pytest.raises(AttributeError):
        DescriptorsExtractor.extract_descriptors(["not_a_molecule"])
