import numpy as np
import pandas as pd
from deepchem.feat import MolGanFeaturizer, GraphMatrix
from rdkit import Chem


class QsarGanFeaturizer(MolGanFeaturizer):
    def __init__(self, **kwargs):
        super(QsarGanFeaturizer, self).__init__(**kwargs)

    @staticmethod
    def get_atom_count(smiles_val: str) -> int:
        """
        Get the number of atoms in a SMILES string.
        Args:
            smiles_val: SMILES string to be processed.

        Returns:
            Number of atoms in the SMILES string.
        """
        mol = Chem.MolFromSmiles(smiles_val)
        return mol.GetNumAtoms() if mol is not None else 0

    def determine_atom_count(self, smiles: pd.DataFrame, quantile: float = 0.95) -> int:
        """
        Determine the maximum number of atoms in a SMILES string.
        Args:
            smiles: DataFrame containing SMILES strings.
            quantile: Quantile to be used for determining the maximum number of atoms.

        Returns:
            Maximum number of atoms in the SMILES strings.
        """
        smiles['atom_count'] = smiles['smiles'].apply(self.get_atom_count)
        self.vertices = int(smiles['atom_count'].quantile(quantile))
        return self.vertices

    def filter_smiles(self, smiles: pd.DataFrame, num_atoms: int = None) -> pd.DataFrame:
        """
        Filter SMILES strings based on the number of atoms.
        Args:
            smiles: DataFrame containing SMILES strings.
            num_atoms: Number of atoms to be used for filtering.
        Returns:
            DataFrame containing SMILES strings with less than num_atoms.
        """
        if num_atoms is None:
            num_atoms = self.vertices

        return smiles[smiles['smiles'].apply(self.get_atom_count) < num_atoms]

    def get_unique_smiles(self, smiles: pd.DataFrame) -> list:
        """
        Get unique SMILES strings.
        Args:
            smiles: DataFrame containing SMILES strings.

        Returns:
            List of unique SMILES strings.
        """
        unique_smiles = smiles['smiles'].dropna().unique()
        return [sm for sm in unique_smiles if Chem.MolFromSmiles(sm) is not None]

    def get_features(self, datapoints, log_every_n=1000, **kwargs) -> np.ndarray:
        """
        Get features for a list of SMILES strings.
        Args:
            datapoints: List of SMILES strings.
            log_every_n: Logging messages reported every `log_every_n` samples.
            **kwargs: Additional keyword arguments.
        Returns:

        """
        features = self.featurize(datapoints, log_every_n=log_every_n, **kwargs)
        valid_features = [f for f in features if isinstance(f, GraphMatrix)]
        return np.array(valid_features)
