"""
The class supports operations such as counting heavy atoms in molecules, filtering molecules based on atom counts,
determining the appropriate atom count based on a dataset's distribution, and converting SMILES strings into unique,
feature-encoded molecular formats compatible with GAN inputs. The design aims to streamline the preparation of chemical
datasets for QSAR modeling in a GAN framework, focusing on molecular feature extraction and preprocessing.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
from deepchem.feat import GraphMatrix, MolGanFeaturizer
from rdkit import Chem


class QsarGanFeaturizer(MolGanFeaturizer):
    """
    Featurizes molecules for a Generative Adversarial Network (GAN) model using the RDKit and DeepChem libraries.

    The class is responsible for processing SMILES strings into a format suitable for GAN models in QSAR applications.
    """

    def __init__(self, **kwargs):
        """
        Initializes the QsarGanFeaturizer with a maximum atom count of 9.
        """
        super().__init__(**kwargs)
        self.max_atom_count = 9

    @staticmethod
    def _get_atom_count(smiles_val: str) -> int:
        """
        Returns the number of heavy atoms in a molecule represented by a SMILES string.

        :param smiles_val: A SMILES string representing a molecule.
        :type smiles_val: str
        :return: The number of heavy atoms in the molecule.
        :rtype: int
        """
        mol = Chem.MolFromSmiles(smiles_val)
        return mol.GetNumHeavyAtoms() if mol else 0

    def determine_atom_count(
        self, smiles: pd.DataFrame, quantile: float = 0.95
    ) -> tuple[int, pd.Series]:
        """
        Determines the atom count for a DataFrame of SMILES strings.

        :param smiles: A DataFrame of SMILES strings.
        :type smiles: pd.DataFrame
        :param quantile: The quantile to use when determining the atom count. Default is 0.95.
        :type quantile: float
        :return: A tuple containing the atom count and a DataFrame of atom counts.
        :rtype: tuple[int, DataFrame]
        """
        atoms_count = smiles["smiles"].apply(self._get_atom_count)
        return int(atoms_count.quantile(quantile)), atoms_count

    def _filter_smiles(self, smiles: pd.Series, num_atoms: int = None) -> np.ndarray:
        """
        Filters SMILES strings based on the number of atoms.

        :param smiles: A Series of SMILES strings.
        :type smiles: pd.Series
        :param num_atoms: The maximum number of atoms a molecule can have to be included. Default is None.
        :type num_atoms: int
        :return: An array of filtered SMILES strings.
        :rtype: np.ndarray
        """
        if num_atoms is None:
            num_atoms = self.max_atom_count

        filtered_smiles = []
        for x in smiles:
            mol = Chem.MolFromSmiles(x)
            if mol is not None and mol.GetNumAtoms() < num_atoms:
                filtered_smiles.append(x)
        return np.array(filtered_smiles)

    @staticmethod
    def get_unique_smiles(nmols: np.ndarray) -> list:
        """
        Returns a list of unique SMILES strings.

        :param nmols: An array of molecules.
        :type nmols: np.ndarray
        :return: A list of unique SMILES strings.
        :rtype: list
        """
        nmols = list(filter(lambda x: x is not None, nmols))
        nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
        nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
        nmols_viz = [Chem.MolFromSmiles(x) for x in nmols_smiles_unique]
        return nmols_viz

    def get_features(self, smiles: pd.DataFrame) -> np.ndarray:
        """
        Returns the features for a DataFrame of SMILES strings.

        :param smiles: A DataFrame of SMILES strings.
        :type smiles: pd.DataFrame
        :return: An array of features for the SMILES strings.
        :rtype: np.ndarray
        """
        filtered_smiles = self._filter_smiles(smiles["smiles"].values)
        mol_objects = [Chem.MolFromSmiles(sm) for sm in filtered_smiles]
        features = [self.featurize([mol]) for mol in mol_objects]
        valid_features = [f[0] for f in features if isinstance(f[0], GraphMatrix)]
        return np.array(valid_features)
