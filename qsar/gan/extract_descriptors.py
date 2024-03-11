"""
This module provides the DescriptorsExtractor class for extracting chemical descriptors
from molecules using RDKit. The extracted descriptors include various molecular properties
such as molecular weight, logP, and more, which are useful in cheminformatics for molecule analysis.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


class DescriptorsExtractor:
    """
    Class for extracting descriptors from molecules.
    """

    @staticmethod
    def extract_descriptors(mols: list) -> pd.DataFrame:
        """
        Extracts descriptors from a list of molecules.

        :param mols: A list of RDKit molecule objects.
        :type mols: list
        :return: A DataFrame where each row corresponds to a molecule and each column to a descriptor.
                 The descriptors are computed using the RDKit library.
        :rtype: pd.DataFrame
        """
        all_discriptors = list(Descriptors.descList)
        features_from_smiles = [[]] * len(mols)
        descriptor_names = [name for name, _ in Descriptors.descList]
        for idx, molecule in enumerate(mols):
            features_from_smiles[idx] = []
            for name, func in all_discriptors:
                features_from_smiles[idx].append(func(molecule))
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        return pd.DataFrame(
            features_from_smiles, columns=descriptor_names, index=smiles
        )
