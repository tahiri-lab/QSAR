import pandas as pd
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
        all_discriptors = [(name, func) for name, func in Descriptors.descList]
        features_from_smiles = [[]] * len(mols)
        descriptor_names = [name for name, _ in Descriptors.descList]
        for idx, molecule in enumerate(mols):
            features_from_smiles[idx] = []
            for name, func in all_discriptors:
                features_from_smiles[idx].append(func(molecule))

        return pd.DataFrame(features_from_smiles, columns=descriptor_names, index=mols)
