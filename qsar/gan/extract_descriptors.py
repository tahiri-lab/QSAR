import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


class DescriptorsExtractor:
    @staticmethod
    def extract_descriptors(mols: list) -> pd.DataFrame:
        all_discriptors = [(name, func) for name, func in Descriptors.descList]
        features_from_smiles = [[]] * len(mols)
        descriptor_names = [name for name, _ in Descriptors.descList]
        for idx, molecule in enumerate(mols):
            features_from_smiles[idx] = []
            for name, func in all_discriptors:
                features_from_smiles[idx].append(func(molecule))
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        return pd.DataFrame(features_from_smiles, columns=descriptor_names, index=smiles)
