import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


class DescriptorsExtractor:
    @staticmethod
    def extract_descriptors(generated_smiles: list) -> pd.DataFrame:
        mols = [Chem.MolFromSmiles(smi) for smi in generated_smiles]
        all_discriptors = [(name, func) for name, func in Descriptors.descList]
        features_from_smiles = [[]] * len(mols)
        descriptor_names = [name for name, _ in Descriptors.descList]
        for idx, molecule in enumerate(mols):
            features_from_smiles[idx] = []
            for name, func in all_discriptors:
                features_from_smiles[idx].append(func(molecule))

        return pd.DataFrame(features_from_smiles, columns=descriptor_names, index=mols)
