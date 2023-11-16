from collections import OrderedDict
from typing import Tuple, Any

import numpy as np
import pandas as pd
from deepchem.feat import MolGanFeaturizer, GraphMatrix
from pandas import Series, DataFrame
from rdkit import Chem
from rdkit.Chem import Mol


class QsarGanFeaturizer(MolGanFeaturizer):
    def __init__(self, **kwargs):
        self.max_atom_count = 36
        super(QsarGanFeaturizer, self).__init__(**kwargs)

    @staticmethod
    def _get_atom_count(smiles_val: str) -> int:
        mol = Chem.MolFromSmiles(smiles_val)
        return mol.GetNumHeavyAtoms() if mol else 0

    def determine_atom_count(self, smiles: pd.DataFrame, quantile: float = 0.95) -> tuple[int, DataFrame]:
        atoms_count = smiles['smiles'].apply(self._get_atom_count)
        self.max_atom_count = int(atoms_count.quantile(quantile))
        return self.max_atom_count, atoms_count

    def _filter_smiles(self, smiles: pd.Series, num_atoms: int = None) -> np.ndarray:
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
        nmols = list(filter(lambda x: x is not None, nmols))
        nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
        nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
        nmols_viz = [Chem.MolFromSmiles(x) for x in nmols_smiles_unique]
        return nmols_viz

    def get_features(self, smiles: pd.DataFrame, log_every_n: int = 1000, **kwargs) -> np.ndarray:
        filtered_smiles = self._filter_smiles(smiles['smiles'].values)
        mol_objects = [Chem.MolFromSmiles(sm) for sm in filtered_smiles]
        features = [self.featurize([mol]) for mol in mol_objects]
        valid_features = [f[0] for f in features if isinstance(f[0], GraphMatrix)]
        return np.array(valid_features)
