import numpy as np
from deepchem.models.molgan import BasicMolGANModel
from deepchem.models.optimizers import ExponentialDecay
from tensorflow import one_hot
import deepchem as dc

from qsar.gan.gan_featurizer import QsarGanFeaturizer


class QsarGan:
    def __init__(self,
                 learning_rate: ExponentialDecay,
                 featurizer: QsarGanFeaturizer,
                 edges: int = 5,
                 nodes: int = 5,
                 embedding_dim: int = 10,
                 dropout_rate: float = 0.0,
                 **kwargs):
        """
        Initialize the model

        Parameters
        ----------
        learning_rate: ExponentialDecay
            Learning rate scheduler
        featurizer: QsarGanFeaturizer
            Featurizer used to convert SMILES to features and extract descriptors
        edges: int, default 5
            Number of bond types includes BondType.Zero
        nodes: int, default 5
            Number of atom types in node features matrix
        embedding_dim: int, default 10
            Size of noise input array
        dropout_rate: float, default = 0.
            Rate of dropout used across whole model
        name: str, default ''
            Name of the model
        """

        self.featurizer = featurizer
        self.gan = BasicMolGANModel(
            learning_rate=learning_rate,
            edges=edges,
            vertices=self.featurizer.max_atom_count,
            nodes=nodes,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            **kwargs)

    def _iterbatches(self, epochs, features):
        dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in features], [x.node_features for x in features])
        for i in range(epochs):
            for batch in dataset.iterbatches(batch_size=self.gan.batch_size, pad_batches=True):
                adjacency_tensor = one_hot(batch[0], self.gan.edges)
                node_tensor = one_hot(batch[1], self.gan.nodes)
                yield {self.gan.data_inputs[0]: adjacency_tensor, self.gan.data_inputs[1]: node_tensor}

    def fit_predict(self, features: np.ndarray, epochs=32, generator_steps=0.2, checkpoint_interval=5000,
                    number_to_generate=10000) -> list:
        """
        Fit the model and return the generated molecules.

        Parameters
        ----------
        features: np.ndarray
            Array of features (Array of GraphMatrix).
        epochs: int, default 32
            Number of epochs to train the model
        generator_steps: float, default 0.2
            Number of generator steps per discriminator step
        checkpoint_interval: int, default 5000
            Number of steps between saving model checkpoints
        number_to_generate: int, default 1000
            Number of molecules to generate
        Returns
        -------
        Generated molecules
        """
        self.gan.fit_gan(self._iterbatches(epochs, features), generator_steps=generator_steps,
                         checkpoint_interval=checkpoint_interval)

        generated_data = self.gan.predict_gan_generator(number_to_generate)

        generated_data = self.featurizer.defeaturize(generated_data)
        return self.featurizer.get_unique_smiles(generated_data)
