from deepchem.models.molgan import BasicMolGANModel
from deepchem.models.optimizers import ExponentialDecay
from tensorflow import one_hot
import deepchem as dc

from qsar.gan.gan_featurizer import QsarGanFeaturizer


class QsarGan(BasicMolGANModel):
    def __init__(self,
                 learning_rate: ExponentialDecay,
                 edges: int = 5,
                 vertices: int = 9,
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
        edges: int, default 5
            Number of bond types includes BondType.Zero
        vertices: int, default 9
            Max number of atoms in adjacency and node features matrices
        nodes: int, default 5
            Number of atom types in node features matrix
        embedding_dim: int, default 10
            Size of noise input array
        dropout_rate: float, default = 0.
            Rate of dropout used across whole model
        name: str, default ''
            Name of the model
        """

        self.learning_rate = learning_rate
        self.edges = edges
        self.vertices = vertices
        self.nodes = nodes
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        self.featurizer = QsarGanFeaturizer(max_atom_count=self.vertices)
        super(BasicMolGANModel, self).__init__(**kwargs)

    def _iterbatches(self, epochs, features):
        dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in features], [x.node_features for x in features])
        for i in range(epochs):
            for batch in dataset.iterbatches(batch_size=self.batch_size, pad_batches=True):
                adjacency_tensor = one_hot(batch[0], self.edges)
                node_tensor = one_hot(batch[1], self.nodes)
                yield {self.data_inputs[0]: adjacency_tensor, self.data_inputs[1]: node_tensor}

    def fit_predict(self, smiles, epochs=32, generator_steps=0.2, checkpoint_interval=5000, number_to_generate=10000):
        """
        Fit the model and return the generated molecules.

        Parameters
        ----------
        smiles: np.ndarray
            Array of SMILES strings.
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

        self.vertices = self.featurizer.determine_atom_count(smiles)
        features = self.featurizer.get_features(smiles)

        self.fit_gan(self._iterbatches(epochs, features), generator_steps=generator_steps,
                     checkpoint_interval=checkpoint_interval)

        generated_data = self.predict_gan_generator(number_to_generate)

        generated_data = self.featurizer.defeaturize(generated_data)
        return self.featurizer.get_unique_smiles(generated_data)
