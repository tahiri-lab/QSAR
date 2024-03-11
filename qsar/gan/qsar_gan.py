"""
This script is used to train a Generative Adversarial Network (GAN) model for generating molecules.
It uses the DeepChem and TensorFlow libraries for the GAN model and the QsarGanFeaturizer for featurizing the molecules.
The QsarGan class is responsible for the training and prediction process.
"""

import deepchem as dc
import numpy as np
from deepchem.models.molgan import BasicMolGANModel
from deepchem.models.optimizers import ExponentialDecay
from tensorflow import one_hot

from qsar.gan.gan_featurizer import QsarGanFeaturizer


class QsarGan:
    """
    A class that trains a Generative Adversarial Network (GAN) model for generating SMILES of synthetic molecules.
    """

    def __init__(
        self,
        learning_rate: ExponentialDecay,
        featurizer: QsarGanFeaturizer,
        edges: int = 5,
        nodes: int = 5,
        embedding_dim: int = 10,
        dropout_rate: float = 0.0,
        **kwargs
    ):

        self.featurizer = featurizer
        self.gan = BasicMolGANModel(
            learning_rate=learning_rate,
            edges=edges,
            vertices=self.featurizer.max_atom_count,
            nodes=nodes,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            **kwargs
        )

    def _iterbatches(self, epochs, features):
        """
        Yields batches of adjacency and node tensors for training the GAN model.

        :param epochs: the number of epochs for training the GAN model
        :type epochs: int
        :param features: the features used for training the GAN model
        :type features: np.ndarray
        """
        dataset = dc.data.NumpyDataset(
            [x.adjacency_matrix for x in features], [x.node_features for x in features]
        )
        for _ in range(epochs):
            for batch in dataset.iterbatches(
                batch_size=self.gan.batch_size, pad_batches=True
            ):
                adjacency_tensor = one_hot(batch[0], self.gan.edges)
                node_tensor = one_hot(batch[1], self.gan.nodes)
                yield {
                    self.gan.data_inputs[0]: adjacency_tensor,
                    self.gan.data_inputs[1]: node_tensor,
                }

    def fit_predict(
        self,
        features: np.ndarray,
        epochs=32,
        generator_steps=0.2,
        checkpoint_interval=5000,
        number_to_generate=10000,
    ) -> list:
        """
        Trains the GAN model and generates new molecules.

        :param features: the features used for training the GAN model
        :type features: np.ndarray
        :param epochs: the number of epochs for training the GAN model, defaults to 32
        :type epochs: int, optional
        :param generator_steps: the number of generator steps in the GAN model, defaults to 0.2
        :type generator_steps: float, optional
        :param checkpoint_interval: the interval for saving checkpoints in the GAN model, defaults to 5000
        :type checkpoint_interval: int, optional
        :param number_to_generate: the number of molecules to generate, defaults to 10000
        :type number_to_generate: int, optional
        :return: a list of unique SMILES strings representing the generated molecules
        :rtype: list
        """
        self.gan.fit_gan(
            self._iterbatches(epochs, features),
            generator_steps=generator_steps,
            checkpoint_interval=checkpoint_interval,
        )

        generated_data = self.gan.predict_gan_generator(number_to_generate)

        generated_data = self.featurizer.defeaturize(generated_data)
        return self.featurizer.get_unique_smiles(generated_data)
