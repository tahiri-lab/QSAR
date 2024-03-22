.. _gan:

Gan
===

This module provides the ``QsarGanFeaturizer`` class, an extension of the ``MolGanFeaturizer`` from DeepChem,
tailored for Quantitative Structure-Activity Relationship (QSAR) applications with Generative Adversarial Networks (GAN).
It includes functionalities for processing SMILES strings into molecular representations suitable for GAN modeling.

This module also provides the ``QsarGan`` class, an extension of the MolGan from DeepChem, tailored for QSAR applications with GAN.
It includes functionalities for training and evaluating GAN models for QSAR.

Finally, this module provides the ``ExtractDescriptors`` class, which extracts molecular descriptors from SMILES strings using RDKit.

.. toctree::
   :maxdepth: 1

   gan/gan_featurizer
   gan/qsar_gan
   gan/extract_descriptors