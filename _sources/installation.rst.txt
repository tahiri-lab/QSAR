.. _installation:

Installation
============

``qsarKit`` is a Python package that utilizes Quantitative Structure-Activity Relationship (QSAR) modeling for evaluating the transfer of environmental contaminants in breast milk. The package is designed for use by researchers and health professionals and integrates multiple predictive models. It provides features for synthetic data generation via Generative Adversarial Networks (GANs). Below are the instructions to install and set up the ``qsarKit`` environment.

Prerequisites
-------------

The package requires Miniconda (https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) to manage the environment dependencies. Please ensure that Miniconda is installed before proceeding with the installation of ``qsarKit``.

Installation Steps
------------------

Once Miniconda is installed, you can create and activate the ``qsarKit`` environment using the following commands:

.. code-block:: bash

    conda env create -f environment.yaml
    conda activate qsar_env

If you encounter any issues activating the environment, try sourcing the Conda script first and then retry activation:

.. code-block:: bash

    source ~/miniconda3/bin/activate qsar_env

or if you installed Anaconda instead of Miniconda:

.. code-block:: bash

    source ~/anaconda3/bin/activate qsar_env

Important Notes
---------------

- We currently support only Python 3.10 due to some dependencies that are not yet compatible with newer versions. We will update the package as soon as all dependencies support Python 3.11+.
- Ensure that the `environment.yaml` file is present in the root directory of the ``qsarKit`` package before creating the Conda environment.