.. _utils:

Utils
=====

This module contains the ``CrossValidator`` class, which facilitates the evaluation of QSAR models through cross-validation techniques.
The class provides methods for creating cross-validation folds, calculating cross-validation scores, evaluating model performance
across multiple metrics, and generating model predictions.

This module provides the ``Extractor`` class, designed for the efficient extraction and management of data from various CSV
files for use in data analysis and modeling, particularly in QSAR studies. The Extractor simplifies the process of loading,
accessing, and splitting datasets into features (X) and target (y) components, facilitating data handling and preprocessing
steps in QSAR modeling workflows.

The ``HyperParameterOptimizer`` class is designed to streamline the process of hyperparameter optimization for QSAR models
using the Optuna framework. This class is tailored to work seamlessly with models derived from the BaselineModel abstract class,
leveraging Optuna's efficient search capabilities to identify optimal hyperparameter settings based on specified evaluation criteria.

.. toctree::
   :maxdepth: 1

   utils/extractor
   utils/cross_validator
   utils/hyperparameter_optimizer