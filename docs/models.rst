.. _models:

Models
======

This module defines the BaselineModel, an abstract base class for constructing Quantitative Structure-Activity
Relationship (QSAR) models within a machine learning framework. The class is designed to streamline the integration
of different machine learning algorithms into QSAR studies by providing a uniform interface and methodology for
model construction, training, prediction, and hyperparameter optimization.

Child classes inheriting from ``BaselineModel`` are required to implement the model-specific logic within the defined
abstract methods. This structure promotes a clear separation between the generic model workflow and the specific
implementation details of different QSAR modeling techniques.

The module also includes several concrete implementations of the ``BaselineModel`` class, each corresponding to a
different machine learning algorithm. These classes provide a starting point for constructing QSAR models using
popular machine learning libraries that use regression algorithms and ensemble methods.

.. toctree::
   :maxdepth: 1

   models/base_model
   models/catboost_model
   models/elasticnet_model
   models/lasso_model
   models/random_forest_model
   models/ridge_model
   models/xgboost_model