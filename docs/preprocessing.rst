.. _preprocessing:

Preprocessing
=============

This module provides the FeatureSelector class designed for feature selection and preprocessing in QSAR modeling.
It includes a series of methods for normalizing data, removing features with low variance, handling multicollinearity,
and managing highly correlated features. The class aims to streamline the preprocessing steps necessary for effective
modeling by providing an integrated approach to handling various data preprocessing tasks such as scaling, variance analysis,
and correlation analysis.

This module offers custom transformer classes for feature selection in the context of QSAR modeling.
It includes transformers for removing low-variance features and high-correlation features from datasets,
as well as a composite preprocessing pipeline that integrates these transformations in a streamlined process.

.. toctree::
   :maxdepth: 1

   preprocessing/feature_selector
   preprocessing/custom_preprocessing