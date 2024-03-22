.. _quickstart:

Quickstart Guide for qsarKit
============================

Welcome to the Quickstart Guide for ``qsarKit``! This guide aims to help new users get started with ``qsarKit`` by providing a brief overview, detailed installation instructions, and a simple example to demonstrate basic usage.

What is qsarKit?
----------------

``qsarKit`` is a Python package designed for robust predictive modeling using Quantitative Structure-Activity Relationship (QSAR) analysis. It is tailored for researchers and health professionals working with environmental contaminants in breast milk. Developed by Professor Nadia Tahiri's team, ``qsarKit`` integrates multiple predictive models and offers tools for synthetic data generation using Generative Adversarial Networks (GANs).

Basic Usage
-----------

Once you have installed and activated your ``qsarKit`` environment, you are ready to use the package. ``qsarKit`` offers flexible ways to utilize its functionalities: you can either run the package as a complete pipeline or use its individual functionalities by importing each module.

Running the Complete Pipeline
------------------------------

To execute the entire pipeline, including preprocessing, data augmentation, model training/optimization, and prediction, you can use a single command. This approach is useful for users who wish to apply the standard workflow with minimal setup:

.. code-block:: bash

    python main.py --config ridge_model.yaml --output results/

In this example, ``ridge_model.yaml`` should contain all the necessary configurations for each step of the pipeline. The results of each step will be saved in the ``results/`` directory.

The advantages of running the full pipeline include simplicity and the assurance that all steps are executed in the correct order. However, this method provides less flexibility compared to running each step individually.

Using Package Functionalities Individually
-------------------------------------------

In addition to running the complete ``qsarKit`` pipeline, users can leverage the package's modular design to use individual functionalities. This approach provides greater flexibility and allows integration with other data processing or analysis workflows.

Data Extraction and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by extracting and preprocessing your dataset. Use the ``Extractor`` and ``PreprocessingPipeline`` classes for these tasks:

.. code-block:: python

    from qsar.utils.extractor import Extractor
    from qsar.preprocessing.custom_preprocessing import PreprocessingPipeline
    import pandas as pd

    # Configuration for data extraction
    datasets_config = {
        'full_train': 'path/to/full_train_dataset.csv',
        'full_test': 'path/to/full_test_dataset.csv'
    }
    target_column = 'desired_target_column'

    # Extract data
    extractor = Extractor(datasets_config)
    df_full = pd.concat([extractor.get_df("full_train"), extractor.get_df("full_test")])

    # Preprocess data
    preprocessing = PreprocessingPipeline(target=target_column, variance_threshold=0.0, cols_to_ignore=[], verbose=False, threshold=0.9)
    pipeline = preprocessing.get_pipeline()
    df_processed = pipeline.fit_transform(df_full)

Cross-validation and Model Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform cross-validation and evaluate model performance using the ``CrossValidator`` class:

.. code-block:: python

    from qsar.utils.cross_validator import CrossValidator
    from qsar.utils.visualizer import Visualizer

    # Setup cross-validation and visualization
    cross_validator = CrossValidator(df_processed)
    visualizer = Visualizer()

    # Create cross-validation folds
    X_list, y_list, df, y, n_folds = cross_validator.create_cv_folds()
    visualizer.display_cv_folds(df, y, n_folds)

Model Training and Hyperparameter Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamically load machine learning models, optimize their hyperparameters, and train them:

.. code-block:: python

    from qsar.utils.hyperparameter_optimizer import HyperParameterOptimizer
    from qsar.utils import get_class_from_path

    # Define model configurations
    models_config = [
        {'name': 'ridge', 'hyperparameters': {...}},
        # Add more models as needed
    ]

    # Dynamically load and optimize models
    for model_config in models_config:
        model_name = model_config['name']
        model_class = get_class_from_path("qsar.models." + model_name, model_name.capitalize() + "Model")
        model_instance = model_class()

        # Optimize model
        optimizer = HyperParameterOptimizer(model=model_instance, data=df_processed, direction='maximize', trials=100)
        study = optimizer.optimize()

        # Set best hyperparameters
        best_params = study.best_params
        model_instance.set_hyperparameters(**best_params)

        # Evaluate model performance
        R2, CV, custom_cv, Q2 = cross_validator.evaluate_model_performance(model_instance, X_list, y_list)
        visualizer.display_model_performance(model_name, R2, CV, custom_cv, Q2)

Remember, this is just a guideline. You should adapt the code examples to fit your specific datasets, models, and requirements. The ``qsarKit`` package is designed to be modular, offering flexibility for diverse QSAR modeling needs.


This approach allows you to customize each step of the pipeline according to your needs. You can modify the configurations, substitute modules, or integrate ``qsarKit``'s functionalities into larger systems.

For further examples and detailed instructions on how to use each module, refer to the tutorials included with the package. The tutorials provide comprehensive guides on each component of ``qsarKit``, helping you to understand and utilize the full potential of the package.

Further Resources
-----------------

- **Tutorials**: Explore the `tutorials/` directory at https://github.com/tahiri-lab/QSAR/tree/main/tutorials for detailed guides on using ``qsarKit``, including model training, data preprocessing, and synthetic data generation.
- **Documentation**: Visit the official ``qsarKit`` documentation at https://tahiri-lab.github.io/QSAR/ for comprehensive information on all features and functionalities.
- **Contact**: For additional support or feedback, please contact Professor Nadia Tahiri at Nadia.Tahiri@USherbrooke.ca.

Thank you for choosing ``qsarKit`` for your QSAR predictive modeling needs. We hope this guide helps you get started smoothly.