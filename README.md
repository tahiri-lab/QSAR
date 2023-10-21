# QSAR

## Installation
[Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) is used to handle the environment dependencies.
Once miniconda is installed, the environment can be created and activated with the following commands:
`conda create --name <env_name>`
`conda activate <env_name>`
`conda install --file requirements.txt`


# Library structure
qsar/
│
├── qsar/               # Source code
│   ├── __init__.py
│   ├── models/             # Machine learning models
│   │   ├── __init__.py
│   │   ├── xgboost.py
│   │   ├── catboost.py
│   │   ├── ...
│   ├── gan/                # Generative adversarial networks
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   ├── ...
│   ├── preprocessing/      # Data preprocessing
│   │   ├── __init__.py
│   │   ├── feature_selector.py
│   │   ├── feature_extractor.py
│   │   ├── ...
│   ├── utils/              # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── ...
│   ├── visualization/      # Visualization functions
│   │   ├── __init__.py
│   │   ├── plot.py
│   │   ├── ...
│   ├── ...

├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_module1.py
│   ├── test_module2.py
│   ├── ...
│
├── docs/                    # Documentation (e.g., Sphinx)
│   ├── source/
│   ├── build/
│   ├── ...
│
├── tutorials/                # Example scripts or notebooks using the library
│   ├── example1.py
│   ├── example2.ipynb
│   ├── ...
│
├── setup.py                 # Setup script for building and packaging
├── README.md                # Project description and user guide
├── MANIFEST.in              # Include additional files in the package
├── requirements.txt         # List of dependencies
├── .gitignore
├── LICENSE


# App structure
myapp/
│
├── app/                     # Source code for the Hydra application
│   ├── __init__.py
│   ├── main.py              # Entry point, likely decorated with @hydra.main(...)
│   ├── training_module.py
│   ├── preprocessing_module.py
│   ├── ...
│
├── conf/                    # Hydra configuration files
│   ├── config.yaml          # Base configuration
│   ├── dataset/             # Dataset specific configurations
│   │   ├── dataset1.yaml
│   │   ├── dataset2.yaml
│   │   ├── ...
│   ├── model/               # Model specific configurations
│   │   ├── model1.yaml
│   │   ├── model2.yaml
│   │   ├── ...
│   ├── ...
│
├── outputs/                 # Default directory where Hydra stores outputs (logs, models, etc.)
│
├── logs/                    # Any other logs or outputs you may want to store
│
├── README.md                # Project description and user guide on how to run the application
├── requirements.txt         # List of dependencies, including your library
├── .gitignore
├── LICENSE

