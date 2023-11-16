from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qsar",
    version="0.0.1",
    author="Tahiri Lab",
    author_email="nadia.tahiri@usherbrooke.ca",
    description="A Python package that offers robust predictive modeling using QSAR for evaluating the transfer of "
                "environmental contaminants in breast milk. It integrates multiple predictive models, provides "
                "synthetic data generation via GANs, and is tailored for researchers and health professionals.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tahiri-lab/QSAR",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "xgboost~=2.0.0",
        "jupyter~=1.0.0",
        "optuna~=3.3.0",
        "pandas~=2.1.1",
        "scikit-learn~=1.3.1",
        "seaborn~=0.12.2",
        "matplotlib~=3.8.0",
        "numpy~=1.25.0",
        "statsmodels~=0.14.0",
        "networkx~=3.1",
        "pydot~=1.4.2",
        "black~=23.9.1",
        "openpyxl~=3.1.2",
        "tensorflow~=2.14.0",
        "rdkit-pypi~=2022.9.5",
        "torch~=2.1.0",
        "tensorboard~=2.14.1",
        "tensorflow~=2.14.0",
        "keras~=2.14.0",
        "keras-tqdm~=2.0.1",
        "tqdm~=4.66.1",
        "ipython~=8.15.0",
        "scipy~=1.11.2",
        "dill~=0.3.7",
        "pymatgen~=2023.10.11",
        "deepchem~=2.7.1"
    ],
    python_requires='>=3.9.0',
)
