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
        "catboost~=1.2.2",
        "deepchem~=2.7.2.dev20231110055545",
        "jax~=0.4.20",
        "keras~=2.12.0",
        "matplotlib~=3.8.1",
        "numpy~=1.23.5",
        "optuna~=3.4.0",
        "pandas~=2.1.3",
        "protobuf~=4.25.1",
        "rdkit~=2023.9.1",
        "scikit-learn~=1.3.2",
        "scipy~=1.11.3",
        "seaborn~=0.13.0",
        "statsmodels~=0.14.0",
        "torch~=2.0.0",
        "tqdm~=4.66.1",
        "xgboost~=2.0.2",
        "tensorflow~=2.12.0",
        "ipython~=8.21.0",
        "sphinx-rtd-theme~=2.0.0",
        "PyYAML~=6.0.1",
        "flake8~=7.0.0",
        "pytest~=8.0.0",
        "pytest-cov~=4.1.0",
        "pylint~=3.1.0",
        "black~=24.2.0",
        "isort~=5.13.2",
    ],
    python_requires='==3.10.*',
)
