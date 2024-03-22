.. _contributing:

Open source code repository
===========================

All code for QSAR Kit is freely available at `<https://github.com/tahiri-lab/qsarkit>`_.

Developer tools
---------------

For developers interested in contributing to ``qsarKit``, here's how you can set up your development environment:

Setting Up and Running Unit Tests
---------------------------------

``qsarKit`` uses pytest for unit testing. To set up and run unit tests:

1. Navigate to the project root directory.
2. Run the tests:

   .. code-block:: bash

      pytest tests/

To generate a coverage report:

   .. code-block:: bash

      pytest --cov=qsar tests/

Linting and Code Formatting
---------------------------

We use flake8, pylint, black, and isort for linting and formatting. To run them:

1. Lint with flake8:

   .. code-block:: bash

      flake8 qsar/ tests/

2. Check code with pylint:

   .. code-block:: bash

      pylint qsar/ tests/

3. Format code with black:

   .. code-block:: bash

      black qsar/ tests/

4. Sort imports with isort:

   .. code-block:: bash

      isort qsar/ tests/

CI/CD Pipeline with GitHub Actions
----------------------------------

The CI/CD pipeline for ``qsarKit`` is defined in `.github/workflows/sphinx.yaml` and includes the following jobs:

- **Lint and Static Analysis**: Checks the code for stylistic errors and potential bugs.
- **Unit Tests and Coverage**: Runs unit tests and reports on code coverage.
- **Security Scan**: Checks the project dependencies for known security vulnerabilities.
- **Build and Deploy Docs**: Builds the Sphinx documentation and deploys it to GitHub Pages.

These jobs ensure that the codebase maintains high quality and that the documentation is always up to date with the latest changes.