.. _documentation:

qsarKit Documentation
=====================

This page provides guidelines on how to build, modify, and update the documentation for the ``qsarKit`` project.

Building the Documentation
--------------------------

The ``qsarKit`` documentation is generated using Sphinx, a tool that converts reStructuredText files into HTML websites and other formats. Here's how to build the documentation:

1. Ensure that Sphinx and other required packages are installed:

   .. code-block:: bash

       conda activate qsar_env
       pip install sphinx sphinx_rtd_theme

2. Navigate to the `docs` directory from the root of the project:

   .. code-block:: bash

       cd docs

3. Build the documentation using the `make` command on Linux or Mac, or `make.bat` on Windows:

   .. code-block:: bash

       # For Linux or Mac
       make html

       # For Windows
       .\make.bat html

The generated HTML files will be located in the `docs/build/html` directory. You can open the `index.html` file in your browser to view the documentation.

Modifying the Documentation
---------------------------

The ``qsarKit`` documentation is written in reStructuredText (.rst) format. To modify the documentation:

1. Navigate to the `docs/source` directory, where you will find the `.rst` files.

2. Open the `.rst` file you wish to modify in your text editor.

3. Make your changes to the text. Refer to the reStructuredText Primer for syntax guidelines: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

4. Save your changes and rebuild the documentation (as described in the "Building the Documentation" section) to view your updates.

Adding New Pages
----------------

To add a new page to the documentation:

1. Create a new `.rst` file in the `docs/source` directory.

2. Write your content in the file using reStructuredText format.

3. Add the file to the table of contents by including it in the appropriate `toctree` directive within an existing `.rst` file, typically `index.rst` or a related section file.

4. Rebuild the documentation to incorporate the new page.

Updating the Documentation
--------------------------

It is important to keep the documentation updated to reflect changes in the project. To update the documentation:

1. Follow the steps in the "Modifying the Documentation" section to edit existing pages.

2. If new features have been added to the project, consider adding new pages following the instructions in the "Adding New Pages" section.

3. Ensure that all changes are thoroughly reviewed for accuracy and completeness.

4. After updating the content, rebuild the documentation and verify that all changes are correctly reflected and that there are no broken links.

Contributing to the Documentation
---------------------------------

Contributions to improve the documentation are always welcome. To contribute:

1. Fork the repository on GitHub.

2. Clone your fork and create a new branch for your changes.

3. Make your changes, commit them, and push them to your fork.

4. Open a pull request against the original repository with a clear description of the improvements.

Remember to follow the project's contribution guidelines and code of conduct. Your contributions will be reviewed and merged if they meet the project's standards.

