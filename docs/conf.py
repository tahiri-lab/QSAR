# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'qsar'
copyright = '2023, Mohammed Benabbassi, Nadia Tahiri'
author = 'Mohammed Benabbassi, Nadia Tahiri'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = []

html_context = {
    'display_github': True,
    'github_user': 'tahiri-lab',
    'github_repo': 'QSAR',
    'github_version': 'main/',
    'conf_py_path': 'docs/',
    'source_suffix': '.rst',
    'commit': False,
}