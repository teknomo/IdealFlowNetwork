import os
import sys
from sphinx.highlighting import lexers
from pygments.lexers import PythonLexer

# Assign the Python lexer to ipython3
lexers['ipython3'] = PythonLexer()

sys.path.insert(0, os.path.abspath('../../pkg'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Ideal Flow Network (IFN)'
copyright = '2024, Kardi Teknomo'
author = 'Kardi Teknomo'
release = '1.5.1'
highlight_language = 'python'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'nbsphinx'
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
