# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MockWalkers"
copyright = "2023, Alouette, David, Jimoh and Matias"
author = "Alouette, David, Jimoh and Matias"
release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax"
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "logo.svg",
    "dark_logo": "logo_dark.svg",
    "light_css_variables": {
        "font-stack": "Inria Sans, sans-serif",
        "font-stack--monospace": "Courier, monospace",
    },
    "dark_css_variables": {
        "font-stack": "Inria Sans, sans-serif",
        "font-stack--monospace": "Courier, monospace",
    },
}
html_css_files = [
    "fonts.css",
]
