# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
print("Autodoc imports successful")
project = 'SOLFD'
copyright = '2025, Jessica Monahan'
author = 'Jessica Monahan'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = [
    'src/j_testing/**',
    'src/functions/SOLWEIGpython/UTIL/clearnessindex_2013b.py',
    'src/functions/SOLWEIGpython/UTIL/shadowingfunction_wallheight_13.py',
    'src/functions/SOLWEIGpython/UTIL/shadowingfunction_wallheight_23.py',
    'src/functions/SOLWEIGpython/UTIL/sun_position.py',
    'src/functions/SOLWEIGpython/UTIL/sun_distance.py',
    'src/functions/SOLWEIGpython/cylindric_wedge.py',
    'src/functions/SOLWEIGpython/gvf_2018a.py',
    'src/functions/SOLWEIGpython/Kside_veg_v2022a.py',
    'src/functions/SOLWEIGpython/Lcyl_v2022a.py',
    'src/functions/SOLWEIGpython/Lside_veg_v2022a.py',
    'src/functions/SOLWEIGpython/patch_characteristics.py',
    'src/functions/SOLWEIGpython/PET_calculations.py',
    'src/functions/SOLWEIGpython/Solweig_2022a_calc_forprocessing.py',
    'src/functions/SOLWEIGpython/sunonsurface_2018a.py',
    'src/functions/SOLWEIGpython/Tgmaps_v1.py',
    'src/functions/svf_functions_original.py'
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']