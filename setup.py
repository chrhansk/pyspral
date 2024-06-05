#cython: language_level=3

from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy as np


extensions = [
    Extension(
        "pyspral.ssids",
        sources=["src/pyspral/ssids.pyx"],
        libraries=["spral"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "pyspral.lsmr",
        sources=["src/pyspral/lsmr.pyx"],
        libraries=["spral"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "pyspral.ssmfe",
        sources=["src/pyspral/ssmfe.pyx"],
        libraries=["spral"],
        include_dirs=[np.get_include()]
    )
]

setup(ext_modules=cythonize(extensions, gdb_debug=True))
