#cython: language_level=3

from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy as np


modules = ["csc", "ssids", "lsmr", "ssmfe", "scaling"]


extensions = [
    Extension(
        "pyspral.{0}".format(module),
        sources=["src/pyspral/{0}.pyx".format(module)],
        libraries=["spral"],
        include_dirs=[np.get_include()]
    )
    for module in modules
]

setup(ext_modules=cythonize(extensions, gdb_debug=True))
