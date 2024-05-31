#cython: language_level=3

from setuptools import Extension, setup
from Cython.Build import cythonize


extensions = [
    Extension(
        "pyspral.ssids",
        sources=["src/pyspral/ssids.pyx"],
        libraries=["spral"],
    )
]

setup(ext_modules=cythonize(extensions, gdb_debug=True))
