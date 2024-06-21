#cython: language_level=3

from enum import Enum


class MatrixType(Enum):
    Unspecified = 0
    RealRect = 1
    ComplexRect = -1
    RealUnsymm = 2
    ComplexUnsymm = -2
    RealSymmPsdef = 3
    ComplexHermPsdef = -3
    RealSymmIndef = 4
    ComplexHermIndef = -4
    ComplexSymm = -5
    RealSkew = 6
    ComplexSkew = -6
