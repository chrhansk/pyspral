#cython: language_level=3str

from enum import Enum

cdef extern from "spral.h":
    enum spral_matrix_type:
        SPRAL_MATRIX_UNSPECIFIED = 0
        SPRAL_MATRIX_REAL_RECT = 1
        SPRAL_MATRIX_CPLX_RECT = -1
        SPRAL_MATRIX_REAL_UNSYM = 2
        SPRAL_MATRIX_CPLX_UNSYM = -2
        SPRAL_MATRIX_REAL_SYM_PSDEF = 3
        SPRAL_MATRIX_CPLX_HERM_PSDEF = -3
        SPRAL_MATRIX_REAL_SYM_INDEF = 4
        SPRAL_MATRIX_CPLX_HERM_INDEF = -4
        SPRAL_MATRIX_CPLX_SYM = -5
        SPRAL_MATRIX_REAL_SKEW = 6
        SPRAL_MATRIX_CPLX_SKEW = -6
