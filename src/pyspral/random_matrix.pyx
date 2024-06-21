#cython: language_level=3str

from enum import Flag, Enum
from libc.stdint cimport int64_t

cimport numpy as np
import numpy as np
import scipy as sp

from .types cimport spral_matrix_type
from .types import MatrixType


cdef extern from "spral.h":
    int SPRAL_RANDOM_INITIAL_SEED

    int spral_random_matrix_generate(int *state,
                                     spral_matrix_type matrix_type,
                                     int m,
                                     int n,
                                     int nnz,
                                     int *ptr,
                                     int *row,
                                     double *val,
                                     int flags)

    int spral_random_matrix_generate_long(int *state,
                                          spral_matrix_type matrix_type,
                                          int m,
                                          int n,
                                          int64_t nnz,
                                          int64_t *ptr,
                                          int *row,
                                          double *val,
                                          int flags)



class MatrixFlags(Flag):
    ZERO = 0
    FINDEX = 1
    NONSINGULAR = 2
    SORTED = 4


class StatusCode(Enum):
    Success = 0
    AllocationError = -1
    InvalidMatrixType = -2
    NonPositiveValue = -3
    InconsistentDimensions = -4
    InsuffiicentNNZ = -5

    def success(self):
        return self == StatusCode.Success


class Error(Exception):
    def __init__(self, code):
        self.code = code

    def __str__(self):
        return "Failed to generate random matrix: {0}".format(self.code)


def random_matrix_generate(m,
                           n,
                           nnz,
                           matrix_type,
                           bint nonsingular=False,
                           random_state=None):

    if nnz <= 0:
        raise ValueError("Invalid number of nonzero values provided")

    if n <= 0 or m <= 0:
        raise ValueError("Invalid dimensions provided")

    cdef int ptr_size = n + 1
    cdef int[:] ptr = np.empty((ptr_size,), dtype=np.intc)
    cdef int[:] row = np.empty((nnz,), dtype=np.intc)
    cdef double[:] val = np.empty((nnz,), dtype=float)
    cdef int result
    cdef int flag
    cdef spral_matrix_type mat_type = matrix_type.value
    cdef int rand_state = SPRAL_RANDOM_INITIAL_SEED

    if random_state is not None:
        rand_state = random_state

    flags = MatrixFlags.SORTED

    if nonsingular:
        flags = flags | MatrixFlags.NONSINGULAR

    flag = flags.value

    result = spral_random_matrix_generate(&rand_state,
                                          mat_type,
                                          m,
                                          n,
                                          nnz,
                                          &ptr[0],
                                          &row[0],
                                          &val[0],
                                          flag)

    code = StatusCode(result)

    if not code.success():
        raise Error(code)

    return sp.sparse.csc_matrix((np.asarray(val),
                                 np.asarray(row),
                                 np.asarray(ptr)),
                                shape=(m, n))
