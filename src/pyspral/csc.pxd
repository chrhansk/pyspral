#cython: language_level=3str


cdef class _CSCMatrix:
    """
    CSC matrix as specified by SPRAL (one-indexed, lower triangular part only)
    """
    cdef const int[:] rows
    cdef const int[:] cols
    cdef const double[:] data
    cdef object shape
