#cython: language_level=3str

cimport numpy as np

import numpy as np
import scipy as sp


def iter_spmatrix(matrix):
    if sp.sparse.isspmatrix_coo(matrix):
        for r, c, m in zip(matrix.row, matrix.col, matrix.data):
            yield r, c, m

    elif sp.sparse.isspmatrix_csc(matrix):
        for c in range(matrix.shape[1]):
            for ind in range(matrix.indptr[c], matrix.indptr[c+1]):
                yield matrix.indices[ind], c, matrix.data[ind]

    elif sp.sparse.isspmatrix_csr(matrix):
        for r in range(matrix.shape[0]):
            for ind in range(matrix.indptr[r], matrix.indptr[r+1]):
                yield r, matrix.indices[ind], matrix.data[ind]

    elif sp.sparse.isspmatrix_lil(matrix):
        for r in range(matrix.shape[0]):
            for c, d in zip(matrix.rows[r], matrix.data[r]):
                yield r, c, d
    else:
        raise NotImplementedError("Invalid matrix format")


cdef class _CSCMatrix:
    """
    CSC matrix as specified by SPRAL (one-indexed)
    """

    def __cinit__(self, mat, symmetric):
        (m, n) = mat.shape

        nnz = mat.getnnz()

        row_entries = [np.array([1], dtype=np.intc)] + [list() for _ in range(n)]
        data_entries = [list() for _ in range(n)]

        for r, c, v in iter_spmatrix(mat):
            if symmetric and (r < c):
                continue

            row_entries[c + 1].append(r + 1)
            data_entries[c].append(v)

        for c in range(n):
            row_entries[c + 1] = np.array(row_entries[c + 1],
                                          dtype=np.intc)

            data_entries[c] = np.array(data_entries[c],
                                       dtype=np.double)

            perm = np.argsort(row_entries[c + 1])
            row_entries[c + 1] = row_entries[c + 1][perm]
            data_entries[c] = data_entries[c][perm]

        self.rows = np.concatenate(row_entries[1:])
        self.cols = np.cumsum([row.size for row in row_entries], dtype=np.intc)
        self.data = np.concatenate(data_entries)
        self.shape = mat.shape
