import numpy as np
import pytest

from pyspral.random_matrix import random_matrix_generate, MatrixType, Error


def test_random_matrix():
    m = 10
    n = 3
    nnz = 12

    mat = random_matrix_generate(m,
                                 n,
                                 nnz,
                                 MatrixType.Unspecified)

    assert mat.shape == (m, n)
    assert mat.nnz == nnz


def test_invalid_dim():
    n = 3
    nnz = 0

    with pytest.raises(Exception):
        random_matrix_generate(n,
                               n,
                               nnz,
                               MatrixType.Unspecified)


def test_insufficient_nnz():
    n = 3
    nnz = 2

    with pytest.raises(Error):
        random_matrix_generate(n,
                               n,
                               nnz,
                               MatrixType.Unspecified,
                               nonsingular=True)


def test_random_matrix_symm():
    n = 3
    nnz = 5

    mat = random_matrix_generate(n,
                                 n,
                                 nnz,
                                 MatrixType.RealSymmIndef)

    mat = mat.toarray()

    upper_indices = np.triu_indices(n, 1)

    upper_vals = mat[upper_indices]

    assert (upper_vals == 0.).all()


@pytest.mark.skip("Not implemented")
def test_random_spd():
    n = 3
    nnz = 3

    mat = random_matrix_generate(n,
                                 n,
                                 nnz,
                                 MatrixType.RealSymmPsdef,
                                 nonsingular=True)

    mat = mat.toarray()

    mat_sym = .5*(mat + mat.T)

    np.linalg.cholesky(mat_sym)
