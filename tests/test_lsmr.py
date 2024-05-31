import scipy as sp
import numpy as np

import pytest

from pyspral.lsmr import solve


@pytest.fixture()
def mat():
    _mat = np.array([[ 1.0, 0.0, -1.0 ],
                     [ 0.0, 2.0,  0.0 ],
                     [ 2.0, 0.0,  2.0 ],
                     [ 5.0, 3.0, -2.0 ],
                     [ 0.0, 0.0,  6.0 ]])

    return sp.sparse.linalg.aslinearoperator(np.copy(_mat))


@pytest.fixture()
def rhs():
    return np.array([1.0, 1.0, 1.0, 1.0, 1.0])


def test_solve(mat, rhs):
    result = solve(mat, rhs)

    assert result.success


def test_solve_damp(mat, rhs):
    result = solve(mat, rhs, lamb=1.)

    assert result.success


def test_solve_options(mat, rhs):
    result = solve(mat, rhs, itnlim=1)

    assert not result.success
    assert result.status == "iter_limit"


def test_solve_precond(mat, rhs):
    (_, n) = mat.shape
    precond = sp.sparse.linalg.aslinearoperator(np.eye(n))

    result = solve(mat, rhs, P=precond)

    assert result.success


def test_simple_callback(mat, rhs):

    def callback(x, iteration):
        return False

    result = solve(mat,
                   rhs,
                   callback=callback,
                   ctest="manual_simple")

    assert result.status == "manual_abort"


def test_callback(mat, rhs):

    all_norms = []

    def callback(x, iteration, norms):
        all_norms.append(norms["normAPr"])
        return True

    result = solve(mat,
                   rhs,
                   callback=callback,
                   ctest="manual_norm")

    assert len(all_norms) > 0
    assert np.isclose(all_norms[-1], 0.)
