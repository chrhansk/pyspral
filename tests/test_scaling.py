import scipy as sp
import numpy as np

import pytest

from pyspral.scaling import (scaling_auction,
                             scaling_hungarian,
                             scaling_equilib,
                             scaling,
                             ScalingError,
                             ScalingMethod)


@pytest.fixture
def symm_mat():
    mat = np.array([[ 2,  1,  0,  0,  0 ],
                    [ 1,  4,  1,  0,  1, ],
                    [ 0,  1,  3,  2,  0 ],
                    [ 0,  0,  2, -1,  0 ],
                    [ 0,  1,  0,  0,  2 ],],
                   dtype=float)

    return sp.sparse.csc_matrix(mat)


@pytest.fixture
def rect_mat():
    mat = np.array([[ 2,  1,  0,  0,  0 ],
                    [ 1,  4,  1,  0,  1, ],
                    [ 0,  1,  3,  2,  0 ]],
                   dtype=float)

    return sp.sparse.csc_matrix(mat)


@pytest.fixture
def singular_mat():
    mat = np.array([[ 2,  1,  0,  0,  0 ],
                    [ 1,  4,  1,  0,  1, ],
                    [ 0,  0,  0,  0,  0 ]],
                   dtype=float)

    return sp.sparse.csc_matrix(mat)


@pytest.fixture
def singular_symm_mat():
    mat = np.array([[ 2,  1,  0],
                    [ 1,  4,  0],
                    [ 0,  0,  0]],
                   dtype=float)

    return sp.sparse.csc_matrix(mat)


def check_dims_symm(symm_mat, result):
    (m, n) = symm_mat.shape
    assert result.scaling.shape == (n,)

    if result.matching is not None:
        assert result.matching.shape == (n,)


def check_dims_unsymm(unsymm_mat, result):
    (m, n) = unsymm_mat.shape
    assert result.row_scaling.shape == (m,)
    assert result.col_scaling.shape == (n,)

    if result.matching is not None:
        assert result.matching.shape == (n,)


def test_scaling_auction_symm(symm_mat):
    result = scaling_auction(symm_mat, symmetric=True)
    check_dims_symm(symm_mat, result)


def test_scaling_auction_options(symm_mat):
    result = scaling_auction(symm_mat, symmetric=True, max_iterations=10)
    check_dims_symm(symm_mat, result)


def test_scaling_auction_invalid_option(symm_mat):
    with pytest.raises(Exception):
        scaling_auction(symm_mat, symmetric=True, invalid_option=True)


def test_scaling_auction(rect_mat):
    result = scaling_auction(rect_mat)
    check_dims_unsymm(rect_mat, result)


def test_scaling_hungarian_symm(symm_mat):
    result = scaling_hungarian(symm_mat, symmetric=True)
    check_dims_symm(symm_mat, result)


def test_scaling_hungarian_options(symm_mat):
    result = scaling_hungarian(symm_mat, symmetric=True, scale_if_singular=True)
    check_dims_symm(symm_mat, result)


def test_scaling_hungarian_invalid_option(symm_mat):
    with pytest.raises(Exception):
        scaling_hungarian(symm_mat, symmetric=True, invalid_option=True)


def test_scaling_hungarian(rect_mat):
    result = scaling_hungarian(rect_mat)
    check_dims_unsymm(rect_mat, result)


@pytest.mark.skip(reason="Memory fault in spral (issue #200)")
def test_scaling_hungarian_singular(singular_mat, singular_symm_mat):
    with pytest.raises(ScalingError):
        scaling_hungarian(singular_mat, scale_if_singular=False)

    with pytest.raises(ScalingError):
        scaling_hungarian(singular_symm_mat, symmetric=True, scale_if_singular=False)

    with pytest.warns(RuntimeWarning):
        scaling_hungarian(singular_mat, scale_if_singular=True)

    with pytest.warns(RuntimeWarning):
        scaling_hungarian(singular_symm_mat, symmetric=True, scale_if_singular=True)


def test_scaling_equilib_symm(symm_mat):
    result = scaling_equilib(symm_mat, symmetric=True)
    check_dims_symm(symm_mat, result)


def test_scaling_equilib_options(symm_mat):
    result = scaling_equilib(symm_mat, symmetric=True, max_iterations=10, tol=1e-6)
    check_dims_symm(symm_mat, result)


def test_scaling_equilib_invalid_option(symm_mat):
    with pytest.raises(Exception):
        scaling_equilib(symm_mat, symmetric=True, invalid_option=True)


def test_scaling_equilib(rect_mat):
    result = scaling_equilib(rect_mat)
    check_dims_unsymm(rect_mat, result)


@pytest.mark.parametrize("scaling_method", list(ScalingMethod))
def test_scaling_symm(symm_mat, scaling_method):

    result = scaling(symm_mat, scaling_method, symmetric=True)
    check_dims_symm(symm_mat, result)


@pytest.mark.parametrize("scaling_method", list(ScalingMethod))
def test_scaling_unsymm(rect_mat, scaling_method):

    result = scaling(rect_mat, scaling_method, symmetric=False)
    check_dims_unsymm(rect_mat, result)
