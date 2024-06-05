import scipy as sp
import numpy as np

import pytest

from pyspral.ssids import analyze, solve, Job, PrintLevel, Ordering


@pytest.fixture
def indef_mat():
    mat = np.array([[ 2,  1,  0,  0,  0 ],
                    [ 1,  4,  1,  0,  1, ],
                    [ 0,  1,  3,  2,  0 ],
                    [ 0,  0,  2, -1,  0 ],
                    [ 0,  1,  0,  0,  2 ],],
                   dtype=float)

    return sp.sparse.csc_matrix(mat)


@pytest.fixture
def indef_rhs():
    return np.array([4.0, 17.0, 19.0, 2.0, 12.0])


def test_solve_indef(indef_mat, indef_rhs):
    mat = indef_mat

    symbolic_factor = analyze(mat, check=True)
    numeric_factor = symbolic_factor.factor(posdef=False)

    b = indef_rhs
    b_rhs = np.copy(b)

    sol = numeric_factor.solve(b_rhs, inplace=True)
    assert (sol == b_rhs).all()
    assert np.allclose(mat.dot(sol) - b, 0.)

    b_rhs = np.copy(b)
    sol = numeric_factor.solve(b_rhs, inplace=False)

    assert (b == b_rhs).all()
    assert np.allclose(mat.dot(sol) - b, 0.)


def test_scale(indef_mat, indef_rhs):
    mat = indef_mat
    scale = np.array([1., 2., 1., 2., 1.], dtype=float)

    symbolic_factor = analyze(mat, check=True)
    numeric_factor = symbolic_factor.factor(posdef=False, scale=scale)

    b = indef_rhs
    x = numeric_factor.solve(b, inplace=False)
    assert np.allclose(mat.dot(x) - b, 0.)


def test_options(indef_mat):
    mat = indef_mat
    analyze(mat,
            check=True,
            print_level=PrintLevel.BasicDiagnostic)


def test_order(indef_mat, indef_rhs):
    mat = indef_mat

    (_, n) = mat.shape

    order = np.arange(start=1, stop=(n + 1), dtype=np.intc)

    print(order)

    symbolic_factor = analyze(mat, check=True, order=order, ordering=Ordering.UserSupplied)
    numeric_factor = symbolic_factor.factor(posdef=False)

    b = indef_rhs
    x = numeric_factor.solve(b, inplace=False)
    assert np.allclose(mat.dot(x) - b, 0.)


def test_solve(indef_mat, indef_rhs):
    sol = solve(indef_mat, indef_rhs, posdef=False)
    assert np.allclose(indef_mat.dot(sol) - indef_rhs, 0.)


def test_solve_indef_multi(indef_mat):
    mat = indef_mat
    rhs = indef_rhs

    symbolic_factor = analyze(mat, check=True)
    numeric_factor = symbolic_factor.factor(posdef=False)

    (_, n) = mat.shape

    num_sols = 2

    rhs = np.eye(n)[:num_sols,:]
    sols_rhs = np.copy(rhs)

    sols = numeric_factor.solve(sols_rhs, inplace=True)

    for i in range(num_sols):
        assert np.allclose(mat.dot(sols[i, :]) - rhs[i, :], 0.)

    sols_rhs = np.copy(rhs)
    sols = numeric_factor.solve(sols_rhs, inplace=False)

    assert (rhs == sols_rhs).all()

    for i in range(num_sols):
        assert np.allclose(mat.dot(sols[i, :]) - rhs[i, :], 0.)

    rhs = np.eye(n + 1)[:num_sols, :n]

    sols_rhs = np.copy(rhs)
    sols = numeric_factor.solve(sols_rhs, inplace=False)

    for i in range(num_sols):
        assert np.allclose(mat.dot(sols[i, :]) - rhs[i, :], 0.)


def test_enquire(indef_mat):
    mat = indef_mat

    symbolic_factor = analyze(mat, check=True)
    numeric_factor = symbolic_factor.factor(posdef=False)

    (piv_order, d) = numeric_factor.enquire()

    (_, n) = mat.shape

    abs_piv_order = np.abs(piv_order)

    assert (abs_piv_order <= n).all()
    assert (abs_piv_order >= 1).all()


def test_inertia(indef_mat):
    mat = indef_mat

    eigvals = np.linalg.eigvalsh(mat.toarray())
    num_neg_evals = np.sum(eigvals < 0)

    symbolic_factor = analyze(mat, check=True)
    numeric_factor = symbolic_factor.factor(posdef=False)

    assert numeric_factor.inform.num_neg == num_neg_evals


def test_diag(indef_mat):
    mat = indef_mat

    symbolic_factor = analyze(mat, check=True)
    numeric_factor = symbolic_factor.factor(posdef=False)

    (_, n) = mat.shape

    rhs = np.eye(n)

    sol = numeric_factor.solve(rhs, job=Job.Dxb)

    assert np.allclose(sol, numeric_factor.D().toarray())
