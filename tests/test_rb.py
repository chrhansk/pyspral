from importlib.resources import files
import tempfile
import os

import numpy as np
import scipy as sp
import pytest

from pyspral.rb import peek, read, write, Matrix, MatrixType, LowerUpperValues, ReadValues
import tests.resources


@pytest.fixture()
def mat():
    _mat = np.array([[ 1.0, 0.0, -1.0 ],
                     [ 0.0, 2.0,  0.0 ],
                     [ 2.0, 0.0,  2.0 ],
                     [ 5.0, 3.0, -2.0 ],
                     [ 0.0, 0.0,  6.0 ]])

    return sp.sparse.csc_matrix(np.copy(_mat))


@pytest.fixture
def rb_file_name():
    return files(tests.resources).joinpath("matrix.rb")


def test_peek(rb_file_name):
    result = peek(rb_file_name)

    assert result.shape == (5, 5)

    assert result.matrix_type == MatrixType.RealSymmIndef


def test_read(rb_file_name):
    mat = read(rb_file_name, lwr_upr_full=LowerUpperValues.Lower)

    assert mat.shape == (5, 5)

    mat_val = mat.tocsc()
    assert mat_val.shape == (5, 5)

    matcoo = mat_val.tocoo()

    assert (matcoo.row >= matcoo.col).all()


def check_symm(mat):
    mat_val = mat.tocsc()
    matcoo = mat_val.tocoo()

    row = matcoo.row
    col = matcoo.col
    data = matcoo.data

    entries = dict()

    for (i, j, d) in zip(row, col, data):
        entries[(i, j)] = d

    for (i, j, d) in zip(row, col, data):
        if i == j:
            continue

        assert (j, i) in entries

        assert entries[(i, j)] == entries[(j, i)]


def test_read_full(rb_file_name):
    mat = read(rb_file_name, lwr_upr_full=LowerUpperValues.Full)

    assert mat.shape == (5, 5)

    check_symm(mat)


def test_read_random(rb_file_name):
    mat = read(rb_file_name,
               random_state=42,
               lwr_upr_full=LowerUpperValues.Full,
               values=ReadValues.RandSymm)

    next_mat = read(rb_file_name,
                    random_state=42,
                    lwr_upr_full=LowerUpperValues.Full,
                    values=ReadValues.RandSymm)

    assert (mat.data == next_mat.data).all()


def test_write_mat(mat):
    title = "Test Matrix"
    identifier = "TestId"

    matrix = Matrix.fromcsc(mat,
                            title=title,
                            identifier=identifier,
                            matrix_type=MatrixType.RealRect)

    tmp_dir = None

    try:
        tmp_dir = tempfile.TemporaryDirectory()

        filename = os.path.join(tmp_dir.name, "matrix.rb")

        write(filename, matrix)

        other_matrix = read(filename)

        assert (other_matrix.tocsc() - mat).nnz == 0

        assert other_matrix.title == title
        assert other_matrix.identifier == identifier
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()
