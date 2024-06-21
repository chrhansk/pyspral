#cython: language_level=3str

from libc.stdint cimport int64_t
from enum import Enum, auto

cimport numpy as np
import numpy as np
import scipy as sp

from .types cimport spral_matrix_type
from .types import MatrixType


cdef extern from "spral.h":
    int SPRAL_RANDOM_INITIAL_SEED

    struct spral_rb_read_options:
       int array_base
       bint add_diagonal
       float extra_space
       int lwr_upr_full
       int values

    struct spral_rb_write_options:
       int array_base
       char val_format[21]

    void spral_rb_default_read_options(spral_rb_read_options *options)
    void spral_rb_default_write_options(spral_rb_write_options *options)

    int spral_rb_peek(const char *filename, int *m, int *n, int64_t *nelt, int64_t *nvar,
                      int64_t *nval, spral_matrix_type *matrix_type, char *type_code,
                      char *title, char *identifier)

    int spral_rb_read_ptr32(const char *filename, void **handle,
                            spral_matrix_type *matrix_type, int *m, int *n, int **ptr,
                            int **row, double **val, const spral_rb_read_options *options,
                            char *title, char *identifier, int *state)

    int spral_rb_write_ptr32(const char *filename,
                             spral_matrix_type matrix_type,
                             int m, int n, const int *ptr, const int *row, const double * val,
                             const spral_rb_write_options *options, const char *title,
                             const char *identifier)

    void spral_rb_free_handle(void **handle)


class LowerUpperValues(Enum):
    Lower = 1
    Upper = 2
    Full = 3


class ReadValues(Enum):
    Values = 0
    PatternOnly = 1
    ValuesOrRandSymm = 2
    ValuesOrRandSymmDiagDom = 3
    ValuesOrRandUnsymm = 4
    RandSymm = -2
    RandSymmDiagDom = -3
    RandUnsymm = -4


cdef class ReadOptions:
    cdef spral_rb_read_options options
    def __cinit__(self, **options):
        spral_rb_default_read_options(&self.options)

    def __init__(self, **options):
        for key, value in options.items():
            setattr(self, key, value)

    property array_base:
        def __get__(self): return self.options.array_base
        def __set__(self, bint value): self.options.array_base = value

    property add_diagonal:
        def __get__(self): return self.options.add_diagonal
        def __set__(self, bint value): self.options.add_diagonal = value

    property extra_space:
        def __get__(self): return self.options.extra_space
        def __set__(self, float value): self.options.extra_space = value

    property lwr_upr_full:
        def __get__(self): return LowerUpperValues(self.options.lwr_upr_full)
        def __set__(self, value): self.options.lwr_upr_full = value.value

    property values:
        def __get__(self): return ReadValues(self.options.values)
        def __set__(self, value): self.options.values = value.value


cdef class WriteOptions:
    cdef spral_rb_write_options options
    def __cinit__(self):
        spral_rb_default_write_options(&self.options)

    property array_base:
        def __get__(self): return self.options.array_base
        def __set__(self, value): self.options.array_base = value

    property val_format:
        def __get__(self): return self.options.val_format
        def __set__(self, value): self.options.val_format = value


class ExitCode(Enum):
    Success = 0
    FailedToOpen = -1
    InvalidFormat = -2
    IOError = -3
    InvalidDataType = -4
    ElementalFormat = -5
    InvalidMatrixType = -6
    InvalidOptionExtraSpace = -10
    InvalidOptionLwrUpr = -11
    InvalidOptionValues = -12
    AllocationFailure = -20


class ReadError(Exception):
    def __init__(self, filename, exit_code):
        self.filename = filename
        self.exit_code = exit_code

    def __str__(self):
        return "Error: reading {0}: {1}".format(self.filename,
                                                self.exit_code.name)


# class DataType(Enum):
#     Real = 'r'
#     Complex = 'c'
#     Integer = 'i'
#     Pattern = 'p'
#     AuxPattern = 'q'


# class MatrixType(Enum):
#     Symm = 's'
#     UnSymm = 'u'
#     Hermitian = 'h'
#     SkewSymm = 'z'
#     Rectangular = 'r'


# class MatrixEncoding(Enum):
#     CSC = 'a'
#     Elemental = 'e'


# class TypeCode:
#     def __init__(self,
#                  data_type,
#                  matrix_type,
#                  matrix_encoding):
#         self.data_type = data_type
#         self.matrix_type = matrix_type
#         self.matrix_encoding = matrix_encoding

#     def __str__(self):
#         d = self.data_type.name
#         m = self.matrix_type.name
#         e = self.matrix_encoding.name
#         return "TypeCode({0}, {1}, {2})".format(d, m, e)

#     @staticmethod
#     def from_string(name):
#         data_type = DataType(name[0])
#         matrix_type = MatrixType(name[1])
#         matrix_encoding = MatrixEncoding(name[2])
#         return TypeCode(data_type, matrix_type, matrix_encoding)


class MatrixInfo:
    def __init__(self,
                 m,
                 n,
                 nelt,
                 nvar,
                 nval,
                 matrix_type,
                 title,
                 identifier):
        self.m = m
        self.n = n
        self.shape = (m, n)
        self.nelt = nelt
        self.nvar = nvar
        self.nval = nval
        self.matrix_type = matrix_type
        self.title = title
        self.identifier = identifier


def _peek(str filename):
    cdef int m, n
    cdef int64_t nelt, nvar, nval
    cdef spral_matrix_type matrix_type
    cdef char type_code[4]
    cdef char title[73]
    cdef char identifier[9]
    cdef int return_code
    cdef bytes _filename = filename.encode('ascii')
    cdef const char* fname = _filename

    return_code = spral_rb_peek(fname,
                                &m,
                                &n,
                                &nelt,
                                &nvar,
                                &nval,
                                &matrix_type,
                                type_code,
                                title,
                                identifier)

    exit_code = ExitCode(return_code)

    if exit_code != ExitCode.Success:
        raise ReadError(filename, exit_code)

    return MatrixInfo(m,
                      n,
                      nelt,
                      nvar,
                      nval,
                      MatrixType(matrix_type),
                      (<bytes> title).decode('ascii'),
                      (<bytes> identifier).decode('ascii'))


def peek(filename):
    return _peek(str(filename))


cdef class Matrix:

    cdef dict __dict__

    def __init__(self, m, n, ptr, row, val, title, identifier, matrix_type):
        self.m = m
        self.n = n

        assert m >= 0
        assert n >= 0

        self.ptr = ptr
        self.row = row
        self.data = val

        assert self.ptr.shape == (self.n + 1,)

        assert self.nnz == ptr[-1]

        assert self.row.shape == (self.nnz,)
        assert self.data.shape == (self.nnz,)

        self.title = title
        self.identifier = identifier
        self.matrix_type = matrix_type

    def tocsc(self):
        ptr = np.copy(self.ptr)
        ptr[-1] = self.nnz
        return sp.sparse.csc_matrix((self.data, self.row, ptr),
                                    shape=self.shape)

    @property
    def nnz(self):
        (nnz,) = self.row.shape
        return nnz

    @property
    def shape(self):
        return (self.m, self.n)

    @staticmethod
    def fromcsc(matrix, title, identifier, matrix_type):

        if not sp.sparse.issparse(matrix):
            raise ValueError("Expected sparse matrix")

        m = matrix.shape[0]
        n = matrix.shape[1]

        indptr = np.copy(matrix.indptr)
        row = np.copy(matrix.indices)
        data = np.copy(matrix.data)

        return Matrix(m,
                      n,
                      indptr,
                      row,
                      data,
                      title,
                      identifier,
                      matrix_type)


def _read(str filename, ReadOptions options, int random_state):
    cdef int m, n
    cdef bytes _filename = filename.encode('ascii')
    cdef const char* fname = _filename
    cdef void* handle
    cdef spral_matrix_type matrix_type
    cdef int* ptr
    cdef int* row
    cdef double* val
    cdef char title[73]
    cdef char identifier[9]
    cdef int inform

    cdef int[:] row_view
    cdef int[:] ptr_view
    cdef double[:] data_view

    try:
        options.array_base = 0

        inform = spral_rb_read_ptr32(fname,
                                     &handle,
                                     &matrix_type,
                                     &m,
                                     &n,
                                     &ptr,
                                     &row,
                                     &val,
                                     &options.options,
                                     title,
                                     identifier,
                                     &random_state)

        code = ExitCode(inform)

        if code != ExitCode.Success:
            raise ReadError(filename, code)

        ptr_view = <int[:n + 1]> ptr

        nnz = ptr_view[-1]

        row_view = <int[:nnz]> row

        if val != NULL:
            data_view = <double[:nnz]> val
        else:
            data_view = None

        return Matrix(m,
                      n,
                      np.copy(ptr_view),
                      np.copy(row_view),
                      np.copy(data_view),
                      title=title.decode('ascii'),
                      identifier=identifier.decode('ascii'),
                      matrix_type=MatrixType(matrix_type))
    finally:
        spral_rb_free_handle(&handle)


def read(filename, random_state=None, **options):
    cdef ReadOptions read_options = ReadOptions(**options)
    cdef int rand_state = SPRAL_RANDOM_INITIAL_SEED

    if random_state is not None:
        rand_state = random_state

    return _read(str(filename), read_options, rand_state)


def _write(str filename, Matrix matrix, WriteOptions options):
    cdef int m, n
    cdef bytes _filename = filename.encode('ascii')
    cdef const char* fname = _filename
    cdef int[:] ptr = matrix.ptr
    cdef int[:] row = matrix.row
    cdef double[:] val = matrix.data
    cdef int inform
    cdef spral_matrix_type matrix_type_value = matrix.matrix_type.value

    m = matrix.m
    n = matrix.n

    options.array_base = 0

    if n < 0 or m < 0:
        raise ValueError("Negative dimensions")

    if matrix.ptr.shape != (n + 1,):
        raise ValueError("Invalid ptr shape")

    data_size = ptr[-1]

    if matrix.row.shape != (data_size,):
        raise ValueError("Invalid row shape")

    if matrix.data.shape != (data_size,):
        raise ValueError("Invalid data shape")

    inform = spral_rb_write_ptr32(fname,
                                  matrix_type_value,
                                  m,
                                  n,
                                  &ptr[0],
                                  &row[0],
                                  &val[0],
                                  &options.options,
                                  matrix.title.encode('ascii'),
                                  matrix.identifier.encode('ascii'))

    code = ExitCode(inform)

    if code != ExitCode.Success:
        raise ReadError(filename, code)



def write(filename, matrix, **options):
    cdef WriteOptions write_options = WriteOptions(**options)

    return _write(str(filename), matrix, write_options)
