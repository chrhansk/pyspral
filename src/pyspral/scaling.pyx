#cython: language_level=3str

import warnings
from enum import Enum

import numpy as np
import scipy as sp

from .csc cimport _CSCMatrix
from .csc import _CSCMatrix

cdef extern from "spral.h":
    struct spral_scaling_auction_options:
        int array_base
        int max_iterations
        int max_unchanged[3]
        float min_proportion[3]
        float eps_initial
        char unused[80]

    struct spral_scaling_auction_inform:
        int flag
        int stat
        int matched
        int iterations
        int unmatchable
        char unused[80]

    struct spral_scaling_equilib_options:
        int array_base
        int max_iterations
        float tol
        char unused[80]

    struct spral_scaling_equilib_inform:
        int flag
        int stat
        int iterations
        char unused[80]

    struct spral_scaling_hungarian_options:
        int array_base
        bint scale_if_singular
        char unused[80]

    struct spral_scaling_hungarian_inform:
        int flag
        int stat
        int matched
        char unused[80]

    void spral_scaling_auction_default_options(spral_scaling_auction_options *options)
    void spral_scaling_equilib_default_options(spral_scaling_equilib_options *options)
    void spral_scaling_hungarian_default_options(spral_scaling_hungarian_options *options)

    void spral_scaling_auction_sym(int n, const int *ptr, const int *row,
                                   const double *val, double *scaling, int *match,
                                   const spral_scaling_auction_options *options,
                                   spral_scaling_auction_inform *inform)

    void spral_scaling_equilib_sym(int n, const int *ptr, const int *row,
                                   const double *val, double *scaling,
                                   const spral_scaling_equilib_options *options,
                                   spral_scaling_equilib_inform *inform)

    void spral_scaling_hungarian_sym(int n, const int *ptr, const int *row,
                                     const double *val, double *scaling, int *match,
                                     const spral_scaling_hungarian_options *options,
                                     spral_scaling_hungarian_inform *inform)

    void spral_scaling_auction_unsym(int m, int n, const int *ptr,
                                     const int *row, const double *val, double *rscaling, double *cscaling,
                                     int *match, const spral_scaling_auction_options *options,
                                     spral_scaling_auction_inform *inform)

    void spral_scaling_equilib_unsym(int m, int n, const int *ptr,
                                     const int *row, const double *val, double *rscaling, double *cscaling,
                                     const spral_scaling_equilib_options *options,
                                     spral_scaling_equilib_inform *inform)

    void spral_scaling_hungarian_unsym(int m, int n, const int *ptr,
                                       const int *row, const double *val, double *rscaling, double *cscaling,
                                       int *match, const spral_scaling_hungarian_options *options,
                                       spral_scaling_hungarian_inform *inform)


class ExitCode(Enum):
    Success = 0
    AllocationError = -1
    RankDeficientError = -2
    RankDeficientWarning = 1

    def error(self):
        return self.value < 0

    def warning(self):
        return self.value > 0

    def success(self):
        return self.value == 0

    def __str__(self):
        return "{0} (exit code: {1})".format(code_messages[self], self.value)


code_messages = {
    ExitCode.Success: "Success",
    ExitCode.AllocationError: "Allocation error",
    ExitCode.RankDeficientError: "Rank deficient matrix error",
    ExitCode.RankDeficientWarning: "Rank deficient matrix warning"
}


class ScalingError(Exception):
    def __init__(self, error_code):
        self.error_code = error_code


cdef class AuctionOptions:
    cdef spral_scaling_auction_options options

    def __cinit__(self, **kwargs):
        spral_scaling_auction_default_options(&self.options)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    property array_base:
        def __get__(self): return self.options.array_base
        def __set__(self, value): self.options.array_base = value

    property max_iterations:
        def __get__(self): return self.options.max_iterations
        def __set__(self, value): self.options.max_iterations = value

    property max_unchanged:
        def __get__(self): return self.options.max_unchanged
        def __set__(self, value): self.options.max_unchanged = value

    property min_proportion:
        def __get__(self): return self.options.min_proportion
        def __set__(self, value): self.options.min_proportion = value


cdef class HungarianOptions:
    cdef spral_scaling_hungarian_options options

    def __cinit__(self, **kwargs):
        spral_scaling_hungarian_default_options(&self.options)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    property array_base:
        def __get__(self): return self.options.array_base
        def __set__(self, value): self.options.array_base = value

    property scale_if_singular:
        def __get__(self): return self.options.scale_if_singular
        def __set__(self, value): self.options.scale_if_singular = value


cdef class EquilibOptions:
    cdef spral_scaling_equilib_options options

    def __cinit__(self, **kwargs):
        spral_scaling_equilib_default_options(&self.options)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    property array_base:
        def __get__(self): return self.options.array_base
        def __set__(self, value): self.options.array_base = value

    property max_iterations:
        def __get__(self): return self.options.max_iterations
        def __set__(self, value): self.options.max_iterations = value

    property tol:
        def __get__(self): return self.options.tol
        def __set__(self, value): self.options.tol = value


def raise_error_from_flag(flag):
    exit_code = ExitCode(flag)
    if exit_code.error():
        raise ScalingError(exit_code)
    elif exit_code.warning():
        warnings.warn(code_messages[exit_code], RuntimeWarning)


class SymmetricScalingResult:
    def __init__(self, scaling, matching=None, **props):
        self.scaling = scaling
        self.matching = matching
        self.props = props

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self.props[name]


class UnsymmetricScalingResult:
    def __init__(self, row_scaling, col_scaling, matching=None, **props):
        self.row_scaling = row_scaling
        self.col_scaling = col_scaling
        self.matching = matching
        self.props = props

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self.props[name]


cdef _auction_props(spral_scaling_auction_inform* inform):
    return {
        "matched": inform.matched,
        "iterations": inform.iterations,
        "unmatchable": inform.unmatchable
    }


cdef _scaling_auction_sym(A: sp.sparse.spmatrix,
                          AuctionOptions options):
    cdef int m, n
    cdef spral_scaling_auction_inform inform
    cdef _CSCMatrix csc_mat
    cdef double[::1] scaling
    cdef int[::1] matching

    csc_mat = _CSCMatrix(A, symmetric=True)

    (m, n) = csc_mat.shape
    assert m == n, "Matrix must be square"

    options.array_base = 1

    scaling = np.zeros(n, dtype=np.float64)
    matching = np.zeros(n, dtype=np.intc)

    spral_scaling_auction_sym(n,
                              &csc_mat.cols[0],
                              &csc_mat.rows[0],
                              &csc_mat.data[0],
                              &scaling[0],
                              &matching[0],
                              &options.options,
                              &inform)

    raise_error_from_flag(inform.flag)

    return SymmetricScalingResult(np.asarray(scaling),
                                  np.asarray(matching),
                                  **_auction_props(&inform))


cdef _scaling_auction_unsym(A: sp.sparse.spmatrix,
                             AuctionOptions options):
    cdef int m, n
    cdef spral_scaling_auction_inform inform
    cdef _CSCMatrix csc_mat
    cdef double[::1] row_scaling
    cdef double[::1] col_scaling
    cdef int[::1] matching

    csc_mat = _CSCMatrix(A, symmetric=False)

    (m, n) = csc_mat.shape
    options.array_base = 1

    row_scaling = np.zeros(m, dtype=np.float64)
    col_scaling = np.zeros(n, dtype=np.float64)
    matching = np.zeros(n, dtype=np.intc)

    spral_scaling_auction_unsym(m,
                                n,
                                &csc_mat.cols[0],
                                &csc_mat.rows[0],
                                &csc_mat.data[0],
                                &row_scaling[0],
                                &col_scaling[0],
                                &matching[0],
                                &options.options,
                                &inform)

    raise_error_from_flag(inform.flag)

    return UnsymmetricScalingResult(np.asarray(row_scaling),
                                    np.asarray(col_scaling),
                                    np.asarray(matching),
                                    **_auction_props(&inform))


def scaling_auction(A: sp.sparse.spmatrix,
                    symmetric=False,
                    **options):
    cdef AuctionOptions opts = AuctionOptions(**options)

    assert sp.sparse.issparse(A)

    if symmetric:
        return _scaling_auction_sym(A, opts)
    else:
        return _scaling_auction_unsym(A, opts)


cdef _scaling_hungarian_sym(A: sp.sparse.spmatrix,
                            HungarianOptions options):
    cdef int m, n
    cdef spral_scaling_hungarian_inform inform
    cdef _CSCMatrix csc_mat
    cdef double[::1] scaling
    cdef int[::1] matching

    csc_mat = _CSCMatrix(A, symmetric=True)

    (m, n) = csc_mat.shape
    assert m == n, "Matrix must be square"

    options.array_base = 1

    scaling = np.zeros(n, dtype=np.float64)
    matching = np.zeros(n, dtype=np.intc)

    spral_scaling_hungarian_sym(n,
                                &csc_mat.cols[0],
                                &csc_mat.rows[0],
                                &csc_mat.data[0],
                                &scaling[0],
                                &matching[0],
                                &options.options,
                                &inform)

    raise_error_from_flag(inform.flag)

    return SymmetricScalingResult(np.asarray(scaling),
                                  np.asarray(matching),
                                  matched=inform.matched)


cdef _scaling_hungarian_unsym(A: sp.sparse.spmatrix,
                              HungarianOptions options):
    cdef int m, n
    cdef spral_scaling_hungarian_inform inform
    cdef _CSCMatrix csc_mat
    cdef double[::1] row_scaling
    cdef double[::1] col_scaling
    cdef int[::1] matching

    csc_mat = _CSCMatrix(A, symmetric=False)

    (m, n) = csc_mat.shape
    options.array_base = 1

    row_scaling = np.zeros(m, dtype=np.float64)
    col_scaling = np.zeros(n, dtype=np.float64)
    matching = np.zeros(n, dtype=np.intc)

    spral_scaling_hungarian_unsym(m,
                                  n,
                                  &csc_mat.cols[0],
                                  &csc_mat.rows[0],
                                  &csc_mat.data[0],
                                  &row_scaling[0],
                                  &col_scaling[0],
                                  &matching[0],
                                  &options.options,
                                  &inform)

    raise_error_from_flag(inform.flag)

    return UnsymmetricScalingResult(np.asarray(row_scaling),
                                    np.asarray(col_scaling),
                                    np.asarray(matching),
                                    matched=inform.matched)


def scaling_hungarian(A: sp.sparse.spmatrix,
                      symmetric=False,
                      **options):
    cdef HungarianOptions opts = HungarianOptions(**options)

    assert sp.sparse.issparse(A)

    if symmetric:
        return _scaling_hungarian_sym(A, opts)
    else:
        return _scaling_hungarian_unsym(A, opts)


cdef _scaling_equilib_sym(A: sp.sparse.spmatrix,
                          EquilibOptions options):
    cdef int m, n
    cdef spral_scaling_equilib_inform inform
    cdef _CSCMatrix csc_mat
    cdef double[::1] scaling

    csc_mat = _CSCMatrix(A, symmetric=True)

    (m, n) = csc_mat.shape
    assert m == n, "Matrix must be square"

    options.array_base = 1

    scaling = np.zeros(n, dtype=np.float64)

    spral_scaling_equilib_sym(n,
                              &csc_mat.cols[0],
                              &csc_mat.rows[0],
                              &csc_mat.data[0],
                              &scaling[0],
                              &options.options,
                              &inform)

    raise_error_from_flag(inform.flag)

    return SymmetricScalingResult(np.asarray(scaling),
                                  iterations=inform.iterations)


cdef _scaling_equilib_unsym(A: sp.sparse.spmatrix,
                            EquilibOptions options):
    cdef int m, n
    cdef spral_scaling_equilib_inform inform
    cdef _CSCMatrix csc_mat
    cdef double[::1] row_scaling
    cdef double[::1] col_scaling
    cdef int[::1] matching

    csc_mat = _CSCMatrix(A, symmetric=False)

    (m, n) = csc_mat.shape
    options.array_base = 1

    row_scaling = np.zeros(m, dtype=np.float64)
    col_scaling = np.zeros(n, dtype=np.float64)

    spral_scaling_equilib_unsym(m,
                                n,
                                &csc_mat.cols[0],
                                &csc_mat.rows[0],
                                &csc_mat.data[0],
                                &row_scaling[0],
                                &col_scaling[0],
                                &options.options,
                                &inform)

    raise_error_from_flag(inform.flag)

    return UnsymmetricScalingResult(np.asarray(row_scaling),
                                    np.asarray(col_scaling),
                                    iterations=inform.iterations)


def scaling_equilib(A: sp.sparse.spmatrix,
                    symmetric=False,
                    **options):
    cdef EquilibOptions opts = EquilibOptions(**options)

    assert sp.sparse.issparse(A)

    if symmetric:
        return _scaling_equilib_sym(A, opts)
    else:
        return _scaling_equilib_unsym(A, opts)


class ScalingMethod(Enum):
    Auction = 0
    Hungarian = 1
    Equilib = 2

    def __str__(self):
        return self.name


def scaling(A: sp.sparse.spmatrix,
            method: ScalingMethod,
            symmetric=False,
            **options):
    if method == ScalingMethod.Auction:
        return scaling_auction(A, symmetric, **options)
    elif method == ScalingMethod.Hungarian:
        return scaling_hungarian(A, symmetric, **options)
    elif method == ScalingMethod.Equilib:
        return scaling_equilib(A, symmetric, **options)
    else:
        raise ValueError("Unknown scaling method")
