#cython: language_level=3str

from enum import Enum
import warnings
from typing import Optional

cimport numpy as np

import numpy as np
import scipy as sp


cdef extern from "spral.h":

    struct spral_ssmfe_rcid:
        int job
        int nx
        int jx
        int kx
        int ny
        int jy
        int ky
        int i
        int j
        int k
        double alpha
        double beta
        double *x
        double *y
        char unused[80]

    struct spral_ssmfe_rciz:
        int job
        int nx
        int jx
        int kx
        int ny
        int jy
        int ky
        int i
        int j
        int k
        np.complex128_t alpha
        np.complex128_t beta
        np.complex128_t *x
        np.complex128_t *y
        char unused[80]

    struct spral_ssmfe_core_options:
        int array_base
        double cf_max
        int err_est
        int extra_left
        int extra_right
        double min_gap
        bint minAprod
        bint minBprod
        char unused[80]

    struct spral_ssmfe_options:
        int array_base
        int print_level
        int unit_error
        int unit_warning
        int unit_diagnostic
        int max_iterations
        int user_x
        int err_est
        double abs_tol_lambda
        double rel_tol_lambda
        double abs_tol_residual
        double rel_tol_residual
        double tol_x
        double left_gap
        double right_gap
        int extra_left
        int extra_right
        int max_left
        int max_right
        bint minAprod
        bint minBprod

    struct spral_ssmfe_inform:
        int flag
        int stat
        int non_converged
        int iteration
        int left
        int right
        int *converged
        double next_left
        double next_right
        double *residual_norms
        double *err_lambda
        double *err_X
        char unused[80]

    void spral_ssmfe_default_options(spral_ssmfe_options *options);

    void spral_ssmfe_standard_double(spral_ssmfe_rcid *rci, int left,
          int mep, double *lamb, int n, double *x, int ldx,
          void **keep, const spral_ssmfe_options *options,
          spral_ssmfe_inform *inform)

    void spral_ssmfe_standard_double_complex(spral_ssmfe_rciz *rci,
          int left, int mep, double *lamb, int n, np.complex128_t *x,
          int ldx, void **keep, const spral_ssmfe_options *options,
          spral_ssmfe_inform *inform)

    void spral_ssmfe_standard_shift_double(spral_ssmfe_rcid *rci,
          double sigma, int left, int right, int mep, double *lamb, int n,
          double *x, int ldx, void **keep,
          const spral_ssmfe_options *options,
          spral_ssmfe_inform *inform)

    void spral_ssmfe_standard_shift_double_complex(
          spral_ssmfe_rciz *rci, double sigma, int left, int right, int mep,
          double *lamb, int n, np.complex128_t *x, int ldx, void **keep,
          const spral_ssmfe_options *options,
          spral_ssmfe_inform *inform)

    void spral_ssmfe_generalized_double(spral_ssmfe_rcid *rci,
          int left, int mep, double *lamb, int n, double *x, int ldx,
          void **keep, const spral_ssmfe_options *options,
          spral_ssmfe_inform *inform)

    void spral_ssmfe_generalized_double_complex(spral_ssmfe_rciz *rci,
          int left, int mep, double *lamb, int n, np.complex128_t *x,
          int ldx, void **keep, const spral_ssmfe_options *options,
          spral_ssmfe_inform *inform)

    void spral_ssmfe_generalized_shift_double(spral_ssmfe_rcid *rci,
          double sigma, int left, int right, int mep, double *lamb, int n,
          double *x, int ldx, void **keep,
          const spral_ssmfe_options *options,
          spral_ssmfe_inform *inform)

    void spral_ssmfe_generalized_shift_double_complex(
          spral_ssmfe_rciz *rci, double sigma, int left, int right, int mep,
          double *lamb, int n, np.complex128_t *x, int ldx, void **keep,
          const spral_ssmfe_options *options,
          spral_ssmfe_inform *inform)

    void spral_ssmfe_buckling_double(spral_ssmfe_rcid *rci,
          double sigma, int left, int right, int mep, double *lamb, int n,
          double *x, int ldx, void **keep,
          const spral_ssmfe_options *options,
          spral_ssmfe_inform *inform)

    void spral_ssmfe_buckling_double_complex(spral_ssmfe_rciz *rci,
          double sigma, int left, int right, int mep, double *lamb, int n,
          np.complex128_t *x, int ldx, void **keep,
          const spral_ssmfe_options *options,
          spral_ssmfe_inform *inform)

    void spral_ssmfe_free_double(void **keep, spral_ssmfe_inform *inform)

    void spral_ssmfe_free_double_complex(void **keep, spral_ssmfe_inform *inform)


class Flag(Enum):
    JOB_OUT_OF_RANGE = -1
    N_OUT_OF_RANGE = -9
    LDX_OUT_OF_RANGE = -10
    LEFT_OUT_OF_RANGE = -11
    RIGHT_OUT_OF_RANGE = -12
    MEP_LT_N_EIGENPAIRS = -13
    NOT_ENOUGH_MEM = -100
    MAT_NOT_POSDEF = -200
    COMPUTATION_FINISHED = 1
    EXCEEDED_MAX_NUM_ITS = 2
    OUT_OF_STORAGE_FOR_EIGENPAIRS = 3


class Result:
    def __init__(self, lamb, x, **props):
        self.lamb = lamb
        self.x = x
        self.props = props

    def __getattr__(self, name):
        return self.props[name]


class PrintLevel(Enum):
    NoPrinting = -1
    ErrorWarning = 0
    BasicDiagnostic = 1
    DetailedDiagnostic = 2


cdef class Options:
    cdef spral_ssmfe_options options

    def __cinit__(self, **options):
        spral_ssmfe_default_options(&self.options)

    def __init__(self, **options):
        for key, value in options.items():
            setattr(self, key, value)

    property array_base:
        def __set__(self, value): self.options.array_base = value

    property print_level:
        def __set__(self, value): self.options.print_level = value.value

    property unit_error:
        def __set__(self, value): self.options.unit_error = value

    property unit_warning:
        def __set__(self, value): self.options.unit_warning = value

    property unit_diagnostic:
        def __set__(self, value): self.options.unit_diagnostic = value

    property max_iterations:
        def __set__(self, value): self.options.max_iterations = value

    property user_x:
        def __set__(self, value): self.options.user_x = value

    property err_est:
        def __set__(self, value): self.options.err_est = value

    property abs_tol_lambda:
        def __set__(self, value): self.options.abs_tol_lambda = value

    property rel_tol_lambda:
        def __set__(self, value): self.options.rel_tol_lambda = value

    property abs_tol_residual:
        def __set__(self, value): self.options.abs_tol_residual = value

    property rel_tol_residual:
        def __set__(self, value): self.options.rel_tol_residual = value

    property tol_x:
        def __set__(self, value): self.options.tol_x = value

    property left_gap:
        def __set__(self, value): self.options.left_gap = value

    property right_gap:
        def __set__(self, value): self.options.right_gap = value

    property extra_left:
        def __set__(self, value): self.options.extra_left = value

    property extra_right:
        def __set__(self, value): self.options.extra_right = value

    property max_left:
        def __set__(self, value): self.options.max_left = value

    property max_right:
        def __set__(self, value): self.options.max_right = value

    property minAprod:
        def __set__(self, value): self.options.minAprod = value

    property minBprod:
        def __set__(self, value): self.options.minBprod = value


cdef _solve_standard(A: sp.sparse.linalg.LinearOperator,
                     left: int,
                     mep: int,
                     Options options,
                     P: Optional[sp.sparse.linalg.LinearOperator] = None,
                     x0: double[:,::1]=None):
    cdef spral_ssmfe_inform inform
    cdef void* keep = NULL

    cdef int _left = left
    cdef int _mep = mep
    cdef int n
    cdef int ldx
    cdef double[:] lamb
    cdef double[:, ::1] x
    cdef spral_ssmfe_rcid rci

    assert mep >= left

    rci.job = 0

    (m, n) = A.shape

    if m != n:
        raise ValueError("Matrix A is not square")

    lamb = np.zeros(left, dtype=np.float64)

    if x0 is None:
        x = np.zeros((mep, n), dtype=np.float64)
    else:
        assert x0.shape == (mep, n)
        x = x0

    ldx = n

    options.left_gap = -0.1

    try:
        while True:
            spral_ssmfe_standard_double(&rci,
                                        _left,
                                        _mep,
                                        &lamb[0],
                                        n,
                                        &x[0, 0],
                                        ldx,
                                        &keep,
                                        &options.options,
                                        &inform)

            if rci.job == -3:
                # Fatal error
                raise ValueError("Fatal error")
            elif rci.job == -2:
                # Error, restart?
                raise ValueError("Error occurred during computation")
            elif rci.job == -1:
                # Computation complete
                break
            elif rci.job == 1:
                # Compute Y = AX
                xv = np.asarray(<double[:rci.nx, :n]> rci.x)
                yv = np.asarray(<double[:rci.nx, :n]> rci.y)
                yv[:] = (A @ xv.T).T
            elif rci.job == 2:
                # Apply precond Y = TX
                xv = np.asarray(<double[:rci.nx, :n]> rci.x)
                yv = np.asarray(<double[:rci.nx, :n]> rci.y)

                if P is None:
                    yv[:] = xv
                else:
                    yv[:] = (P @ xv.T).T
                continue
            else:
                raise ValueError("Invalid job")
    finally:
        spral_ssmfe_free_double(&keep, &inform)

    num_eigenpairs = inform.left

    props = {
        'iteration': inform.iteration,
        'next_left': inform.next_left,
    }

    if num_eigenpairs > 0:
        lambv = np.asarray(lamb[:num_eigenpairs])
        xv = np.asarray(x[:num_eigenpairs, :]).T
    else:
        lambv = np.empty(shape=(0,), dtype=np.float64)
        xv = np.empty(shape=(n, 0), dtype=np.float64)

    return Result(lambv, xv, **props)


def solve_standard(A: sp.sparse.linalg.LinearOperator,
                   left: int,
                   mep: int,
                   P: Optional[sp.sparse.linalg.LinearOperator] = None,
                   x0: double[:,::1]=None,
                   **options):
    cdef Options opts = Options(**options)
    return _solve_standard(A, left, mep, opts, P, x0)


cdef _solve_generalized(A: sp.sparse.linalg.LinearOperator,
                        B: sp.sparse.linalg.LinearOperator,
                        left: int,
                        mep: int,
                        Options options,
                        P: Optional[sp.sparse.linalg.LinearOperator] = None,
                        x0: double[:,::1]=None):
    cdef spral_ssmfe_inform inform
    cdef void* keep = NULL

    cdef int _left = left
    cdef int _mep = mep
    cdef int n
    cdef int ldx
    cdef double[:] lamb
    cdef double[:, ::1] x
    cdef spral_ssmfe_rcid rci

    assert mep >= left

    rci.job = 0

    (m, n) = A.shape

    if m != n:
        raise ValueError("Matrix A is not square")

    lamb = np.zeros(left, dtype=np.float64)

    if x0 is None:
        x = np.zeros((mep, n), dtype=np.float64)
    else:
        assert x0.shape == (mep, n)
        x = x0

    ldx = n

    try:
        while True:
            spral_ssmfe_generalized_double(&rci,
                                           _left,
                                           _mep,
                                           &lamb[0],
                                           n,
                                           &x[0, 0],
                                           ldx,
                                           &keep,
                                           &options.options,
                                           &inform)

            if rci.job == -3:
                # Fatal error
                raise ValueError("Fatal error")
            elif rci.job == -2:
                # Error, restart?
                raise ValueError("Error occurred during computation")
            elif rci.job == -1:
                # Computation complete
                break
            elif rci.job == 1:
                # Compute Y = AX
                xv = np.asarray(<double[:rci.nx, :n]> rci.x)
                yv = np.asarray(<double[:rci.nx, :n]> rci.y)
                yv[:] = (A @ xv.T).T
            elif rci.job == 2:
                # Apply precond Y = TX
                xv = np.asarray(<double[:rci.nx, :n]> rci.x)
                yv = np.asarray(<double[:rci.nx, :n]> rci.y)

                if P is None:
                    yv[:] = xv
                else:
                    yv[:] = (P @ xv.T).T
                continue

            elif rci.job == 3:
                # Compute Y = BX
                xv = np.asarray(<double[:rci.nx, :n]> rci.x)
                yv = np.asarray(<double[:rci.nx, :n]> rci.y)

                yv[:] = (B @ xv.T).T
            else:
                raise ValueError("Invalid job")
    finally:
        spral_ssmfe_free_double(&keep, &inform)

    num_eigenpairs = inform.left

    props = {
        'iteration': inform.iteration,
        'next_left': inform.next_left,
    }

    if num_eigenpairs > 0:
        lambv = np.asarray(lamb[:num_eigenpairs])
        xv = np.asarray(x[:num_eigenpairs, :]).T
    else:
        lambv = np.empty(shape=(0,), dtype=np.float64)
        xv = np.empty(shape=(n, 0), dtype=np.float64)

    return Result(lambv, xv, **props)


def solve_generalized(A: sp.sparse.linalg.LinearOperator,
                      B: sp.sparse.linalg.LinearOperator,
                      left: int,
                      mep: int,
                      P: Optional[sp.sparse.linalg.LinearOperator] = None,
                      x0: double[:,::1]=None,
                      **options):
    cdef Options opts = Options(**options)
    return _solve_generalized(A, B, left, mep, opts, P, x0)


cpdef solve_buckling():
    raise NotImplementedError()
