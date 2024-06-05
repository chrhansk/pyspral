#cython: language_level=3str

from enum import Enum
import warnings

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


cpdef solve_standard(A: sp.sparse.linalg.LinearOperator,
                     left: int,
                     mep: int,
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
    cdef spral_ssmfe_options options

    assert mep >= left

    spral_ssmfe_default_options(&options)

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
                                        &options,
                                        &inform)

            print("Iteration {0}, job = {1}".format(inform.iteration,
                                                      rci.job))

            if rci.job == -3:
                # Fatal error
                raise ValueError("Fatal error")
            elif rci.job == -2:
                # Error, restart?
                raise ValueError("Error occurred during computation")
            elif rci.job == -1:
                print("Computation complete")
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
                yv[:] = xv
                continue
            else:
                raise ValueError("Invalid job")
    finally:
        spral_ssmfe_free_double(&keep, &inform)

    num_eigenpairs = inform.left

    print("{0} eigenpairs converged in {1} iterations".format(inform.left, inform.iteration));
    # print("Flag: {0}".format(Flag(inform.flag)))

    if num_eigenpairs > 0:
        return np.asarray(lamb[:num_eigenpairs]), np.asarray(x[:num_eigenpairs, :]).T

    return None


cpdef solve_generalized():
    pass

cpdef solve_buckling():
    pass
