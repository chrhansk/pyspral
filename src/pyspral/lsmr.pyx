#cython: language_level=3str

from enum import Enum
from typing import Optional

import numpy as np
import scipy as sp


cdef extern from "spral.h":
    struct spral_lsmr_options:
        double atol
        double btol
        double conlim
        int ctest
        int itnlim
        int itn_test
        int localSize
        int print_freq_head
        int print_freq_itn
        int unit_diagnostics
        int unit_error

    struct spral_lsmr_inform:
        int flag
        int itn
        int stat
        double normb
        double normAP
        double condAP
        double normr
        double normAPr
        double normy

    void spral_lsmr_default_options(spral_lsmr_options *options)

    int spral_lsmr_solve(int *action,
                         int m,
                         int n,
                         double u[],
                         double v[],
                         double y[],
                         void **keep,
                         const spral_lsmr_options *options,
                         spral_lsmr_inform *inform,
                         double *damp)

    int spral_lsmr_free(void **keep)


class ConvergenceTest(Enum):
    manual_simple = 1
    manual_norm = 2
    fond_saunders = 3


cdef class Options:
    cdef spral_lsmr_options options

    def __cinit__(self, **options):
        spral_lsmr_default_options(&self.options)

        for key, value in options.items():
            setattr(self, key, value)

    @property
    def atol(self):
        return self.options.atol

    @atol.setter
    def atol(self, value):
        self.options.atol = value

    @property
    def btol(self):
        return self.options.btol

    @btol.setter
    def btol(self, value):
        self.options.btol = value

    @property
    def conlim(self):
        return self.options.conlim

    @conlim.setter
    def conlim(self, value):
        self.options.conlim = value

    @property
    def ctest(self):
        return ConvergenceTest(self.options.ctest)

    @ctest.setter
    def ctest(self, value):
        self.options.ctest = ConvergenceTest[value].value

    @property
    def itnlim(self):
        return self.options.itnlim

    @itnlim.setter
    def itnlim(self, value):
        self.options.itnlim = value

    @property
    def itn_test(self):
        return self.options.itn_test

    @itn_test.setter
    def itn_test(self, value):
        self.options.itn_test = value

    @property
    def localSize(self):
        return self.options.localSize

    @localSize.setter
    def localSize(self, value):
        self.options.localSize = value

    @property
    def print_freq_head(self):
        return self.options.print_freq_head

    @print_freq_head.setter
    def print_freq_head(self, value):
        self.options.print_freq_head = value

    @property
    def print_freq_itn(self):
        return self.options.print_freq_itn

    @print_freq_itn.setter
    def print_freq_itn(self, value):
        self.options.print_freq_itn = value

    @property
    def unit_diagnostics(self):
        return self.options.unit_diagnostics

    @unit_diagnostics.setter
    def unit_diagnostics(self, value):
        self.options.unit_diagnostics = value

    @property
    def unit_error(self):
        return self.options.unit_error

    @unit_error.setter
    def unit_error(self, value):
        self.options.unit_error = value


class Status(Enum):
    zero_sol = 0
    compat_sol = 1
    suff_sol = 2
    cond_limit = 3
    res_suff_small = 4
    iter_limit = 7
    alloc_fail = 8
    dealloc_fail = 9
    neg_dimen = 10
    manual_abort = 11

    def error(self):
        error_codes = [Status.alloc_fail,
                       Status.dealloc_fail,
                       Status.neg_dimen]

        return self in error_codes

    @staticmethod
    def from_int(value):
        try:
            return Status(value)
        except ValueError:
            if value in [5, 6]:
                return Status.res_suff_small
            else:
                raise ValueError("Invalid status: {0}".format(value))

    def success(self):
        success_codes = [Status.zero_sol,
                         Status.compat_sol,
                         Status.suff_sol,
                         Status.cond_limit,
                         Status.res_suff_small]

        return self in success_codes


status_descriptions = {
    Status.zero_sol: "x = 0 is the exact solution. No iterations were performed.",
    Status.compat_sol: "The equations Ax = b are probably compatible. ||Ax - b|| is sufficiently small, given the values of `atol` and `btol`.",
    Status.suff_sol: "If lamb is zero then the system is probably not compatible. A least-squares solution has been obtained that is sufficiently accurate, given the value of `atol`. Otherwise, damped least-squares solution has been obtained that is sufficiently accurate, given the value of `atol`.",
    Status.cond_limit: "An estimate of cond(A) has exceeded `conlim`. The system appears to be ill-conditioned, or there could be an error in the products with A, A^T, P, or P^T.",
    Status.res_suff_small: "||APy - b|| is small enough for this machine.",
    Status.iter_limit: "Reached the iteration limit `itnlim` before the other stopping criteria were satisfied.",
    Status.alloc_fail: "Failure to allocate working space.",
    Status.dealloc_fail: "Failure to deallocate working space.",
    Status.neg_dimen: "One of the input dimensions is negative.",
    Status.manual_abort: "Computation was terminated by the user.",
}


class Result:
    def __init__(self, x, **props):
        self.x = x
        self.props = props

    def __getattr__(self, name):
        return self.props[name]


cpdef _solve(A: sp.sparse.linalg.LinearOperator,
             b: np.ndarray,
             P: Optional[sp.sparse.linalg.LinearOperator] = None,
             lamb: float = 0,
             options: Options = Options(),
             callback=None):
    cdef int action = 0
    cdef int m, n
    cdef double[::1] u_arr
    cdef double[::1] v_arr
    cdef double[::1] y_arr
    cdef void* keep = NULL
    cdef double damp = lamb
    cdef spral_lsmr_inform inform

    (m, n) = A.shape

    assert b.shape == (m,)

    if P is not None:
        assert P.shape == (n, n)

    u = np.copy(b)
    v = np.empty(shape=(n,))
    y = np.empty(shape=(n,))

    if options.ctest != ConvergenceTest.fond_saunders:
        if callback is None:
            raise ValueError("No callback provided for convergence test")
    else:
        if callback is not None:
            raise ValueError("Callback provided without convergence test")

    u_arr = u
    v_arr = v
    y_arr = y

    manual_abort = False

    try:
        while True:
            spral_lsmr_solve(&action,
                             m,
                             n,
                             &u_arr[0],
                             &v_arr[0],
                             &y_arr[0],
                             &keep,
                             &options.options,
                             &inform,
                             &damp)

            if action == 0:
                break
            elif action == 1:
                ATu = A.rmatvec(u)

                if P is not None:
                    v += P.rmatvec(ATu)
                else:
                    v += ATu

                continue
            elif action == 2:
                vv = v

                if P is not None:
                    vv = P.matvec(v)

                u += A.matvec(vv)
                continue
            elif action == 3:
                assert callback is not None

                itn = inform.itn

                if options.ctest == ConvergenceTest.manual_simple:
                    if not(callback(y, iteration=itn)):
                        manual_abort = True
                        break
                else:
                    assert options.ctest == ConvergenceTest.manual_norm

                    norms = {"condAP": inform.condAP,
                             "normAP": inform.normAP,
                             "normAPr": inform.normAPr,
                             "normr": inform.normr,
                             "normy": inform.normy}

                    if not(callback(y, iteration=itn, norms=norms)):
                        manual_abort = True
                        break

                continue
            else:
                raise ValueError("Invalid action: {0}".format(action))
    finally:
        spral_lsmr_free(&keep)

    if manual_abort:
        status = Status.manual_abort
    else:
        status = Status(inform.flag)

    if status.error():
        raise ValueError("Invalid status: {0}".format(status))

    return Result(y,
                  iterations=inform.itn,
                  status=status.name,
                  description=status_descriptions[status],
                  success=status.success())


def solve(A: sp.sparse.linalg.LinearOperator,
          b: np.ndarray,
          P: Optional[sp.sparse.linalg.LinearOperator] = None,
          lamb: float = 0,
          callback=None,
          **options):

    return _solve(A,
                  b,
                  P,
                  lamb,
                  Options(**options),
                  callback=callback)
