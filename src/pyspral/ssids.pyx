#cython: language_level=3str

from libc.stdint cimport int64_t

from enum import Enum


import warnings

import numpy as np
import scipy as sp

from .csc cimport _CSCMatrix
from .csc import _CSCMatrix


class EfficiencyWarning:
    pass


cdef extern from "spral.h":

    struct spral_ssids_options:
        int array_base
        int print_level
        int unit_diagnostics
        int unit_error
        int unit_warning
        int ordering
        int nemin
        bint ignore_numa
        bint use_gpu
        int64_t min_gpu_work
        float max_load_inbalance
        float gpu_perf_coeff
        int scaling
        int64_t small_subtree_threshold
        int cpu_block_size
        bint action
        int pivot_method
        double small
        double u
        char unused[80]

    struct spral_ssids_inform:
        int flag
        int matrix_dup
        int matrix_missing_diag
        int matrix_outrange
        int matrix_rank
        int maxdepth
        int maxfront
        int num_delay
        int64_t num_factor
        int64_t num_flops
        int num_neg
        int num_sup
        int num_two
        int stat
        int cuda_error
        int cublas_error
        int maxsupernode
        char unused[76]

    void spral_ssids_default_options(spral_ssids_options *options)

    void spral_ssids_analyse_ptr32(bint check, int n, int *order, const int *ptr,
                                   const int *row, const double *val, void **akeep,
                                   const spral_ssids_options *options,
                                   spral_ssids_inform *inform)

    void spral_ssids_factor_ptr32(bint posdef, const int *ptr, const int *row,
                                  const double *val, double *scale, void *akeep, void **fkeep,
                                  const spral_ssids_options *options,
                                  spral_ssids_inform *inform)

    void spral_ssids_solve(int job, int nrhs, double *x, int ldx, void *akeep,
                           void *fkeep, const spral_ssids_options *options,
                           spral_ssids_inform *inform)

    void spral_ssids_solve1(int job, double *x1, void *akeep, void *fkeep,
                            const spral_ssids_options *options,
                            spral_ssids_inform *inform)

    int spral_ssids_free_akeep(void **akeep)
    int spral_ssids_free_fkeep(void **fkeep)
    int spral_ssids_free(void **akeep, void **fkeep)

    void spral_ssids_enquire_posdef(const void *akeep, const void *fkeep,
                                    const spral_ssids_options *options,
                                    spral_ssids_inform *inform, double *d)

    void spral_ssids_enquire_indef(const void *akeep, const void *fkeep,
                                   const spral_ssids_options *options,
                                   spral_ssids_inform *inform, int *piv_order, double *d)

    void spral_ssids_alter(const double *d, const void *akeep, void *fkeep,
                           const spral_ssids_options *options,
                           spral_ssids_inform *inform)


ErrorFlags = {
    0: "Success.",
    # Errors
    -1: "Error in sequence of calls (may be caused by failure of a preceding call).",
    -2: "n<0 or ne<1.",
    -3: "Error in ptr(:).",
    -4: "CSC format: All variable indices in one or more columns are out-of-range. Coordinate format: All entries are out-of-range.",
    -5: "Matrix is singular and options%action=.false.",
    -6: "Matrix found not to be positive definite but posdef=true.",
    -7: "ptr(:) and/or row(:) not present, but required as ssids_analyse() was called with check=.false,.",
    -8: "options%ordering out of range, or options%ordering=0 and order parameter not provided or not a valid permutation.",
    -9: "options%ordering=-2 but val(:) was not supplied.",
    -10: "ldx<n or nrhs<1.",
    -11: "job is out-of-range.",
    -13: "Called ssids_enquire_posdef() on indefinite factorization.",
    -14: "Called ssids_enquire_indef() on positive-definite factorization.",
    -15: "options%scaling=3 but a matching-based ordering was not performed during analyse phase.",
    -50: "Allocation error. If available, the stat parameter is returned in inform%stat.",
    -51: "CUDA error. The CUDA error return value is returned in inform%cuda_error.",
    -52: "CUBLAS error. The CUBLAS error return value is returned in inform%cublas_error.",
    -53: "OpenMP cancellation is disabled. Please set the environment variable OMP_CANCELLATION=true.",
    # Warnings
    1: "Out-of-range variable indices found and ignored in input data. inform%matrix_outrange is set to the number of such entries.",
    2: "Duplicate entries found and summed in input data. inform%matrix_dup is set to the number of such entries.",
    3: "Combination of +1 and +2.",
    4: "One or more diagonal entries of A are missing.",
    5: "Combination of +4 and +1 or +2.",
    6: "Matrix is found be (structurally) singular during analyse phase. This will overwrite any of the above warning flags.",
    7: "Matrix is found to be singular during factorize phase.",
    8: "Matching-based scaling found as side-effect of matching-based ordering ignored (consider setting options%scaling=3).",
    50: "OpenMP processor binding is disabled. Consider setting the environment variable OMP_PROC_BIND=true (this may affect performance on NUMA systems)."
}


class PivotMethod(Enum):
    AgressiveAPosteriori = 1
    BlockAPosteriori = 2
    ThresholdPartial = 3


class Ordering(Enum):
    UserSupplied = 0
    METIS = 1
    MatchingBased = 2


class Scaling(Enum):
    NoScaling = 0
    MC64 = 1
    Auction = 2
    MC77 = 3
    NormEquilibration = 4


class PrintLevel(Enum):
    NoPrinting = -1
    ErrorWarning = 0
    BasicDiagnostic = 1
    DetailedDiagnostic = 2


cdef class _Inform:
    cdef spral_ssids_inform inform

    @property
    def flag(self):
        return self.inform.flag

    @property
    def matrix_dup(self):
        return self.inform.matrix_dup

    @property
    def matrix_missing_diag(self):
        return self.inform.matrix_missing_diag

    @property
    def matrix_outrange(self):
        return self.inform.matrix_outrange

    @property
    def matrix_rank(self):
        return self.inform.matrix_rank

    @property
    def maxdepth(self):
        return self.inform.maxdepth

    @property
    def maxfront(self):
        return self.inform.maxfront

    @property
    def num_delay(self):
        return self.inform.num_delay

    @property
    def num_factor(self):
        return self.inform.num_factor

    @property
    def num_flops(self):
        return self.inform.num_flops

    @property
    def num_neg(self):
        return self.inform.num_neg

    @property
    def num_sup(self):
        return self.inform.num_sup

    @property
    def num_two(self):
        return self.inform.num_two

    @property
    def stat(self):
        return self.inform.stat

    @property
    def cuda_error(self):
        return self.inform.cuda_error

    @property
    def cublas_error(self):
        return self.inform.cublas_error

    @property
    def maxsupernode(self):
        return self.inform.maxsupernode


cdef class Options:
    cdef spral_ssids_options options

    def __cinit__(self, **kwds):
        spral_ssids_default_options(&self.options)

    def __init__(self, **kwds):
        for key, value in kwds.items():
            setattr(self, key, value)

    property array_base:
        def __get__(self): return self.options.array_base
        def __set__(self, value): self.options.array_base = value

    property print_level:
        def __get__(self): return PrintLevel(self.options.print_level)
        def __set__(self, value): self.options.print_level = value.value

    property unit_diagnostics:
        def __get__(self): return self.options.unit_diagnostics
        def __set__(self, value): self.options.unit_diagnostics = value

    property unit_error:
        def __get__(self): return self.options.unit_error
        def __set__(self, value): self.options.unit_error = value

    property unit_warning:
        def __get__(self): return self.options.unit_warning
        def __set__(self, value): self.options.unit_warning = value

    property ordering:
        def __get__(self): return Ordering(self.options.ordering)
        def __set__(self, value): self.options.ordering = value.value

    property nemin:
        def __get__(self): return self.options.nemin
        def __set__(self, value): self.options.nemin = value

    property ignore_numa:
        def __get__(self): return self.options.ignore_numa
        def __set__(self, value): self.options.ignore_numa = value

    property use_gpu:
        def __get__(self): return self.options.use_gpu
        def __set__(self, value): self.options.use_gpu = value

    property min_gpu_work:
        def __get__(self): return self.options.min_gpu_work
        def __set__(self, value): self.options.min_gpu_work = value

    property max_load_imbalance:
        def __get__(self): return self.options.max_load_imbalance
        def __set__(self, value): self.options.max_load_imbalance = value

    property gpu_perf_coeff:
        def __get__(self): return self.options.gpu_perf_coeff
        def __set__(self, value): self.options.gpu_perf_coeff = value

    @property
    def scaling(self):
        scaling = min(max(self.options.scaling, 0), 4)
        return Scaling(scaling)

    @scaling.setter
    def scaling(self, value):
        self.options.scaling = value.value

    property small_subtree_threshold:
        def __get__(self): return self.options.small_subtree_threshold
        def __set__(self, value): self.options.small_subtree_threshold = value

    property cpu_block_size:
        def __get__(self): return self.options.cpu_block_size
        def __set__(self, value): self.options.cpu_block_size = value

    property action:
        def __get__(self): return self.options.action
        def __set__(self, value): self.options.action = value

    property pivot_method:
        def __get__(self): return PivotMethod(self.options.pivot_method)
        def __set__(self, value): self.options.pivot_method = value.value

    property small:
        def __get__(self): return self.options.small
        def __set__(self, value): self.options.small = value

    property u:
        def __get__(self): return self.options.u
        def __set__(self, value): self.options.u = value


cdef class SymbolicFactor:
    cdef void *akeep
    cdef _CSCMatrix csc_mat
    cdef Options options

    def __cinit__(self):
        self.akeep = NULL
        self.csc_mat = None
        self.options = None

    def __dealloc__(self):
        spral_ssids_free_akeep(&self.akeep)

    def factor(self, bint posdef, double[:] scale=None):
        cdef void* fkeep = NULL
        cdef double* scale_ptr = NULL
        cdef NumericFactor numeric_factor = NumericFactor()

        if scale is not None:
            if self.options.scaling != Scaling.NoScaling:
                raise ValueError("User-provided scaling with automatic scaling option")
            assert scale.ndim == 1
            (_, n) = self.csc_mat.shape
            assert scale.size == n
            scale_ptr = &scale[0]

        spral_ssids_factor_ptr32(posdef,
                                 &self.csc_mat.cols[0],
                                 &self.csc_mat.rows[0],
                                 &self.csc_mat.data[0],
                                 scale_ptr,
                                 self.akeep,
                                 &fkeep,
                                 &self.options.options,
                                 &numeric_factor._inform.inform)

        if numeric_factor._inform.inform.flag < 0:
            spral_ssids_free_fkeep(&fkeep)
            raise SSIDSError(numeric_factor._inform.inform.flag)

        numeric_factor.fkeep = fkeep
        numeric_factor.csc_mat = self.csc_mat
        numeric_factor.symbolic_factor = self
        numeric_factor.posdef = posdef

        return numeric_factor


class SSIDSError(Exception):
    def __init__(self, flag):
        self.flag = flag
        self.message = ErrorFlags.get(flag, "Unknown error")

    def __str__(self):
        return self.message


class Job(Enum):
    Axb = 0
    PLxSb = 1
    Dxb = 2
    PLTSinvxb = 3
    DPLTSinvxb = 4


cdef class NumericFactor:
    cdef _CSCMatrix csc_mat
    cdef SymbolicFactor symbolic_factor
    cdef bint posdef
    cdef void *fkeep
    cdef _Inform _inform

    def __cinit__(self):
        self.fkeep = NULL
        self._inform = _Inform()

    def __dealloc__(self):
        spral_ssids_free_fkeep(&self.fkeep)

    @property
    def inform(self):
        return self._inform

    cdef _solve_single(self, double[:] rhs, job=Job.Axb, inplace=False):
        cdef int _job = job.value
        cdef double[:] sol_rhs

        (m, _) = self.csc_mat.shape

        if rhs.shape[0] != m:
            raise ValueError("Invalid shape of rhs")

        if not inplace:
            sol_rhs = np.copy(rhs)
        else:
            sol_rhs = rhs

        spral_ssids_solve1(_job,
                           &sol_rhs[0],
                           self.symbolic_factor.akeep,
                           self.fkeep,
                           &self.symbolic_factor.options.options,
                           &self._inform.inform)

        if self._inform.inform.flag < 0:
            raise SSIDSError(self._inform.inform.flag)

        return sol_rhs

    cdef _solve_multi(self, double[:, ::1] rhs, job=Job.Axb, inplace=False):
        cdef int _job = job.value
        cdef double[:, ::1] sol_rhs
        cdef int nrhs
        cdef int leadim

        (m, _) = self.csc_mat.shape

        if rhs.shape[1] != m:
            raise ValueError("Invalid shape of rhs")

        if not inplace:
            sol_rhs = np.copy(rhs)
        else:
            sol_rhs = rhs

        nrhs = sol_rhs.shape[0]
        leadim = sol_rhs.strides[0] // sol_rhs.itemsize

        spral_ssids_solve(_job,
                          nrhs,
                          &sol_rhs[0, 0],
                          leadim,
                          self.symbolic_factor.akeep,
                          self.fkeep,
                          &self.symbolic_factor.options.options,
                          &self._inform.inform)

        if self._inform.inform.flag < 0:
            raise SSIDSError(self._inform.inform.flag)

        return np.asarray(sol_rhs)

    def solve(self, rhs, job=Job.Axb, inplace=False):

        if rhs.ndim == 1:
            return self._solve_single(rhs, job, inplace)
        else:
            return self._solve_multi(rhs, job, inplace)

    cdef _enquire_posdef(self):
        cdef double[:] d

        (_, n) = self.csc_mat.shape
        d = np.empty(n, dtype=np.double)

        spral_ssids_enquire_posdef(self.symbolic_factor.akeep,
                                   self.fkeep,
                                   &self.symbolic_factor.options.options,
                                   &self._inform.inform,
                                   &d[0])

        if self._inform.inform.flag < 0:
            raise SSIDSError(self._inform.inform.flag)

        return d

    cdef _enquire_indef(self):
        cdef double[:, ::1] d
        cdef int[:] piv_order

        (_, n) = self.csc_mat.shape

        d_arr = np.empty(shape=(n, 2), dtype=np.double)
        d = d_arr
        piv_order_arr = np.empty(n, dtype=np.intc)
        piv_order = piv_order_arr

        spral_ssids_enquire_indef(self.symbolic_factor.akeep,
                                 self.fkeep,
                                 &self.symbolic_factor.options.options,
                                 &self._inform.inform,
                                 &piv_order[0],
                                 &d[0, 0])

        if self._inform.inform.flag < 0:
            raise SSIDSError(self._inform.inform.flag)

        return piv_order_arr, d_arr

    def alter(self, double[:, ::1] d):
        (_, n) = self.csc_mat.shape

        if (d.shape[0] != n) or (d.shape[1] != 2):
            raise ValueError("Invalid shape of d")

        spral_ssids_alter(&d[0, 0],
                          self.symbolic_factor.akeep,
                          self.fkeep,
                          &self.symbolic_factor.options.options,
                          &self._inform.inform)

        if self._inform.inform.flag < 0:
            raise SSIDSError(self._inform.inform.flag)

    def D(self):
        if self.posdef:
            d = self._enquire_posdef()
        else:
            piv_order, d = self._enquire_indef()
            piv_perm = np.abs(piv_order) - 1
            d = d[piv_perm, :]

        (_, n) = self.csc_mat.shape

        dsub = d[1:, 1]
        dmain = d[:, 0]

        diagonals = [dsub, dmain, dsub]
        return sp.sparse.diags_array(diagonals, offsets=[-1, 0, 1])

    def enquire(self):
        if self.posdef:
            return self._enquire_posdef()
        else:
            return self._enquire_indef()


cdef _analyze(mat, Options options, check=False, int[:] order=None):
    cdef void* akeep = NULL
    cdef void* fkeep = NULL
    cdef int* order_ptr = NULL
    cdef const spral_ssids_options* _options
    cdef _CSCMatrix csc_mat
    cdef SymbolicFactor symbolic_factor = SymbolicFactor()
    cdef spral_ssids_inform inform

    assert options is not None
    options.array_base = 1

    if not sp.sparse.issparse(mat):
        raise ValueError("Input matrix must be a sparse matrix")

    if mat.format != 'csc':
        warnings.warn("Input matrix is not in CSC format. Converting to CSC format.", EfficiencyWarning)
        mat = mat.tocsc()

    (m, n) = mat.shape

    _options = &options.options
    csc_mat = _CSCMatrix(mat, symmetric=True)

    if m != n:
        raise ValueError("Input matrix must be square")

    if order is not None:
        if options.ordering != Ordering.UserSupplied:
            raise ValueError("User-provided ordering with automatic ordering option")
        assert order.ndim == 1
        assert order.size == n
        order_ptr = &order[0]

    if mat.dtype != float:
        raise ValueError("Input matrix must be of type float")

    spral_ssids_analyse_ptr32(check,
                              n,
                              order_ptr,
                              &csc_mat.cols[0],
                              &csc_mat.rows[0],
                              &csc_mat.data[0],
                              &akeep,
                              _options,
                              &inform)

    if inform.flag < 0:
        spral_ssids_free(&akeep, &fkeep)
        raise SSIDSError(inform.flag)

    symbolic_factor.akeep = akeep
    symbolic_factor.csc_mat = csc_mat
    symbolic_factor.options = options

    return symbolic_factor


def analyze(mat, check=False, int[:] order=None, **options):
    cdef Options _options = Options(**options)
    return _analyze(mat, options=_options, check=check, order=order)


def solve(mat, rhs, posdef, check=False, inplace=False, **options):
    symbolic_factor = analyze(mat, check=check, **options)
    numeric_factor = symbolic_factor.factor(posdef=posdef)
    return numeric_factor.solve(rhs, inplace=inplace)
