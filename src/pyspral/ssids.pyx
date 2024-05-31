#cython: language_level=3str

from libc.stdint cimport int64_t

from enum import Enum


import warnings

import numpy as np
import scipy as sp


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

    void spral_ssids_analyse(bint check, int n, int *order, const int64_t *ptr,
                             const int *row, const double *val, void **akeep,
                             const spral_ssids_options *options,
                             spral_ssids_inform *inform)

    void spral_ssids_factor(bint posdef, const int64_t *ptr, const int *row,
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


def iter_spmatrix(matrix):
    if sp.sparse.isspmatrix_coo(matrix):
        for r, c, m in zip(matrix.row, matrix.col, matrix.data):
            yield r, c, m

    elif sp.sparse.isspmatrix_csc(matrix):
        for c in range(matrix.shape[1]):
            for ind in range(matrix.indptr[c], matrix.indptr[c+1]):
                yield matrix.indices[ind], c, matrix.data[ind]

    elif sp.sparse.isspmatrix_csr(matrix):
        for r in range(matrix.shape[0]):
            for ind in range(matrix.indptr[r], matrix.indptr[r+1]):
                yield r, matrix.indices[ind], matrix.data[ind]

    elif sp.sparse.isspmatrix_lil(matrix):
        for r in range(matrix.shape[0]):
            for c, d in zip(matrix.rows[r], matrix.data[r]):
                yield r, c, d
    else:
        raise NotImplementedError("Invalid matrix format")


class PivotMethod(Enum):
    AgressiveAPosteriori = 1
    BlockAPosteriori = 2
    ThresholdPartial = 3


class Ordering(Enum):
    UserSupplied = 0
    METIS = 1
    MatchingBased = 2


class ScalingAlgorithm(Enum):
    ExplicitScaling = 0
    HungarianMethod = 1
    AuctionAlgorithm = 2
    MatchingBased = 3
    NormEquilibration = 4


class PrintLevel(Enum):
    NoPrinting = -1
    ErrorWarning = 0
    BasicDiagnostic = 1
    DetailedDiagnostic = 2


cdef class Inform:
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

    def __cinit__(self):
        spral_ssids_default_options(&self.options)
        self.options.array_base = 1

    def __init__(self, **kwds):
        for key, value in kwds.items():
            setattr(self, key, value)

    @property
    def array_base(self):
        return self.options.array_base

    @array_base.setter
    def array_base(self, value):
        self.options.array_base = value

    @property
    def print_level(self):
        return PrintLevel(self.options.print_level)

    @print_level.setter
    def print_level(self, value):
        self.options.print_level = value.value

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

    @property
    def unit_warning(self):
        return self.options.unit_warning

    @unit_warning.setter
    def unit_warning(self, value):
        self.options.unit_warning = value

    @property
    def ordering(self):
        return Ordering(self.options.ordering)

    @ordering.setter
    def ordering(self, value):
        self.options.ordering = value.value

    @property
    def nemin(self):
        return self.options.nemin

    @nemin.setter
    def nemin(self, value):
        self.options.nemin = value

    @property
    def ignore_numa(self):
        return self.options.ignore_numa

    @ignore_numa.setter
    def ignore_numa(self, value):
        self.options.ignore_numa = value

    @property
    def use_gpu(self):
        return self.options.use_gpu

    @use_gpu.setter
    def use_gpu(self, value):
        self.options.use_gpu = value

    @property
    def min_gpu_work(self):
        return self.options.min_gpu_work

    @min_gpu_work.setter
    def min_gpu_work(self, value):
        self.options.min_gpu_work = value

    @property
    def max_load_inbalance(self):
        return self.options.max_load_inbalance

    @max_load_inbalance.setter
    def max_load_inbalance(self, value):
        self.options.max_load_inbalance = value

    @property
    def gpu_perf_coeff(self):
        return self.options.gpu_perf_coeff

    @gpu_perf_coeff.setter
    def gpu_perf_coeff(self, value):
        self.options.gpu_perf_coeff = value

    @property
    def scaling(self):
        scaling = min(max(self.options.scaling, 0), 4)
        return ScalingAlgorithm(scaling)

    @scaling.setter
    def scaling(self, value):
        self.options.scaling = value.value

    @property
    def small_subtree_threshold(self):
        return self.options.small_subtree_threshold

    @small_subtree_threshold.setter
    def small_subtree_threshold(self, value):
        self.options.small_subtree_threshold = value

    @property
    def cpu_block_size(self):
        return self.options.cpu_block_size

    @cpu_block_size.setter
    def cpu_block_size(self, value):
        self.options.cpu_block_size = value

    @property
    def action(self):
        return self.options.action

    @action.setter
    def action(self, value):
        self.options.action = value

    @property
    def pivot_method(self):
        return PivotMethod(self.options.pivot_method)

    @pivot_method.setter
    def pivot_method(self, value):
        self.options.pivot_method = value.value

    @property
    def small(self):
        return self.options.small

    @small.setter
    def small(self, value):
        self.options.small = value

    @property
    def u(self):
        return self.options.u

    @u.setter
    def u(self, value):
        self.options.u = value


cdef class CSCMatrix:
    cdef const int[:] rows
    cdef const int64_t[:] cols
    cdef const double[:] data
    cdef object shape

    def __cinit__(self, mat):
        (m, n) = mat.shape

        nnz = mat.getnnz()

        row_entries = [np.array([1], dtype=np.intc)] + [list() for _ in range(m)]
        data_entries = [list() for _ in range(m)]

        for r, c, v in iter_spmatrix(mat):
            if r < c:
                continue

            row_entries[c + 1].append(r + 1)
            data_entries[c].append(v)

        for c in range(m):
            row_entries[c + 1] = np.array(row_entries[c + 1],
                                          dtype=np.intc)

            data_entries[c] = np.array(data_entries[c],
                                       dtype=np.double)

            perm = np.argsort(row_entries[c + 1])
            row_entries[c + 1] = row_entries[c + 1][perm]
            data_entries[c] = data_entries[c][perm]

        self.rows = np.concatenate(row_entries[1:])
        self.cols = np.cumsum([row.size for row in row_entries], dtype=np.int64)
        self.data = np.concatenate(data_entries)
        self.shape = mat.shape


cdef class SymbolicFactor:
    cdef void *akeep
    cdef CSCMatrix csc_mat
    cdef Options options

    def __cinit__(self):
        self.akeep = NULL
        self.csc_mat = None
        self.options = None

    def __dealloc__(self):
        spral_ssids_free_akeep(&self.akeep)

    def factor(self, bint posdef):
        cdef void* fkeep = NULL
        cdef NumericFactor numeric_factor = NumericFactor()

        spral_ssids_factor(posdef,
                           &self.csc_mat.cols[0],
                           &self.csc_mat.rows[0],
                           &self.csc_mat.data[0],
                           NULL,
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
    cdef CSCMatrix csc_mat
    cdef SymbolicFactor symbolic_factor
    cdef bint posdef
    cdef void *fkeep
    cdef Inform _inform

    def __cinit__(self):
        self.fkeep = NULL
        self._inform = Inform()

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


cpdef analyze(mat, Options options=None, check=False):
    cdef void* akeep = NULL
    cdef void* fkeep = NULL
    cdef const spral_ssids_options* _options
    cdef CSCMatrix csc_mat
    cdef SymbolicFactor symbolic_factor = SymbolicFactor()
    cdef spral_ssids_inform inform

    if not sp.sparse.issparse(mat):
        raise ValueError("Input matrix must be a sparse matrix")

    if mat.format != 'csc':
        warnings.warn("Input matrix is not in CSC format. Converting to CSC format.", EfficiencyWarning)
        mat = mat.tocsc()

    (m, n) = mat.shape

    if options is None:
        options = Options()

    _options = &options.options
    csc_mat = CSCMatrix(mat)

    if m != n:
        raise ValueError("Input matrix must be square")

    if mat.dtype != float:
        raise ValueError("Input matrix must be of type float")

    spral_ssids_analyse(check, n, NULL,
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
