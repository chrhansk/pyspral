import scipy as sp
import numpy as np

from pyspral.ssmfe import solve_standard


class Laplacian(sp.sparse.linalg.LinearOperator):

    def __init__(self, m):
        self.m = m

        n = m * m

        super().__init__(dtype=float,
                         shape=(n, n))

    def _matvec(self, x):
        m = self.m

        y = np.empty_like(x)

        xv = x.reshape((m, m))
        yv = y.reshape((m, m))

        for i in range(m):
            for j in range(m):
                z = 4. * xv[i, j]

                if i > 0:
                    z -= xv[i - 1, j]
                if j > 0:
                    z -= xv[i, j - 1]
                if i + 1 < m:
                    z -= xv[i + 1, j]
                if j + 1 < m:
                    z -= xv[i, j + 1]

                yv[i, j] = z

        return y


def test_solve_standard():
    m = 20
    n = m * m

    # left = 5
    # mep = 2*left
    left = 1
    mep = 10

    A = Laplacian(m)

    result = solve_standard(A,
                            left,
                            mep)

    lamb, x = result

    neig = x.shape[1]

    assert lamb.shape == (neig,)

    for i in range(neig):
        r = A.dot(x[:, i]) - (lamb[i] * x[:, i])
        assert np.allclose(r, 0.)
