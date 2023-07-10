from scipy.sparse.linalg import LinearOperator


class DenseTensorLinearOperator(LinearOperator):
    def __init__(self, A0, A1):
        self.A0 = A0
        self.A1 = A1
        self.n0 = self.A0.shape[0]
        self.n1 = self.A1.shape[0]
        self.shape = self.n0 * self.n1, self.n0 * self.n1
        self.dtype = self.A0.dtype

    def _matvec(self, x):
        n0 = self.A0.shape[0]
        n1 = self.A1.shape[0]
        xbis0 = x.reshape([n0, n1])
        ytemp = self.A0 * xbis0
        ztemp = (self.A1 * ytemp.T).T.reshape(n0 * n1)
        return ztemp
