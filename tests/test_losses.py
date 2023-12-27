import cvxpy as cp
import numpy as np
import rswjax

m = 10
f = np.random.randn(m)
fdes = np.random.randn(m)
lam = 1

def test_equality_loss():
    equality = rswjax.EqualityLoss(fdes)
    fhat = cp.Variable(m)
    cp.Problem(cp.Minimize(1 / lam * cp.sum_squares(fhat - f)),
                [fhat == fdes]).solve()
    np.testing.assert_allclose(fhat.value, equality.prox(f, lam))