import pytest
import cvxpy as cp
import numpy as np
import rswjax

np.random.seed(605)
w = np.random.randn(10)
prior = np.random.uniform(10)
prior /= 10
lam = .5

def test_zero_regularizer():
    zero_reg = rswjax.ZeroRegularizer()
    np.testing.assert_allclose(zero_reg.prox(w, .5), w)

def test_entropy_regularizer():
    entropy_reg = rswjax.EntropyRegularizer()
    what = cp.Variable(10)
    cp.Problem(cp.Minimize(-cp.sum(cp.entr(what)) + 1 /
                           (2 * lam) * cp.sum_squares(what - w))).solve(solver=cp.ECOS)
    np.testing.assert_allclose(what.value, entropy_reg.prox(w, lam), atol=1e-4)

def test_kl_regularizer():
    kl_reg = rswjax.KLRegularizer(prior)
    what = cp.Variable(10)
    cp.Problem(cp.Minimize(-cp.sum(cp.entr(what)) - cp.sum(cp.multiply(what, np.log(prior))) + 1 /
                           (2 * lam) * cp.sum_squares(what - w))).solve(solver=cp.ECOS)
    np.testing.assert_allclose(what.value, kl_reg.prox(w, lam), atol=1e-4)

