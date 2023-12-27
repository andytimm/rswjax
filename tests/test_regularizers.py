import pytest
import cvxpy as cp
import numpy as np
import rswjax

w = np.random.randn(10)
prior = np.random.uniform(10)
prior /= 10
lam = .5

def test_zero_regularizer():
    zero_reg = rswjax.ZeroRegularizer()
    np.testing.assert_allclose(zero_reg.prox(w, .5), w)
