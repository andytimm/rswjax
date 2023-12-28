import numpy as np
import jax.numpy as jnp
import cvxpy as cp
import rswjax
from rswjax.solver import _projection_simplex

# The function shouldn't have changed in conversion to jax at all, just ensuring that
def projection_simplex_numpy(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def test_projection_simplex():
    np.random.seed(0)  # For reproducibility
    v = np.random.randn(10)  # Random vector

    # Apply JAX projection
    v_jax = jnp.array(v)
    projected_jax = _projection_simplex(v_jax)

    # Apply NumPy projection
    projected_numpy = projection_simplex_numpy(v)

    # Convert JAX array to NumPy for comparison
    projected_jax_numpy = np.array(projected_jax)

    # Assert that the projections are close
    np.testing.assert_allclose(projected_jax_numpy, projected_numpy, atol=1e-5)

def test_solver():
    n = 100
    m = 20
    F = np.random.randn(m, n)
    fdes1 = np.random.randn(m // 2)
    fdes2 = np.random.randn(m // 2)
    
    # Create loss instances
    loss1 = rswjax.LeastSquaresLoss(fdes1)
    loss2 = rswjax.InequalityLoss(fdes2, -1 * np.ones(m // 2), 1 * np.ones(m // 2))
    losses = [loss1, loss2]

    # Extract proximal functions and sizes
    prox_functions = [loss.prox for loss in losses]
    ms = [loss.m for loss in losses]

    # Regularizer
    reg = rswjax.EntropyRegularizer()
    reg_prox = reg.prox

    # Call admm with new signature
    sol = rswjax.admm(F, prox_functions, ms, reg_prox, 1, verbose=True)
    
    # Solve equivalent problem with cvxpy for comparison
    w = cp.Variable(n)
    cp.Problem(cp.Minimize(0.5 * cp.sum_squares(F[:m // 2] @ w - fdes1) - cp.sum(cp.entr(w))),
               [cp.sum(w) == 1, w >= 0, cp.max(cp.abs(F[m // 2:] @ w - fdes2)) <= 1]).solve(solver=cp.ECOS)

    # Assert that the solutions are close
    np.testing.assert_allclose(w.value, sol["w"], atol=1e-3)