import numpy as np
import jax.numpy as jnp
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

