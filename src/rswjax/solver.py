import jax.numpy as jnp

from rswjax.losses import *
from rswjax.regularizers import *


def _projection_simplex(v, z=1):
    n_features = v.shape[0]
    u = jnp.sort(v)[::-1]
    cssv = jnp.cumsum(u) - z
    ind = jnp.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = jnp.maximum(v - theta, 0)
    return w