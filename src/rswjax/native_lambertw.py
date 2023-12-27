import functools
import jax
import jax.numpy as jnp
from typing import Tuple


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def lambertw(z: jnp.ndarray, tol: float = 1e-8, max_iter: int = 100) -> jnp.ndarray:
    """Principal branch of the
    `Lambert W function <https://en.wikipedia.org/wiki/Lambert_W_function>`_.

    This implementation uses Halley's iteration and the global initialization
    proposed in :cite:`iacono:17`, Eq. 20 .

    Args:
      z: Array.
      tol: Tolerance threshold.
      max_iter: Maximum number of iterations.

    Returns:
      The Lambert W evaluated at ``z``.
    """
    def initial_iacono(x: jnp.ndarray) -> jnp.ndarray:
        y = jnp.sqrt(1.0 + jnp.e * x)
        num = 1.0 + 1.14956131 * y
        denom = 1.0 + 0.45495740 * jnp.log1p(y)
        return -1.0 + 2.036 * jnp.log(num / denom)

    def halley_iteration(container):
        it, _, w = container
        f = w - z * jnp.exp(-w)
        delta = f / (w + 1.0 - 0.5 * (w + 2.0) * f / (w + 1.0))
        w_next = w - delta
        not_converged = jnp.abs(delta) > tol * jnp.abs(w_next)
        return it + 1, not_converged, w_next

    def cond_fun(container):
        it, converged, _ = container
        return jnp.logical_and(jnp.any(~converged), it < max_iter)

    w0 = initial_iacono(z)
    converged = jnp.zeros_like(w0, dtype=bool)
    _, _, w = jax.lax.while_loop(cond_fun=cond_fun, body_fun=halley_iteration, init_val=(0, converged, w0))
    return w

@lambertw.defjvp
def _lambertw_jvp(tol: float, max_iter: int, primals: Tuple[jnp.ndarray, ...], tangents: Tuple[jnp.ndarray, ...]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    z, = primals
    dz, = tangents
    w = lambertw(z, tol=tol, max_iter=max_iter)
    pz = jnp.where(z == 0.0, 1.0, w / ((1.0 + w) * z))
    return w, pz * dz
