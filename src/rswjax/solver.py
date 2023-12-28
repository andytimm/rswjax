import jax.numpy as jnp
import numpy as np
import scipy.sparse as sparse
import qdldl
from rswjax.losses import *
from rswjax.regularizers import *
from rswjax.losses import *
from rswjax.regularizers import *

@jit
def _projection_simplex(v, z=1):
    n_features = v.shape[0]
    u = jnp.sort(v)[::-1]
    cssv = jnp.cumsum(u) - z
    ind = jnp.arange(n_features) + 1
    cond = u - cssv / ind > 0

    def last_true_index(carry, x):
        idx, val = x
        return lax.cond(val, lambda _: idx, lambda _: carry, None), None

    rho, _ = lax.scan(last_true_index, 0, (ind, cond), length=n_features)
    
    # Compute theta safely, handling the case when cond is always False
    theta = lax.cond(rho > 0, lambda _: cssv[rho - 1] / rho, lambda _: 0.0, None)

    w = jnp.maximum(v - theta, 0)
    return w

def admm(F, losses, reg, lam, rho=50, maxiter=5000, eps=1e-6, warm_start={}, verbose=False,
         eps_abs=1e-5, eps_rel=1e-5):
    m, n = F.shape
    ms = [l.m for l in losses]

    # Initialization with JAX arrays
    f = warm_start.get("f", jnp.array(F.mean(axis=1)).flatten())
    w = warm_start.get("w", jnp.ones(n) / n)
    w_bar = warm_start.get("w_bar", jnp.ones(n) / n)
    w_tilde = warm_start.get("w_tilde", jnp.ones(n) / n)
    y = warm_start.get("y", jnp.zeros(m))
    z = warm_start.get("z", jnp.zeros(n))
    u = warm_start.get("u", jnp.zeros(n))

    # Constructing and factorizing the Q matrix with scipy and qdldl
 
    Q = sparse.bmat([
        [2 * sparse.eye(n), F.T],
        [F, -sparse.eye(m)]
    ])
    factor = qdldl.Solver(Q)

    w_best = None
    best_objective_value = float("inf")

    for k in range(maxiter):
        ct_cum = 0
        for l in losses:
            f = f.at[ct_cum:ct_cum + l.m].set(l.prox(F[ct_cum:ct_cum + l.m] @ w -
                                                    y[ct_cum:ct_cum + l.m], 1 / rho))
            ct_cum += l.m

        w_tilde = reg.prox(w - z, lam / rho)
        w_bar = _projection_simplex(w - u)

        rhs_np = np.concatenate([
            np.array(F.T @ (f + y) + w_tilde + z + w_bar + u),
            np.zeros(m)
        ])
        w_new_np = factor.solve(rhs_np)[:n]
        w_new = jnp.array(w_new_np)

        s = rho * jnp.concatenate([
            F @ w_new - f,
            w_new - w,
            w_new - w
        ])
        w = w_new

        y = y + f - F @ w
        z = z + w_tilde - w
        u = u + w_bar - w

        r = jnp.concatenate([
            f - F @ w,
            w_tilde - w,
            w_bar - w
        ])

        p = m + 2 * n
        Ax_k_norm = jnp.linalg.norm(jnp.concatenate([f, w_tilde, w_bar]))
        Bz_k_norm = jnp.linalg.norm(jnp.concatenate([w, w, w]))
        ATy_k_norm = jnp.linalg.norm(rho * jnp.concatenate([y, z, u]))
        eps_pri = jnp.sqrt(p) * eps_abs + eps_rel * jnp.maximum(Ax_k_norm, Bz_k_norm)
        eps_dual = jnp.sqrt(p) * eps_abs + eps_rel * ATy_k_norm

        s_norm = jnp.linalg.norm(s)
        r_norm = jnp.linalg.norm(r)
        if verbose and k % 50 == 0:
            print(f'It {k:03d} / {maxiter:03d} | {r_norm / eps_pri:8.5e} | {s_norm / eps_dual:8.5e}')

        if isinstance(reg, BooleanRegularizer):
            ct_cum = 0
            objective = 0.
            for l in losses:
                objective += l.evaluate(F[ct_cum:ct_cum + l.m] @ w_tilde)
                ct_cum += l.m
            if objective < best_objective_value:
                if verbose:
                    print(f"Found better objective value: {best_objective_value:3.5f} -> {objective:3.5f}")
                best_objective_value = objective
                w_best = w_tilde

        if r_norm <= eps_pri and s_norm <= eps_dual:
            break

    if not isinstance(reg, BooleanRegularizer):
        w_best = w_bar

    return {
        "f": np.array(f),
        "w": np.array(w),
        "w_bar": np.array(w_bar),
        "w_tilde": np.array(w_tilde),
        "y": np.array(y),
        "z": np.array(z),
        "u": np.array(u),
        "w_best": np.array(w_best) if w_best is not None else None
    }