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

@jit
def compute_convergence_criteria(r, s, f, w_tilde, w_bar, w, y, z, u, rho, eps_abs, eps_rel):
    p = f.size + 2 * w.size
    Ax_k_norm = jnp.linalg.norm(jnp.concatenate([f, w_tilde, w_bar]))
    Bz_k_norm = jnp.linalg.norm(jnp.concatenate([w, w, w]))
    ATy_k_norm = jnp.linalg.norm(rho * jnp.concatenate([y, z, u]))
    eps_pri = jnp.sqrt(p) * eps_abs + eps_rel * jnp.maximum(Ax_k_norm, Bz_k_norm)
    eps_dual = jnp.sqrt(p) * eps_abs + eps_rel * ATy_k_norm
    r_norm = jnp.linalg.norm(r)
    s_norm = jnp.linalg.norm(s)
    return r_norm, s_norm, eps_pri, eps_dual

@jit
def update_f(f, F_w_y, prox_functions, ms, lam, rho):
    ct_cum = 0
    for prox, m in zip(prox_functions, ms):
        # Pass the correct number of arguments to prox
        f = f.at[ct_cum:ct_cum + m].set(prox(F_w_y[ct_cum:ct_cum + m], lam, rho))
        ct_cum += m
    return f


@jit
def update_w_bar(w, u):
    return _projection_simplex(w - u)

@jit
def compute_residuals(f, w, w_tilde, w_bar, F, rho):
    r = jnp.concatenate([f - F @ w, w_tilde - w, w_bar - w])
    s = rho * jnp.concatenate([F @ w - f, w - w_tilde, w - w_bar])
    return r, s

def admm(F, prox_functions, ms, reg_prox, lam, rho=50, maxiter=5000, eps=1e-6, warm_start={}, verbose=False,
         eps_abs=1e-5, eps_rel=1e-5):
    m, n = F.shape

    # Initialization with JAX arrays
    f = warm_start.get("f", jnp.array(F.mean(axis=1)).flatten())
    w = warm_start.get("w", jnp.ones(n) / n)
    w_bar = warm_start.get("w_bar", jnp.ones(n) / n)
    w_tilde = warm_start.get("w_tilde", jnp.ones(n) / n)
    y = warm_start.get("y", jnp.zeros(m))
    z = warm_start.get("z", jnp.zeros(n))
    u = warm_start.get("u", jnp.zeros(n))

    # Factorize Q matrix outside JIT-compiled function
    Q = sparse.bmat([
        [2 * sparse.eye(n), F.T],
        [F, -sparse.eye(m)]
    ])
    factor = qdldl.Solver(Q)

    w_best = None
    best_objective_value = float("inf")

    for k in range(maxiter):
        F_w_y = F @ w - y
        # Perform updates (JIT-compiled)
        f = update_f(f, F_w_y, prox_functions, ms, lam, rho)
        w_tilde = reg_prox(w - z, lam / rho)
        w_bar = update_w_bar(w, u)

        # Compute concatenated rhs for solving
        rhs_np = np.concatenate([np.array(F.T @ (f + y) + w_tilde + z + w_bar + u), np.zeros(m)])
        w_new_np = factor.solve(rhs_np)[:n]
        w_new = jnp.array(w_new_np)

        # Compute residuals (JIT-compiled)
        r, s = compute_residuals(f, w_new, w_tilde, w_bar, F, rho)

        # Update primal and dual variables
        w = w_new
        y = y + f - F @ w
        z = z + w_tilde - w
        u = u + w_bar - w

        # Check convergence (JIT-compiled)
        r_norm, s_norm, eps_pri, eps_dual = compute_convergence_criteria(r, s, f, w_tilde, w_bar, w, y, z, u, rho, eps_abs, eps_rel)

        if verbose and k % 50 == 0:
            print(f'It {k:03d} / {maxiter:03d} | {r_norm / eps_pri:8.5e} | {s_norm / eps_dual:8.5e}')

        # ... (rest of the loop)

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