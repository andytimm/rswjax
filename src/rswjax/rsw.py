import pandas as pd
import jax.numpy as jnp
import numpy as np
from scipy import sparse
import time

from rswjax.solver import admm


def rsw(df, funs, losses, regularizer, lam=1, **kwargs):
    """Optimal representative sample weighting.

    Arguments:
        - df: Pandas dataframe
        - funs: functions to apply to each row of df.
        - losses: list of losses, each one of rsw.EqualityLoss, rsw.InequalityLoss, rsw.LeastSquaresLoss,
            or rsw.KLLoss()
        - regularizer: One of rsw.ZeroRegularizer, rsw.EntropyRegularizer,
            or rsw.KLRegularizer, rsw.BooleanRegularizer
        - lam (optional): Regularization hyper-parameter (default=1).
        - kwargs (optional): additional arguments to be sent to solver. For example: verbose=False,
            maxiter=5000, rho=50, eps_rel=1e-5, eps_abs=1e-5.

    Returns:
        - w: Final sample weights.
        - out: Final induced expected values as a list of numpy arrays.
        - sol: Dictionary of final ADMM variables. Can be ignored.
    """
    if funs is not None:
        F = []
        for f in funs:
            F += [df.apply(f, axis=1)]
        F = jnp.array(F, dtype=float)
    else:
        F = np.array(df).T
    m, n = F.shape

    # Function to replace NaNs in a JAX array
    def replace_nans(F, desired):
        # Identify the NaN elements
        is_nan = jnp.isnan(F)

        # Ensure 'desired' is a JAX array and has the right shape
        desired_jax = jnp.array(desired)

        # This will broadcast 'desired' across NaN positions in F
        desired_expanded = jnp.expand_dims(desired_jax, axis=1)
        desired_expanded = jnp.broadcast_to(desired_expanded, F.shape)

        # Replace NaNs with the corresponding values from 'desired'
        F_no_nans = jnp.where(is_nan, desired_expanded, F)
        return F_no_nans

    # remove nans by changing F
    # Prepare 'desired' array based on 'losses'
    desireds = [l.fdes for l in losses]
    desired = np.concatenate(desireds)

# Replace NaNs in F
    F = replace_nans(F, desired)

    tic = time.time()
    sol = admm(F, losses, regularizer, lam, **kwargs)
    toc = time.time()
    if kwargs.get("verbose", False):
        print("ADMM took %3.5f seconds" % (toc - tic))

    out = []
    means = F @ sol["w_best"]
    ct = 0
    for m in [l.m for l in losses]:
        out += [means[ct:ct + m]]
        ct += m
    return sol["w_best"], out, sol