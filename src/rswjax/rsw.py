import pandas as pd
import jax.numpy as jnp
import numpy as np
from scipy import sparse
import time

from rswjax.solver import admm


def rsw(df, funs, losses, regularizer, lam=1, **kwargs):
    if funs is not None:
        F = np.vstack([df.apply(f, axis=1) for f in funs]).T
    else:
        F = np.array(df).T

    # Remove NaNs by changing F
    rows_nan, cols_nan = np.where(np.isnan(F))
    desireds = [l.fdes for l in losses]
    desired = np.concatenate(desireds)
    for i in np.unique(rows_nan):
        F[i, cols_nan[rows_nan == i]] = desired[i]

    F_sparse = sparse.csc_matrix(F)

    # Extract proximal functions and their sizes from losses
    prox_functions = [l.prox for l in losses]
    ms = [l.m for l in losses]

    # Get the proximal function for the regularizer
    reg_prox = regularizer.prox

    tic = time.time()
    sol = admm(F_sparse, prox_functions, ms, reg_prox, lam, **kwargs)
    toc = time.time()

    if kwargs.get("verbose", False):
        print("ADMM took %3.5f seconds" % (toc - tic))

    out = []
    means = F @ sol["w_best"]
    ct = 0
    for m in ms:
        out.append(means[ct:ct + m])
        ct += m

    return sol["w_best"], out, sol