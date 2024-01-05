import pandas as pd
import numpy as np
from numpy import linalg
import rswjax

np.random.seed(5)
n = 100
age = np.random.randint(20, 30, size=n) * 1.
sex = np.random.choice([0., 1.], p=[.4, .6], size=n)
height = np.random.normal(5, 1, size=n)

df = pd.DataFrame({
    "age": age,
    "sex": sex,
    "height": height
})

def test_entropy_weights():
    funs = [
        lambda x: x.age,
        lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
        lambda x: x.height
    ]
    losses = [rswjax.EqualityLoss(25), rswjax.EqualityLoss(.5),
            rswjax.EqualityLoss(5.3)]
    regularizer = rswjax.EntropyRegularizer()
    w, out, sol = rswjax.rsw(df, funs, losses, regularizer, 1., verbose=True)

def test_kl_loss():
    losses = [rswjax.EqualityLoss(25), rswjax.EqualityLoss(.5),
          rswjax.KLLoss(5.3,scale=1)]
    regularizer = rswjax.EntropyRegularizer()
    w, out, sol = rswjax.rsw(df, None, losses, regularizer, 1., eps_abs=1e-8, verbose = True)

def test_kl_reg():
    funs = [
        lambda x: x.age,
        lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
        lambda x: x.height
    ]
    
    fdes = np.random.uniform(0, 1, size=100)
    fdes /= fdes.sum()

    losses = [rswjax.EqualityLoss(25), rswjax.EqualityLoss(.5),
            rswjax.EqualityLoss(5.3)]
    regularizer = rswjax.KLRegularizer(prior = fdes)
    rswjax.rsw(df, funs, losses, regularizer, 1., verbose=True)

# This is a common design pattern from the examples, and for handling very large numbers of columns
def test_array_inputs():
    funs = [
    lambda x: x.age,
    lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
    lambda x: x.height
]

    array_of_inputs = np.array([25,.5,5.3])

    losses = [rswjax.EqualityLoss(array_of_inputs.flatten())]
    regularizer = rswjax.EntropyRegularizer()
    rswjax.rsw(df, funs, losses, regularizer, 1., eps_abs=1e-8, verbose = True)

def test_bool_weights():
    funs = [
        lambda x: x.age,
        lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
        lambda x: x.height
    ]
    losses = [rswjax.LeastSquaresLoss(25), rswjax.LeastSquaresLoss(.5),
            rswjax.LeastSquaresLoss(5.3)]
    regularizer = rswjax.BooleanRegularizer(5)
    rswjax.rsw(df, funs, losses, regularizer, 1., verbose=True)

def test_nans():
    for i, j in zip(np.random.randint(50, size=25), np.random.randint(3, size=25)):
        df.iat[i, j] *= np.nan
    # Real
    funs = [
        lambda x: x.age,
        lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
        lambda x: x.height
    ]
    losses = [rswjax.EqualityLoss(25), rswjax.EqualityLoss(.5),
            rswjax.EqualityLoss(5.3)]
    regularizer = rswjax.EntropyRegularizer()
    rswjax.rsw(df, funs, losses, regularizer, 1.)