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



def test_bool_weights():
    funs = [
        lambda x: x.age,
        lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
        lambda x: x.height
    ]
    losses = [rswjax.LeastSquaresLoss(25), rswjax.LeastSquaresLoss(.5),
            rswjax.LeastSquaresLoss(5.3)]
    regularizer = rswjax.BooleanRegularizer(5)
    w, out, sol = rswjax.rsw(df, funs, losses, regularizer, 1., verbose=True)
    df["weight"] = w

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
    w, out, sol = rswjax.rsw(df, funs, losses, regularizer, 1.)