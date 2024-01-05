import pytest
import numpy as np
import pandas as pd
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

funs = [
        lambda x: x.age,
        lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
        lambda x: x.height
    ]
losses = [rswjax.EqualityLoss(25), rswjax.EqualityLoss(.5),
            rswjax.EqualityLoss(5.3)]
regularizer = rswjax.EntropyRegularizer()
    

def test_nan_from_admm_exception():
    # Create an array with NaN values

    # Use pytest.raises to test for ValueError
    with pytest.raises(ValueError) as e:
        # very small numbers of lambda are an easy way to make NaNs
        rswjax.rsw(df, funs, losses, regularizer, .000001, verbose=True)