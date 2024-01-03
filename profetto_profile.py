import rswjax
import pandas as pd
import numpy as np
import jax

np.random.seed(5)
n = 1000000
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

def profile_rswjax():
    w, out, sol = rswjax.rsw(df, None, losses, regularizer, .01,  eps_abs=1e-8,verbose=True)

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    profile_rswjax()