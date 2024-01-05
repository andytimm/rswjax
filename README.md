# rswjax

A [Jax](https://github.com/google/jax) implementation of optimal representative sample weighting, drawing heavily on the original `rsw` implementation of Barratt, Angeris, and Boyd ([rsw](https://github.com/cvxgrp/rsw/tree/master)).

Thanks to rewriting some core operations in Jax, it is significantly faster than `rsw` for medium-large datasets, especially those with many columns (ex: 5k+ rows, 20+ columns). For more thoughts on performance, see the section below. In addition,
it adds a number of quality of life improvements, like early stopping if optimization produces NaNs, warnings for common issues like not including loss functions for every column, and a
broader test suite.

## Installation

```bash
$ pip install rswjax
```

## Usage
`rswjax` has one main user facing method, `rsw`, with signature:
```{python}
"""Optimal representative sample weighting.

    Arguments:
        - df: Pandas dataframe
        - funs: functions to apply to each row of df. Function warns if len(losses) != ncols.
        - losses: list of losses, each one of rswjax.EqualityLoss, rswjax.InequalityLoss, rswjax.LeastSquaresLoss,
            or rswjax.KLLoss.
        - regularizer: One of rswjax.ZeroRegularizer, rswjax.EntropyRegularizer,
            or rswjax.KLRegularizer, rswjax.BooleanRegularizer
        - lam (optional): Regularization hyper-parameter (default=1).
        - kwargs (optional): additional arguments to be sent to solver. For example: verbose=True,
            maxiter=5000, rho=50, eps_rel=1e-5, eps_abs=1e-5.

    Returns:
        - w: Final sample weights.
        - out: Final induced expected values as a list of numpy arrays.
        - sol: Dictionary of final ADMM variables. Can be ignored.
    """
```

Example usage to fit a weights set to simulated data:
```{python}
import pandas as pd
import numpy as np
import rswjax

np.random.seed(605)
n = 5000
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
w, out, sol = rswjax.rsw(df, funs, losses, regularizer, .01, eps_abs=1e-8, verbose = True)
```

For more details on how one might use the package to do survey weighting, check out my recent [talk](https://andytimm.github.io/posts/NYSOPM_talk_regularized_raking/NYOSPM_talk.html), at [NYOSP](https://nyhackr.org/index.html). The talk uses the original `rsw`, but all ideas should transfer over cleanly.

## Performance

`rswjax` is generally faster than `rsw` for medium-large datasets, especially those with many columns. As both packages take neglible amounts of time for data ~3k rows or less,
`rswjax` should be superior for many but not all applications.

Here is a simple scaling test in n (# of rows), with structure similar to the simulated example in `/examples`:

```
rsw
n=1,000- 109 ms ± 6.6 ms per loop (mean ± std. dev. of 7 runs)
n=3,000- 461 ms ± 14 ms per loop (mean ± std. dev. of 7 runs)
n=5,000- 1.72 s ± 155 ms per loop (mean ± std. dev. of 7 runs)
n=10,000- 3.47 s ± 138 ms per loop (mean ± std. dev. of 7 runs)
n=100,000- 43.7 s ± 508 ms per loop (mean ± std. dev. of 7 runs)
n=1,000,000- 3min 28s ± 11.6 s per loop (mean ± std. dev. of 7 runs)

rswjax
n=1,000- 284 ms ± 9.09 ms per loop (mean ± std. dev. of 7 runs)
n=3,000- 832 ms ± 20.9 ms per loop (mean ± std. dev. of 7 runs)
n=5,000- 1.4 s ± 28.1 ms per loop (mean ± std. dev. of 7 runs)
n=10,000- 2.93 s ± 36.4 ms per loop (mean ± std. dev. of 7 runs)
n=100,000- 9.77 s ± 84.3 ms per loop (mean ± std. dev. of 7 runs
n=1,000,000- 1min 26s ± 1.03 s per loop (mean ± std. dev. of 7 runs)
```

For a rough sense of scaling in the number of columns m, consider these results on a simple test with n = 5000 rows, and m = 20/50/100 columns to weight on:

```
n = 5000, m = 20,50,100

rsw -
m=20- 8.5s
m=50-166s
m=100- 448s

rswjax -
m=20- 7.4s
m=50- 60s
m=100- 183s
```

(Note that this test case uses randomly generated targets and data, and is therfore hard to weight in high dimensions. Thus, most well specified real world examples
should run significantly faster due to early termination.)

This speed is achieved by doing the core qdldl factorization and solve using the `qdldl` package, but using JITed ([just-in-time compiled](https://github.com/google/jax?tab=readme-ov-file#compilation-with-jit)) rewrites of many pieces of the admm solver, losses, and regularizers. There are still some minor opportunities to speed up the package by further refactoring the code to allow greater portions to be JITed, or by optimizing how and when data is converted back and forth between `numpy` and `jax.numpy`.

## Running the examples

There are two examples, one on simulated data and one on the [CDC BRFSS dataset](https://stanford.edu/~boyd/papers/optimal_representative_sampling.html). Both are due to the original package authors.

### Simulated
To run the simulated example, after installing `rswjax`, navigate to the `examples` folder and run:
```
$ python simulated.py
```

### CDC BRFSS
To run the CDC BRFSS example, first download the data:
```
$ cd examples/data
$ wget https://www.cdc.gov/brfss/annual_data/2018/files/LLCP2018XPT.zip
$ unzip LLCP2018XPT.zip
```

In the examples folder, to run all the examples in the paper, execute the following command:
```
$ python brfss.py
```


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

Some possible ideas for contributing would be:
1. making a jittable version of the steps to update *f* in the solver. This would require a decide amount of refactoring (so as to pass data instead of objects into the logic to update f, a requirement for jit).
2. Finding opportunities to convert data back and forth less from  `numpy` to `jax.numpy` and vice versa.
3. Adding additional losses, regularizers, or examples.

## License

`rswjax` was created by Andrew Timm. It is licensed under the terms of the Apache License 2.0 license.

See the NOTICE file for attributions due to the original `rsw` authors, whose code and paper are the primary origin of most logic in my package.

## Credits

`rswjax` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
