# Changelog

<!--next-version-placeholder-->

## v1.1.0 (14/11/2023)

### Feature

- Replaced the jitted `_projection_simplex` jax function with a numpy implementation that is ~5x faster for most data.

### Fix

- Ensured that the comparison of loss length to columns in data worked works for vectorized losses consistently, added test to cover case as well.


## v1.0.0 (01/05/2023)

- First release of `rswjax`!