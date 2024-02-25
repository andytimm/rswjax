# Changelog

<!--next-version-placeholder-->

## v1.2.0 (25/02/2024)

### Feature

- added a new regularizer, `SumSquaresRegularizer`, which is the regularizer preferred in [Ben-Michael et al. (2023)](https://www.cambridge.org/core/journals/political-analysis/article/multilevel-calibration-weighting-for-survey-data/CCB1183BA82E7589F4187DE23406C153?utm_source=pocket_saves)'s,
a similarly flexible, optimization based approach to building weights.

## v1.1.0 (14/11/2023)

### Feature

- Replaced the jitted `_projection_simplex` jax function with a numpy implementation that is ~5x faster for most data.

### Fix

- Ensured that the comparison of loss length to columns in data worked works for vectorized losses consistently, added test to cover case as well.


## v1.0.0 (01/05/2023)

- First release of `rswjax`!