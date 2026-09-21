"""
src/validate/utils/ -- validation helpers that are NOT one of the nine generic checks.

`outliers.py` is the shared MAD kernel (`scripts/dod/data_profile.py` and the `timeseries`
check both read it, and two subtly different "outlier counts" for the same column would be
worse than one). `prices.py` is the prices-specific invariant suite `step_build_cube` gates on.
"""
