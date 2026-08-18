"""
outliers.py  (src/utils/outliers.py)
------------------------------------
The MAD-based modified Z-score, shared by the two places that need it: the fundamentals
history audit (`src/validate/analyze_history.py::detect_level_outliers`) and the
Definition-of-Done data profiler (`scripts/dod/data_profile.py`).

Why MAD and not a standard deviation: a single 10x mis-tagged XBRL fact inflates the standard
deviation enough to hide itself. The median and the median absolute deviation do not move, so
the spike still scores far from the centre. That is the whole reason this repo flags level
outliers this way, and it is why the profiler must use the SAME kernel -- two subtly different
"outlier counts" for the same column would be worse than one.

Design notes
  * `reference` EXISTS FOR THE YoY PATH. The year-on-year check scores a 4-period-lagged
    difference whose first (up to) 4 entries are undefined. Those entries must be scored (as
    NaN, so they can never be flagged) but must NOT contribute to the median/MAD -- filling
    them with 0 first was a real false-positive bug, since a 0 in a series of large diffs is
    itself an outlier. So: statistics from `reference`, scores over `values`.
  * `fallback_to_mean_abs_dev` IS NOT COSMETIC. When MAD == 0 (more than half the series is
    identical -- common for a flat balance-sheet line) the level check falls back to the mean
    absolute deviation so a genuine lone spike is still caught. The YoY check deliberately
    does NOT: a zero MAD there means the differences are essentially constant, and every
    caller of that path treats "no dispersion" as "nothing to say".
  * Returns NaN where the input is NaN, and zeros when there is no dispersion at all. Both
    compare False against any threshold, so a caller's `z > threshold` mask is safe without
    any extra NaN handling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Consistency constant for the modified Z-score. Kept as the value this repo's audit has
#: always used, so historical flags stay reproducible.
MODIFIED_Z_SCALE: float = 0.6745


def mad_center_scale(values: np.ndarray | pd.Series,
                     *, fallback_to_mean_abs_dev: bool = True) -> tuple[float, float]:
    """`(median, scale)` for the modified Z-score, NaNs ignored.

    `scale == 0.0` means "no dispersion" -- the caller must score every point as 0 rather than
    dividing. Returned separately from `modified_zscore` because the profiler reports the
    centre and scale as metrics in their own right."""
    arr = np.asarray(pd.Series(values).astype(float).dropna().values, dtype=float)
    if arr.size == 0:
        return (float("nan"), 0.0)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad > 0:
        return (median, mad)
    if fallback_to_mean_abs_dev:
        mean_abs_dev = float(np.mean(np.abs(arr - median)))
        if mean_abs_dev > 0:
            return (median, mean_abs_dev)
    return (median, 0.0)


def modified_zscore(values: np.ndarray | pd.Series,
                    *, reference: np.ndarray | pd.Series | None = None,
                    fallback_to_mean_abs_dev: bool = True) -> np.ndarray:
    """`|MODIFIED_Z_SCALE * (x - median) / scale|` as a float array the length of `values`.

    `reference` supplies the median/MAD sample when it differs from `values` (the YoY path).
    NaN in -> NaN out; no dispersion -> all zeros. See the module docstring for why both of
    those are load-bearing rather than defensive."""
    arr = np.asarray(pd.Series(values).astype(float).values, dtype=float)
    median, scale = mad_center_scale(reference if reference is not None else arr,
                                     fallback_to_mean_abs_dev=fallback_to_mean_abs_dev)
    if scale <= 0 or not np.isfinite(median):
        return np.zeros_like(arr)
    return MODIFIED_Z_SCALE * np.abs(arr - median) / scale


def count_mad_outliers(values: np.ndarray | pd.Series, *, threshold: float = 3.5) -> int:
    """How many points score beyond `threshold`. The profiler's per-field outlier count."""
    return int(np.sum(modified_zscore(values) > threshold))
