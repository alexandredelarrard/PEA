"""
outliers.py  (src/utils/outliers.py)
------------------------------------
The MAD-based modified Z-score and the one `detect_level_outliers` built on it, shared by
every place that needs to judge whether a number sits far from its own history: the
`FundamentalsValidator`'s `level_outlier` check, the external-source ratio checks in
`tiingo_comparison.py` / `yahoo_comparison.py`, and the Definition-of-Done data profiler
(`scripts/dod/data_profile.py`).

Why MAD and not a standard deviation: a single 10x mis-tagged XBRL fact inflates the standard
deviation enough to hide itself. The median and the median absolute deviation do not move, so
the spike still scores far from the centre. That is the whole reason this repo flags level
outliers this way, and it is why every caller must use the SAME kernel -- two subtly different
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


# --------------------------------------------------------------------------- #
# level / YoY outlier detection over a (ticker, field) time series             #
# --------------------------------------------------------------------------- #
#: Ordinal position of each fiscal-period label. Tiny and stable, so it lives here rather
#: than being imported from the extraction package -- this module must stay free of
#: `src/` cross-package imports.
_FISCAL_PERIOD_ORDER: dict[str, int] = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4}

#: Columns `detect_level_outliers` returns, in order. Declared once so the empty-result
#: path and the populated path cannot drift apart.
LEVEL_OUTLIER_COLUMNS: tuple[str, ...] = (
    "ticker", "field", "fiscal_year", "fiscal_period", "duration_type", "filing_date",
    "value", "source_tag", "is_amendment", "derived", "is_level_outlier",
    "level_z_score", "is_yoy_outlier",
)


def _latest_per_period(sub: pd.DataFrame) -> pd.DataFrame:
    """Collapse to ONE row per (fiscal_year, fiscal_period, duration_type): the LATEST-filed
    value. An amendment coexisting as its own fact row must not look like a second,
    disagreeing observation of the same period."""
    return (sub.sort_values("filing_date")
            .drop_duplicates(subset=["fiscal_year", "fiscal_period", "duration_type"],
                             keep="last"))


def _chronological_sort(sub: pd.DataFrame) -> pd.DataFrame:
    """Order strictly by (fiscal_year, fiscal_period-ordinal) -- NOT filing_date, which is only
    a proxy and can be out of step with the true fiscal sequence (a late amendment, a delayed
    filing). This is what makes the 4-quarter YoY lag and the level comparison meaningful."""
    sub = sub.copy()
    sub["_fp_ord"] = sub["fiscal_period"].map(_FISCAL_PERIOD_ORDER).fillna(99)
    return sub.sort_values(["fiscal_year", "_fp_ord"]).drop(columns="_fp_ord")


def detect_level_outliers(
    df: pd.DataFrame,
    ticker: str,
    field: str,
    *,
    duration_type: str = "quarterly",
    threshold: float = 3.5,
    check_yoy: bool = True,
) -> pd.DataFrame:
    """Modified Z-score level-outlier + YoY-shift-anomaly detection for ONE (ticker, field)'s
    time series, scoped to a SINGLE `duration_type`.

    Quarterly figures and the annual total are different scales -- never mixed in one
    statistical pass; call twice to cover both.

    The YoY check excludes the first (up to) 4 periods, whose 4-period-lagged difference is
    undefined: filling them with 0 before scoring was a real false-positive bug, since a 0 in
    a series of large diffs is itself an outlier.

    Deliberately does NOT judge `source_tag` consistency. Comparing each row's tag against a
    single global mode cannot distinguish a clean, permanent taxonomy transition (every ticker
    with enough history has one, e.g. the ASC-606 cutover) from a genuine anomaly -- it flags
    every period in whichever era has fewer rows. Tag-consistency is the validator's
    era-aware `tag_switch_break` check instead.

    Returns one row per PERIOD, already deduplicated to the latest-filed value, so a caller
    can filter or aggregate across many (ticker, field) pairs.
    """
    cols = list(LEVEL_OUTLIER_COLUMNS)
    sub = df.loc[
        (df["ticker"] == ticker) & (df["field"] == field) & (df["duration_type"] == duration_type)
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)

    sub = _latest_per_period(sub)
    sub = _chronological_sort(sub).reset_index(drop=True)
    if len(sub) < 3:
        return pd.DataFrame(columns=cols)

    vals = sub["value"].astype(float).values
    modified_z = modified_zscore(vals)
    level_outlier = modified_z > threshold

    yoy_outlier = np.zeros(len(sub), dtype=bool)
    if check_yoy and len(sub) >= 5:
        yoy_change = sub["value"].astype(float).diff(4)
        has_yoy = yoy_change.notna()   # first (up to) 4 periods: nothing to lag against
        yoy_vals = yoy_change[has_yoy].values
        if len(yoy_vals) >= 3:
            # Statistics from the DEFINED diffs only, scores over the whole series; and no
            # mean-abs-dev fallback here -- a zero MAD on the diffs means "nothing to say".
            yoy_z = modified_zscore(yoy_change, reference=yoy_vals,
                                    fallback_to_mean_abs_dev=False)
            yoy_outlier = (has_yoy.values & (yoy_z > threshold))

    out = sub.copy()
    out["is_level_outlier"] = level_outlier
    out["level_z_score"] = modified_z
    out["is_yoy_outlier"] = yoy_outlier
    return out[cols]
