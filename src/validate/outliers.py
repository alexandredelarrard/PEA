"""
outliers.py  (src/validate/outliers.py)
--------------------------------------------------------------------------------------------
The MAD-based modified Z-score and the one `detect_level_outliers` built on it, shared by
every place that needs to judge whether a number sits far from its own history: the
`FundamentalsValidator`'s `level_outlier` check, the external-source ratio checks in
`external/tiingo_comparison.py` / `external/yahoo_comparison.py`, and the Definition-of-Done
data profiler (`scripts/dod/data_profile.py`).

Moved here from `src/utils/` by plan-5b decision 62: `src/validate/` is the ONE home for
validation code across every domain, and `utils` is where code goes when nobody decided.

Why MAD and not a standard deviation: a single 10x mis-tagged XBRL fact inflates the standard
deviation enough to hide itself. The median and the median absolute deviation do not move, so
the spike still scores far from the centre. That is the whole reason this repo flags level
outliers this way, and it is why every caller must use the SAME kernel -- two subtly different
"outlier counts" for the same column would be worse than one.

## THE KERNEL SCORES A LOG CHANGE, NOT A LEVEL (plan-5b decision 60)

`detect_level_outliers` used to score `modified_zscore(raw levels)`. That rule flags the
entire recent era of any growing company: a filer whose revenue compounds 10x over 15 years
has a median sitting near its mid-history level, and every recent quarter is many MADs above
it -- correctly, and uselessly. The measurement is in `tests/validate/test_outliers.py`: a
smooth 10x compound-growth series over 60 quarters produced findings on the raw-level kernel
and produces ZERO here.

So both passes score a SCALE-FREE log change:

  * the level pass scores `log(v_t / v_{t-1})` -- "is this STEP unlike this field's other
    steps?", which is the question the check was always described as asking;
  * the YoY pass scores `log(v_t / v_{t-4})` rather than the old `diff(4)`. Decision 60 says
    to keep the YoY check, and this keeps it: a year-on-year comparison immune to seasonality.
    Its kernel had the SAME defect as the level pass -- a 4-period difference on an
    exponentially growing series grows exponentially too -- so fixing one and not the other
    would have left half the bug in place. Recorded as a deviation from the letter of
    decision 60 (which says only "keep the YoY `diff(4)` check") for exactly that reason.

## WHERE IT ABSTAINS -- read this before trusting a zero

A log ratio does not exist across a sign change or a zero. `log_change` returns NaN when
either endpoint is <= 0 or when the two disagree in sign, and NaN scores as NaN, which
compares False against every threshold. So this check is BLIND to:

  * a value crossing zero (APA's revenue going to 0 / -$467M) -- that is `impossible_value`
    and `sign_convention`, not this;
  * a series that is zero throughout (VRT's pre-merger shell) -- no dispersion, no finding;
  * the first period (level pass) and the first four (YoY pass), which have no prior.

Two negatives are scored on their magnitude ratio, because a liability line tagged negative
throughout has a perfectly well-defined series of steps.

Design notes
  * `reference` EXISTS FOR THE LAGGED PATHS. Both passes score a lagged quantity whose first
    entries are undefined. Those entries must be scored (as NaN, so they can never be
    flagged) but must NOT contribute to the median/MAD -- filling them with 0 first was a
    real false-positive bug, since a 0 in a series of large changes is itself an outlier.
    So: statistics from `reference`, scores over `values`.
  * `fallback_to_mean_abs_dev` IS NOT COSMETIC. When MAD == 0 (more than half the series is
    identical -- common for a flat balance-sheet line) the level check falls back to the mean
    absolute deviation so a genuine lone spike is still caught. The YoY check deliberately
    does NOT: a zero MAD there means the changes are essentially constant, and every caller
    of that path treats "no dispersion" as "nothing to say".
  * Returns NaN where the input is NaN, and zeros when there is no dispersion at all. Both
    compare False against any threshold, so a caller's `z > threshold` mask is safe without
    any extra NaN handling.
  * `mad_center_scale` / `modified_zscore` / `count_mad_outliers` are UNCHANGED. They are the
    statistical kernel `scripts/dod/data_profile.py` consumes directly, on its own columns;
    decision 60 touches only what `detect_level_outliers` feeds them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Consistency constant for the modified Z-score. Kept as the value this repo's audit has
#: always used, so historical flags stay reproducible.
MODIFIED_Z_SCALE: float = 0.6745

#: A dispersion below this FRACTION of the series' own magnitude is floating-point dust, not
#: dispersion, and must be treated as zero.
#:
#: `mad > 0` alone is not that test. Measured while writing `tests/validate/test_outliers.py`:
#: a smooth compounding series has log changes that are equal to within ~1e-17, so more than
#: half the deviations from the median are 1e-17 rather than exactly 0 -- `mad > 0` accepted
#: that as the scale and a planted 3x spike scored z = 3.6e15. The number was still "right"
#: (far beyond any threshold) but it was decided by rounding error, and a series whose dust
#: happened to cancel would have scored the same spike at 1.4. Anything at 1e-12 of the
#: series' own size is below float64's ability to say otherwise.
_DISPERSION_REL_TOL: float = 1e-12


def _is_dispersion(scale: float, arr: np.ndarray, median: float) -> bool:
    """Is `scale` real dispersion, or rounding error? See `_DISPERSION_REL_TOL`."""
    magnitude = max(abs(median), float(np.max(np.abs(arr))) if arr.size else 0.0)
    return scale > _DISPERSION_REL_TOL * max(magnitude, 1.0)


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
    if _is_dispersion(mad, arr, median):
        return (median, mad)
    if fallback_to_mean_abs_dev:
        mean_abs_dev = float(np.mean(np.abs(arr - median)))
        if _is_dispersion(mean_abs_dev, arr, median):
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


def log_change(values: np.ndarray | pd.Series, *, lag: int = 1) -> np.ndarray:
    """`log(v_t / v_{t-lag})` as a float array the length of `values`, NaN where undefined.

    THE scale-free transform decision 60 puts under both outlier passes. Undefined -- and so
    NaN, which can never be flagged -- for the first `lag` entries, for any endpoint that is
    zero or NaN, and across a SIGN CHANGE, where no ratio is a growth rate. Two negative
    endpoints score on their magnitude ratio: a liability line tagged negative throughout has
    a perfectly well-defined series of steps, and refusing it would blind the check to an
    entire class of correctly-signed field.

    Why this and not a percentage change: `(v_t - v_{t-1}) / v_{t-1}` is asymmetric -- a
    halving is -50% and the doubling that undoes it is +100% -- so a spike and its reversion
    score differently and the MAD is pulled by whichever direction the field happens to move
    in more often. In logs they are exactly +/-0.693.
    """
    arr = np.asarray(pd.Series(values).astype(float).values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if arr.size <= lag:
        return out
    current, prior = arr[lag:], arr[:-lag]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(current) / np.abs(prior)
        defined = (np.sign(current) == np.sign(prior)) & (prior != 0) & (current != 0)
        defined &= np.isfinite(current) & np.isfinite(prior)
        out[lag:] = np.where(defined, np.log(np.where(defined, ratio, 1.0)), np.nan)
    return out


def _score_changes(changes: np.ndarray, *, fallback_to_mean_abs_dev: bool) -> np.ndarray:
    """Modified Z over a change series, statistics taken from its DEFINED entries only.

    The undefined head (and any interior abstention) must be scored so the result lines up
    with the input row-for-row, but must not enter the median/MAD -- see the module's
    `reference` design note for the false-positive bug that came of doing it the other way.
    """
    defined = changes[np.isfinite(changes)]
    if defined.size < 3:
        return np.full(changes.shape, np.nan, dtype=float)
    scores = modified_zscore(changes, reference=defined,
                             fallback_to_mean_abs_dev=fallback_to_mean_abs_dev)
    return np.where(np.isfinite(changes), scores, np.nan)


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

    BOTH passes score a LOG CHANGE (decision 60), not a level: `level_z_score` is the modified
    Z of `log(v_t / v_{t-1})` and the YoY flag is the same at lag 4. "Level outlier" is
    retained as the NAME because it is what the finding means to a reader -- this value's
    level is out of line with its own history -- but the statistic is a step, so a smoothly
    compounding series scores flat instead of flagging its whole recent era.

    Each pass excludes the periods whose lag is undefined: filling them with 0 before scoring
    was a real false-positive bug, since a 0 in a series of large changes is itself an
    outlier. `log_change` also abstains across a sign change and at a zero, so a value
    crossing zero is INVISIBLE here by construction -- see the module docstring.

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
    # Decision 60: the QoQ LOG CHANGE, not the raw level. See the module docstring for the
    # 10x-growth measurement that retired the raw-level kernel, and for where this abstains.
    modified_z = _score_changes(log_change(vals, lag=1), fallback_to_mean_abs_dev=True)
    level_outlier = modified_z > threshold        # NaN compares False: abstentions never fire

    yoy_outlier = np.zeros(len(sub), dtype=bool)
    if check_yoy and len(sub) >= 5:
        # The same transform at lag 4 -- a year-on-year comparison, so a seasonal filer's Q4
        # is measured against its own Q4. No mean-abs-dev fallback: a zero MAD on the annual
        # changes means "nothing to say", and every caller of this path treats it that way.
        yoy_z = _score_changes(log_change(vals, lag=4), fallback_to_mean_abs_dev=False)
        yoy_outlier = yoy_z > threshold

    out = sub.copy()
    out["is_level_outlier"] = level_outlier
    out["level_z_score"] = modified_z
    out["is_yoy_outlier"] = yoy_outlier
    return out[cols]
