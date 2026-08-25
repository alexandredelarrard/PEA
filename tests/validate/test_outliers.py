"""
test_outliers.py  (tests/validate/)
--------------------------------------------------------------------------------------------
The guard on plan-5b decision 60: `detect_level_outliers` scores a QoQ LOG CHANGE, so a
smoothly growing company is not flagged for growing.

The named test the plan asks for is `test_smooth_compound_growth_is_silent` +
`test_planted_spike_is_the_only_finding`. The raw-level kernel this replaced fails the first
outright -- a 10x compounding series puts its whole recent era many MADs above its own median
-- and the second is what proves the fix did not simply blind the check.

Everything here is synthetic known-truth parsing math, which is the one place AGENTS.md
allows a fixture instead of real data: the question is whether an arithmetic rule fires where
it should, not whether a filer's numbers are right.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validate.outliers import (
    count_mad_outliers, detect_level_outliers, log_change, mad_center_scale, modified_zscore)

QUARTERS = 60


def _series(values: list[float] | np.ndarray, ticker: str = "TEST",
            field: str = "totalRevenue") -> pd.DataFrame:
    """A `detect_level_outliers`-shaped frame: one quarterly fact per period, in order."""
    n = len(values)
    return pd.DataFrame({
        "ticker": ticker,
        "field": field,
        "fiscal_year": [2010 + i // 4 for i in range(n)],
        "fiscal_period": [f"Q{i % 4 + 1}" for i in range(n)],
        "duration_type": "quarterly",
        "filing_date": pd.date_range("2010-05-01", periods=n, freq="QE"),
        "value": np.asarray(values, dtype=float),
        "source_tag": "us-gaap:Revenues",
        "is_amendment": False,
        "derived": False,
    })


def _smooth_10x(n: int = QUARTERS) -> np.ndarray:
    """A perfectly smooth 10x compounding series: the shape the raw-level kernel destroyed."""
    return 1_000.0 * np.power(10.0, np.arange(n) / (n - 1))


# --------------------------------------------------------------------------- #
# decision 60: the two named tests                                             #
# --------------------------------------------------------------------------- #

def test_smooth_compound_growth_is_silent() -> None:
    """A 10x compound-growth series over 60 quarters produces ZERO findings, level AND YoY."""
    out = detect_level_outliers(_series(_smooth_10x()), "TEST", "totalRevenue")

    level, yoy = int(out["is_level_outlier"].sum()), int(out["is_yoy_outlier"].sum())
    print(f"\nsmooth 10x over {QUARTERS} quarters "
          f"({out['value'].iloc[0]:,.0f} -> {out['value'].iloc[-1]:,.0f}): "
          f"level findings={level}, YoY findings={yoy}, "
          f"max |z|={np.nanmax(out['level_z_score'].values):.2f}")
    print("  SANITY: a company that merely GREW is not an outlier -- the raw-level kernel "
          "this replaced flagged its entire recent era.")
    assert level == 0, f"{level} level findings on a perfectly smooth growth series"
    assert yoy == 0, f"{yoy} YoY findings on a perfectly smooth growth series"


def test_planted_spike_is_the_only_finding() -> None:
    """The same series with ONE planted 3x spike flags exactly the spike and its reversion."""
    values = _smooth_10x()
    spike_at = 40
    values[spike_at] *= 3.0

    out = detect_level_outliers(_series(values), "TEST", "totalRevenue")
    flagged = sorted(np.flatnonzero(out["is_level_outlier"].values).tolist())

    print(f"\nsame series, 3x spike planted at index {spike_at}: level findings at {flagged}")
    print(f"  z at the spike={out['level_z_score'].iloc[spike_at]:.1f}, "
          f"at the reversion={out['level_z_score'].iloc[spike_at + 1]:.1f}")
    print("  SANITY: the STEP up and the STEP back down are both anomalous, so two rows is "
          "the correct answer -- one defect, its two boundaries.")
    assert flagged == [spike_at, spike_at + 1], \
        f"expected the spike and its reversion, got {flagged}"


# --------------------------------------------------------------------------- #
# the transform's declared abstentions                                         #
# --------------------------------------------------------------------------- #

def test_log_change_abstains_across_zero_and_sign_flips() -> None:
    """No ratio is a growth rate across a zero or a sign change: those score NaN, never a hit."""
    values = np.array([100.0, 0.0, 100.0, -100.0, -200.0, -400.0])
    changes = log_change(values)

    print(f"\nlog_change({values.tolist()}) = "
          f"{[('nan' if not np.isfinite(c) else round(float(c), 3)) for c in changes]}")
    print("  SANITY: index 0 has no prior; 1 and 2 touch a zero; 3 flips sign -- all NaN. "
          "4 and 5 are two NEGATIVES, scored on their magnitude ratio (log 2 = 0.693).")
    assert not np.isfinite(changes[:4]).any(), "an undefined ratio was scored"
    assert np.allclose(changes[4:], np.log(2.0)), "two negatives must score on |ratio|"


def test_zero_series_and_short_series_produce_nothing() -> None:
    """VRT's all-zero pre-merger shell, and a series too short to have a distribution."""
    zeros = detect_level_outliers(_series([0.0] * 12), "TEST", "totalRevenue")
    short = detect_level_outliers(_series([1.0, 2.0]), "TEST", "totalRevenue")

    print(f"\nall-zero 12-quarter series: {int(zeros['is_level_outlier'].sum())} findings; "
          f"2-quarter series: {len(short)} rows returned")
    print("  SANITY: a correct constant zero is not an anomaly, and 2 points are not a "
          "distribution -- both abstain rather than guess.")
    assert int(zeros["is_level_outlier"].sum()) == 0
    assert short.empty


# --------------------------------------------------------------------------- #
# the statistical kernel scripts/dod/data_profile.py consumes is UNCHANGED     #
# --------------------------------------------------------------------------- #

def test_mad_kernel_is_untouched_by_decision_60() -> None:
    """`mad_center_scale` / `modified_zscore` / `count_mad_outliers` keep their old contract.

    `scripts/dod/data_profile.py` imports those three directly and feeds them its own
    columns; decision 60 changes only what `detect_level_outliers` feeds them."""
    values = pd.Series([10.0, 11.0, 10.5, 10.2, 10.8, 90.0])
    median, scale = mad_center_scale(values)
    scores = modified_zscore(values)

    print(f"\nmad_center_scale -> median={median}, scale={scale}; "
          f"z of the planted 90.0 = {scores[-1]:.1f}; "
          f"count_mad_outliers = {count_mad_outliers(values)}")
    print("  SANITY: the profiler's kernel still scores RAW levels and still catches a lone "
          "9x spike -- that is the right rule for an arbitrary column with no time order.")
    assert median == 10.65 and scale > 0
    assert count_mad_outliers(values) == 1
