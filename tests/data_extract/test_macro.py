"""Macro data-quality fix: short-gap mean-fill (holiday gaps, < 1 week only). The credit
spread is now the single real Moody's Baa-10Y series (`baa_credit_spread`), so there is no
backfill/model to test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_extract.utils.fundamentals.fetch_macro import fill_short_gaps


def test_fill_short_gaps_mean_and_week_guard():
    """A short interior gap (< 1 week) is filled with the MEAN of the two bracketing days;
    a >= 1 week gap and leading/trailing NaNs are left untouched."""
    idx = pd.date_range("2024-01-01", "2024-01-20", freq="D")
    s = pd.Series(np.nan, index=idx)
    s["2024-01-01"] = 10.0            # valid
    # 01-02, 01-03 missing -> gap span 01-01..01-04 = 3 days (< 7) -> fill mean(10,16)=13
    s["2024-01-04"] = 16.0            # valid
    # 01-05 .. 01-14 missing -> gap span 01-04..01-15 = 11 days (>= 7) -> NOT filled
    s["2024-01-15"] = 20.0            # valid
    s["2024-01-16":] = np.nan         # trailing NaNs -> untouched
    df = pd.DataFrame({"x": s})

    out = fill_short_gaps(df, ["x"], max_gap_days=7)
    assert out.loc["2024-01-02", "x"] == pytest.approx(13.0)   # mean(10, 16)
    assert out.loc["2024-01-03", "x"] == pytest.approx(13.0)
    assert pd.isna(out.loc["2024-01-08", "x"])                 # long gap left NaN
    assert pd.isna(out.loc["2024-01-18", "x"])                 # trailing NaN untouched
    assert out.loc["2024-01-01", "x"] == 10.0 and out.loc["2024-01-15", "x"] == 20.0
    print("\n=== SANITY CHECK: short-gap mean fill ===")
    print("  2-day gap filled with mean(10,16)=13; 11-day gap + trailing NaNs untouched. Validated.")
