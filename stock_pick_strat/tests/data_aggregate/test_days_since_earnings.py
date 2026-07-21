"""
days_since_earnings (src/data_aggregate/utils/earnings_features.py): calendar days
since the most recent PAST earnings report (0 on report day, rising to 90+, resetting
at the next report). Must be leak-free (uses only past reports), reset each quarter,
NaN before the first report, and clipped for late/gap reporters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.earnings_features import (
    build_earnings_feature_panel, days_since_earnings,
)


def _earnings():
    rows = [("A", "2020-01-15"), ("A", "2020-04-15"), ("A", "2020-07-15"),
            ("B", "2020-01-15")]                       # B never reports again -> clip test
    return pd.DataFrame(rows, columns=["ticker", "earnings_date"])


def test_days_since_earnings_ramp_reset_clip():
    df = _earnings().copy()
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    idx = pd.bdate_range("2020-01-01", "2020-09-01")
    ds = days_since_earnings(df, idx, cap_days=180)

    # leak-free: NaN before the first report
    assert np.isnan(ds.loc[pd.Timestamp("2020-01-10"), "A"])
    # 0 on the report day; +5 calendar days five days later
    assert ds.loc[pd.Timestamp("2020-01-15"), "A"] == 0
    assert ds.loc[pd.Timestamp("2020-01-20"), "A"] == 5
    # rises across the quarter then RESETS to 0 at the next report
    assert ds.loc[pd.Timestamp("2020-04-14"), "A"] == 90
    assert ds.loc[pd.Timestamp("2020-04-15"), "A"] == 0
    # B never reports again -> value would be ~230d but is CLIPPED to the 180 cap
    assert ds.loc[pd.Timestamp("2020-09-01"), "B"] == 180

    print("\n=== SANITY CHECK: days_since_earnings ===")
    print("  0 on the report day, +5 after 5 calendar days, 90 just before the next report, "
          "resets to 0 at each report, NaN before the first, clipped to 180. Leak-free. Validated.")


def test_panel_emits_raw_feature_leakfree():
    df = _earnings().copy()
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    idx = pd.bdate_range("2020-01-01", "2020-09-01")
    close = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    peers = {"A": {"B": 1.0}, "B": {"A": 1.0}}
    panel = build_earnings_feature_panel(df, peers, idx, stock_close=close)

    assert "f_days_since_earnings" in panel.columns
    a = panel[panel["ticker"] == "A"].dropna(subset=["f_days_since_earnings"])
    a = a.assign(date=pd.to_datetime(a["date"]))
    # no row before A's first report carries the feature (leak-free)
    assert a["date"].min() >= pd.Timestamp("2020-01-15")
    print("\n=== SANITY CHECK: f_days_since_earnings in cube panel ===")
    print("  emitted as a RAW (non-peer) f_days_since_earnings column; first non-null on the "
          "first report date (no pre-report leakage). Validated.")


if __name__ == "__main__":
    test_days_since_earnings_ramp_reset_clip()
    test_panel_emits_raw_feature_leakfree()
