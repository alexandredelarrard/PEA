"""Calendar age follows actual-bearing reports on their next observable session."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.fundamentals.earnings_features import build_earnings_feature_panel, days_since_earnings


def _earnings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "B"],
            "earnings_date": ["2020-01-15", "2020-04-15", "2020-07-15", "2020-01-15"],
            "eps_actual": [1.0, 0.0, np.nan, 1.0],
        }
    )


def test_days_since_earnings_ramp_reset_clip() -> None:
    idx = pd.bdate_range("2020-01-01", "2020-09-01")
    days = days_since_earnings(_earnings(), idx, cap_days=180)
    assert pd.isna(days.loc["2020-01-15", "A"])
    assert days.loc["2020-01-16", "A"] == 1
    assert days.loc["2020-01-20", "A"] == 5
    assert days.loc["2020-04-15", "A"] == 91
    assert days.loc["2020-04-16", "A"] == 1
    assert days.loc["2020-07-16", "A"] == 92
    assert days.loc["2020-09-01", "B"] == 180
    print("Sanity: report age uses elapsed calendar days, updates next session, accepts zero actual and ignores scheduled-only events.")


def test_panel_emits_raw_feature_leakfree() -> None:
    idx = pd.bdate_range("2020-01-01", "2020-09-01")
    df_close = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
    peers = {"A": {"B": 1.0}, "B": {"A": 1.0}}
    panel = build_earnings_feature_panel(_earnings(), peers, idx, stock_close=df_close)
    df_a = panel.loc[panel["ticker"].eq("A")].dropna(subset=["f_days_since_earnings"])
    assert pd.to_datetime(df_a["date"]).min() == pd.Timestamp("2020-01-16")
    assert "f_days_since_earnings_vs_peers" not in panel
    print("Sanity: the raw report-age column first appears on the next session after an actual-bearing release.")
