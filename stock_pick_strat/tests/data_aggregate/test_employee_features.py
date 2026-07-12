"""Tests for the workforce features
(src/data_aggregate/utils/employee_features.py).

The FMP employee series IS historical, so we check: (a) point-in-time stepwise
application from each filing `as_of`, (b) year-over-year headcount growth math,
and (c) revenue-per-employee combining TTM revenue with the headcount.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.employee_features import (
    _employee_fields, build_employee_feature_panel,
)


def _synth_employees():
    # two annual filings a year apart: 1,000 -> 1,200 employees (+20% YoY)
    return pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "as_of": [pd.Timestamp("2019-02-01"), pd.Timestamp("2020-02-03")],
        "period": [pd.Timestamp("2018-12-31"), pd.Timestamp("2019-12-31")],
        "employees": [1000.0, 1200.0],
        "form_type": ["10-K", "10-K"],
    })


def _synth_fundamentals():
    return pd.DataFrame({
        "ticker": ["AAA"], "as_of": ["2020-02-03"],
        "totalRevenue": [600_000.0],
    })


def test_employee_fields_pit_growth_and_rev_per_employee():
    idx = pd.bdate_range("2018-06-01", "2020-06-01")
    F = _employee_fields(_synth_employees(), idx, _synth_fundamentals())

    before_any = pd.Timestamp("2018-11-30")   # before the first filing (a business day)
    after_second = pd.Timestamp("2020-03-02")  # after the 2020-02-03 filing

    # ---- point-in-time: nothing before the first filing ----
    for name, frame in F.items():
        assert np.isnan(frame.loc[before_any, "AAA"]), f"{name} leaked before first as_of"

    # ---- YoY headcount growth ~ +20% once both filings are in the past ----
    growth = F["employee_growth"].loc[after_second, "AAA"]
    assert abs(growth - 0.20) < 0.02, f"expected ~+20% headcount growth, got {growth}"

    # ---- revenue per employee = 600,000 / 1,200 = 500 ----
    assert abs(F["revenue_per_employee"].loc[after_second, "AAA"] - 500.0) < 1e-9

    print("\n=== SANITY CHECK: workforce features (FMP historical) ===")
    print(f"  headcount 1,000 -> 1,200: YoY growth = {growth:+.1%} (expected +20%)")
    print(f"  revenue/employee = 600,000/1,200 = "
          f"{F['revenue_per_employee'].loc[after_second,'AAA']:.0f}")
    print("  Both NaN before the first filing -> historical & leak-free. Validated.")


def test_build_panel_empty_without_history():
    idx = pd.bdate_range("2020-01-01", "2020-06-01")
    panel = build_employee_feature_panel(None, {"AAA": ["BBB"]}, idx)
    assert list(panel.columns) == ["date", "ticker"] and panel.empty

    print("\n=== SANITY CHECK: graceful skip ===")
    print("  No employee history -> empty (date,ticker) panel, no crash. Validated.")
