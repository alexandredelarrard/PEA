"""Tests for the workforce features
(src/data_aggregate/utils/employee_features.py).

The headcount series (fundamentals_history."employees", parsed from 10-K text)
IS historical, so we check: (a) point-in-time stepwise application from each
filing `as_of`, (b) year-over-year headcount growth math, and (c)
revenue-per-employee combining TTM revenue with the headcount.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.fundamentals.employee_features import (
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

    print("\n=== SANITY CHECK: workforce features (10-K headcount history) ===")
    print(f"  headcount 1,000 -> 1,200: YoY growth = {growth:+.1%} (expected +20%)")
    print(f"  revenue/employee = 600,000/1,200 = "
          f"{F['revenue_per_employee'].loc[after_second,'AAA']:.0f}")
    print("  Both NaN before the first filing -> historical & leak-free. Validated.")


def test_revenue_per_employee_growth():
    """rev/employee GROWTH = is revenue outgrowing headcount (productivity up) or
    scaling linearly with it (flat)? Past-vs-past, leak-free."""
    emp = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "as_of": [pd.Timestamp("2019-02-01"), pd.Timestamp("2020-02-03")],
        "employees": [1000.0, 1200.0],
    })
    fund = pd.DataFrame({
        "ticker": ["AAA", "AAA"], "as_of": ["2019-02-01", "2020-02-03"],
        "totalRevenue": [500_000.0, 720_000.0],   # rev/emp 500 -> 600 (+20%)
    })
    idx = pd.bdate_range("2018-06-01", "2020-06-01")
    F = _employee_fields(emp, idx, fund)
    after = pd.Timestamp("2020-03-02")
    assert "revenue_per_employee_growth" in F
    g = F["revenue_per_employee_growth"].loc[after, "AAA"]
    assert abs(g - 0.20) < 0.03, f"expected ~+20% rev/employee growth, got {g}"
    # leak-free: NaN before the first filing
    assert pd.isna(F["revenue_per_employee_growth"].loc[pd.Timestamp("2018-11-30"), "AAA"])
    print("\n=== SANITY CHECK: revenue-per-employee GROWTH ===")
    print(f"  rev/emp 500 -> 600 => growth {g:+.1%} (~+20%, revenue outrunning headcount). Validated.")


def test_rev_per_employee_growth_handles_inf_no_crash():
    """Regression: a zero prior-period revenue-per-employee makes the YoY growth inf,
    so the frame mixes inf with the NaN warmup. `DataFrame.replace([inf,-inf], pd.NA)`
    used to raise 'IndexError: pop index out of range' on exactly that shape (pandas
    3.x). Now inf -> NaN cleanly and _employee_fields must not raise."""
    # AAA: finite +20% rev/employee growth; ZZZ: 0 -> 5000 => +inf. The combined frame
    # is the finite + inf + NaN-warmup mix that triggered the old crash.
    emp = pd.DataFrame({
        "ticker": ["AAA", "AAA", "ZZZ", "ZZZ"],
        "as_of": [pd.Timestamp("2019-02-01"), pd.Timestamp("2020-02-03")] * 2,
        "employees": [1000.0, 1200.0, 100.0, 100.0],
    })
    fund = pd.DataFrame({
        "ticker": ["AAA", "AAA", "ZZZ", "ZZZ"],
        "as_of": ["2019-02-01", "2020-02-03", "2019-02-01", "2020-02-03"],
        "totalRevenue": [500_000.0, 720_000.0, 0.0, 500_000.0],   # AAA 500->600 (+20%); ZZZ 0->5000 (inf)
    })
    idx = pd.bdate_range("2018-06-01", "2020-06-01")

    F = _employee_fields(emp, idx, fund)         # must NOT raise IndexError
    assert "revenue_per_employee_growth" in F    # AAA's finite values keep the field alive
    g = F["revenue_per_employee_growth"]
    after = pd.Timestamp("2020-03-02")
    assert abs(g.loc[after, "AAA"] - 0.20) < 0.03            # finite ticker unaffected
    assert not np.isinf(g.to_numpy()).any(), "inf not scrubbed -> would poison the z-score"
    assert pd.isna(g.loc[after, "ZZZ"])                      # 5000/0 = inf -> NaN, no crash
    print("\n=== SANITY CHECK: rev/employee growth inf handling ===")
    print(f"  AAA finite +{g.loc[after,'AAA']:.0%}; ZZZ 0->5000 gives +inf -> scrubbed to NaN "
          "via replace([inf,-inf], np.nan) with NO IndexError (pandas 3.x). Validated.")


def test_build_panel_empty_without_history():
    idx = pd.bdate_range("2020-01-01", "2020-06-01")
    panel = build_employee_feature_panel(None, {"AAA": ["BBB"]}, idx)
    assert list(panel.columns) == ["date", "ticker"] and panel.empty

    print("\n=== SANITY CHECK: graceful skip ===")
    print("  No employee history -> empty (date,ticker) panel, no crash. Validated.")
