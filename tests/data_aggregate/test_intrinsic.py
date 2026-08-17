"""Tests for the intrinsic-value DCF (src/data_aggregate/utils/intrinsic.py).

Unit (known truth): the two-stage DCF must reproduce the closed-form value of a
constant-growth cash-flow stream, return NaN for cash-burning firms, and reject
a terminal growth >= the discount rate. Then the daily wrapper must produce a
point-in-time yield / per-share consistent with the market cap identity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.fundamentals.intrinsic import two_stage_dcf, intrinsic_value_daily


def _closed_form(fcf, g, r, gt, n):
    pv = sum(fcf * (1 + g) ** t / (1 + r) ** t for t in range(1, n + 1))
    tv = fcf * (1 + g) ** n * (1 + gt) / (r - gt)
    return pv + tv / (1 + r) ** n


def test_two_stage_dcf_matches_closed_form():
    idx = pd.bdate_range("2020-01-01", periods=1)
    base = pd.DataFrame({"AAA": [100.0], "BBB": [250.0]}, index=idx)
    growth = pd.DataFrame({"AAA": [0.0], "BBB": [0.08]}, index=idx)
    r, gt, n = 0.10, 0.025, 5

    v = two_stage_dcf(base, growth, r, gt, n)
    exp_a = _closed_form(100.0, 0.0, r, gt, n)
    exp_b = _closed_form(250.0, 0.08, r, gt, n)
    assert abs(v.loc[idx[0], "AAA"] - exp_a) < 1e-6
    assert abs(v.loc[idx[0], "BBB"] - exp_b) < 1e-6

    print("\n=== SANITY CHECK: two-stage DCF closed form ===")
    print(f"  FCF100 g=0%%  -> V={v.loc[idx[0],'AAA']:.2f} (closed form {exp_a:.2f})")
    print(f"  FCF250 g=8%%  -> V={v.loc[idx[0],'BBB']:.2f} (closed form {exp_b:.2f})")
    print("  DCF reproduces the analytic value exactly.")


def test_two_stage_dcf_nan_for_cash_burners():
    idx = pd.bdate_range("2020-01-01", periods=1)
    base = pd.DataFrame({"AAA": [-50.0], "BBB": [0.0]}, index=idx)
    growth = pd.DataFrame({"AAA": [0.05], "BBB": [0.05]}, index=idx)
    v = two_stage_dcf(base, growth, 0.10, 0.025, 5)
    assert np.isnan(v.loc[idx[0], "AAA"]) and np.isnan(v.loc[idx[0], "BBB"])
    print("\n=== SANITY CHECK: cash-burning firms -> NaN intrinsic ===")
    print("  FCF<=0 has no cash-flow intrinsic value -> NaN, never a bogus number.")


def test_two_stage_dcf_rejects_terminal_above_discount():
    idx = pd.bdate_range("2020-01-01", periods=1)
    base = pd.DataFrame({"AAA": [100.0]}, index=idx)
    with pytest.raises(ValueError):
        two_stage_dcf(base, base * 0, discount_rate=0.03, terminal_growth=0.05, years=5)
    print("\n=== SANITY CHECK: r <= g_term rejected ===")
    print("  a perpetuity needs r > g_term -> ValueError, no divide-by-<=0.")


def test_intrinsic_value_daily_pit_and_identity():
    idx = pd.bdate_range("2020-01-01", "2020-06-01")
    fund = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "as_of": ["2019-06-03", "2020-03-02"],
        "freeCashflow": [80.0, 100.0],
        "revenueGrowth": [0.05, 0.05],
        "sharesOutstanding": [1000.0, 1000.0],
    })
    close = pd.DataFrame({"AAA": 2.0}, index=idx)

    out = intrinsic_value_daily(fund, close, idx, discount_rate=0.10,
                                terminal_growth=0.025, years=5,
                                growth_cap=0.15, growth_floor=-0.10)
    d = pd.Timestamp("2020-04-01")
    exp_total = _closed_form(100.0, 0.05, 0.10, 0.025, 5)
    assert abs(out["total"].loc[d, "AAA"] - exp_total) < 1e-6
    # per-share = total / shares ; yield = total / (shares*price) = per_share/price
    assert abs(out["per_share"].loc[d, "AAA"] - exp_total / 1000.0) < 1e-6
    assert abs(out["yield"].loc[d, "AAA"] - exp_total / (1000.0 * 2.0)) < 1e-6

    print("\n=== SANITY CHECK: intrinsic_value_daily (PIT) ===")
    print(f"  total={out['total'].loc[d,'AAA']:.1f}  per_share="
          f"{out['per_share'].loc[d,'AAA']:.4f}  yield={out['yield'].loc[d,'AAA']:.4f}")
    print("  total/shares == per_share and total/mcap == yield (identity holds),")
    print("  and the value is keyed on the 2020-03-02 filing -> point-in-time.")
