"""Tests for the LATEST-QUARTER momentum features
(src/data_aggregate/utils/fundamental_features.py: _fiscal_apply_to_daily and
the discrete single-quarter characteristics in _derived_fields).

These capture the acceleration/inflection that the TTM series smooths away, and
must be point-in-time (a value only appears once its filing is public).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.fundamental_features import (
    _fiscal_apply_to_daily,
    _derived_fields,
)


def _synth_quarterly():
    """6 fiscal quarters for one ticker, discrete single-quarter revenue/income
    plus the TTM level fields _derived_fields expects."""
    ends = pd.bdate_range("2019-03-01", periods=6, freq="63D")  # ~quarterly
    rev_q = [100.0, 110.0, 120.0, 130.0, 121.0, 143.0]
    ni_q = [10.0, 12.0, 14.0, 16.0, 13.0, 20.0]
    rows = []
    for i, e in enumerate(ends):
        ttm_rev = sum(rev_q[max(0, i - 3):i + 1])
        ttm_ni = sum(ni_q[max(0, i - 3):i + 1])
        rows.append({
            "ticker": "AAA", "as_of": e.date().isoformat(),
            "totalRevenue": ttm_rev, "netIncome": ttm_ni,
            "profitMargins": ttm_ni / ttm_rev,
            "revenue_q": rev_q[i], "netIncome_q": ni_q[i],
        })
    return pd.DataFrame(rows), ends, rev_q, ni_q


def test_fiscal_apply_yoy_and_acceleration():
    fund, ends, rev_q, _ = _synth_quarterly()
    idx = pd.bdate_range("2019-01-01", "2021-06-01")

    yoy = _fiscal_apply_to_daily(fund, "revenue_q", idx, lambda s: s.pct_change(4))
    accel = _fiscal_apply_to_daily(fund, "revenue_q", idx,
                                   lambda s: s.pct_change(4).diff(1))

    after_q5 = ends[4] + pd.Timedelta(days=5)   # 5th quarter public
    after_q6 = ends[5] + pd.Timedelta(days=5)   # 6th quarter public
    yoy_q5 = rev_q[4] / rev_q[0] - 1             # 121/100 - 1 = 0.21
    yoy_q6 = rev_q[5] / rev_q[1] - 1             # 143/110 - 1 = 0.30

    assert abs(yoy.loc[after_q5, "AAA"] - yoy_q5) < 1e-9
    assert abs(yoy.loc[after_q6, "AAA"] - yoy_q6) < 1e-9
    assert abs(accel.loc[after_q6, "AAA"] - (yoy_q6 - yoy_q5)) < 1e-9
    # no look-ahead: before the very first year-over-year is computable -> NaN
    assert np.isnan(yoy.loc[ends[0] + pd.Timedelta(days=5), "AAA"])

    print("\n=== SANITY CHECK: latest-quarter YoY + acceleration ===")
    print(f"  Q5 YoY={yoy.loc[after_q5,'AAA']:.2%} (121/100), "
          f"Q6 YoY={yoy.loc[after_q6,'AAA']:.2%} (143/110)")
    print(f"  acceleration Q6 = {accel.loc[after_q6,'AAA']:+.2%} = Q6 YoY - Q5 YoY.")
    print("  NaN before the first YoY is computable -> point-in-time, no look-ahead.")


def test_latest_quarter_margin_inflection_exact():
    fund, ends, rev_q, ni_q = _synth_quarterly()
    idx = pd.bdate_range("2019-01-01", "2021-06-01")

    F = _derived_fields(fund, idx, close=None, yoy_periods=4)
    assert "q_margin_vs_ttm" in F and "q_rev_growth" in F

    d = ends[5] + pd.Timedelta(days=5)          # after Q6 filing
    q_margin = ni_q[5] / rev_q[5]               # 20/143
    ttm_margin = sum(ni_q[2:6]) / sum(rev_q[2:6])
    assert abs(F["q_margin_vs_ttm"].loc[d, "AAA"] - (q_margin - ttm_margin)) < 1e-9

    print("\n=== SANITY CHECK: latest-quarter margin inflection ===")
    print(f"  Q6 margin={q_margin:.3f}  vs TTM margin={ttm_margin:.3f}  -> "
          f"inflection={F['q_margin_vs_ttm'].loc[d,'AAA']:+.3f}")
    print("  latest-quarter margin minus TTM margin = the inflection feature. Exact.")
