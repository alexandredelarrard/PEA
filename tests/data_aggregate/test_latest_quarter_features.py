"""Tests for the LATEST-QUARTER momentum features
(fiscal_apply_to_daily, a test-only transform kept here, and
the discrete single-quarter characteristics in _derived_fields).

These capture the acceleration/inflection that the TTM series smooths away, and
must be point-in-time (a value only appears once its filing is public).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import _observations_to_daily
from src.data_aggregate.utils.fundamentals.fundamental_features import _derived_fields
from src.data_aggregate.utils.fundamentals.inputs import FundamentalHistory
from src.data_aggregate.utils.price.frames import PriceFrames
from tests.data_aggregate.fundamental_pit_reference import known_no_share_events


def fiscal_apply_to_daily(
    fund_hist: pd.DataFrame,
    field: str,
    idx: pd.DatetimeIndex,
    func: Callable[[pd.Series], pd.Series],
    *,
    max_age_days: int | None = None,
) -> pd.DataFrame:
    """Apply a per-ticker series transform (e.g. YoY growth, or acceleration =
    change in YoY) to a fiscal field, forward-filled point-in-time onto trading
    days. `func` receives one ticker's chronological series and returns a series
    of the same length."""
    if field not in fund_hist.columns:
        return pd.DataFrame(index=idx)
    df = fund_hist[["ticker", "as_of", field]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    df[field] = pd.to_numeric(df[field], errors="coerce")
    df = df.dropna(subset=[field]).sort_values(["ticker", "as_of"])
    if df.empty:
        return pd.DataFrame(index=idx)
    df["v"] = df.groupby("ticker")[field].transform(func)
    df["v"] = df["v"].replace([np.inf, -np.inf], np.nan)
    return _observations_to_daily(df, "v", idx, max_age_days=max_age_days)


def _synth_quarterly():
    """6 fiscal quarters for one ticker, discrete single-quarter revenue/income
    plus the TTM level fields _derived_fields expects."""
    ends = pd.bdate_range("2019-03-01", periods=6, freq="63D")  # ~quarterly
    rev_q = [100.0, 110.0, 120.0, 130.0, 121.0, 143.0]
    ni_q = [10.0, 12.0, 14.0, 16.0, 13.0, 20.0]
    rows = []
    for i, e in enumerate(ends):
        ttm_rev = sum(rev_q[max(0, i - 3) : i + 1])
        ttm_ni = sum(ni_q[max(0, i - 3) : i + 1])
        rows.append(
            {
                "ticker": "AAA",
                "as_of": e.date().isoformat(),
                "totalRevenue": ttm_rev,
                "netIncome": ttm_ni,
                "profitMargins": ttm_ni / ttm_rev,
                "revenue_q": rev_q[i],
                "netIncome_q": ni_q[i],
            }
        )
    return pd.DataFrame(rows), ends, rev_q, ni_q


def test_fiscal_apply_yoy_and_acceleration():
    fund, ends, rev_q, _ = _synth_quarterly()
    idx = pd.bdate_range("2019-01-01", "2021-06-01")

    yoy = fiscal_apply_to_daily(fund, "revenue_q", idx, lambda s: s.pct_change(4))
    accel = fiscal_apply_to_daily(fund, "revenue_q", idx, lambda s: s.pct_change(4).diff(1))

    after_q5 = ends[4] + pd.Timedelta(days=5)  # 5th quarter public
    after_q6 = ends[5] + pd.Timedelta(days=5)  # 6th quarter public
    yoy_q5 = rev_q[4] / rev_q[0] - 1  # 121/100 - 1 = 0.21
    yoy_q6 = rev_q[5] / rev_q[1] - 1  # 143/110 - 1 = 0.30

    assert abs(yoy.loc[after_q5, "AAA"] - yoy_q5) < 1e-9
    assert abs(yoy.loc[after_q6, "AAA"] - yoy_q6) < 1e-9
    assert abs(accel.loc[after_q6, "AAA"] - (yoy_q6 - yoy_q5)) < 1e-9
    # no look-ahead: before the very first year-over-year is computable -> NaN
    assert np.isnan(yoy.loc[ends[0] + pd.Timedelta(days=5), "AAA"])

    print("\n=== SANITY CHECK: latest-quarter YoY + acceleration ===")
    print(f"  Q5 YoY={yoy.loc[after_q5, 'AAA']:.2%} (121/100), Q6 YoY={yoy.loc[after_q6, 'AAA']:.2%} (143/110)")
    print(f"  acceleration Q6 = {accel.loc[after_q6, 'AAA']:+.2%} = Q6 YoY - Q5 YoY.")
    print("  NaN before the first YoY is computable -> point-in-time, no look-ahead.")


def test_latest_quarter_margin_inflection_exact():
    fund, ends, rev_q, ni_q = _synth_quarterly()
    idx = pd.bdate_range("2019-01-01", "2021-06-01")

    fields = _derived_fields(
        FundamentalHistory(raw_history=fund, share_basis=known_no_share_events()),
        PriceFrames(trading_index=idx, universe=(), peers={}),
        yoy_periods=4,
    )
    assert "q_margin_vs_ttm" in fields
    assert "q_rev_growth" not in fields

    d = ends[5] + pd.Timedelta(days=5)  # after Q6 filing
    q_margin = ni_q[5] / rev_q[5]  # 20/143
    ttm_margin = sum(ni_q[2:6]) / sum(rev_q[2:6])
    assert abs(fields["q_margin_vs_ttm"].loc[d, "AAA"] - (q_margin - ttm_margin)) < 1e-9

    print("\n=== SANITY CHECK: latest-quarter margin inflection ===")
    print(f"  Q6 margin={q_margin:.3f}  vs TTM margin={ttm_margin:.3f}  -> inflection={fields['q_margin_vs_ttm'].loc[d, 'AAA']:+.3f}")
    print("  latest-quarter margin minus TTM margin = the inflection feature; redundant q_rev_growth is absent. Exact.")
