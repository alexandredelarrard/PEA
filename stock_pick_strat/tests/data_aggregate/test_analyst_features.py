"""Tests for the analyst-estimate features
(src/data_aggregate/utils/analyst_features.py).

The estimates are only a CURRENT snapshot (no free historical archive), so the
one guarantee that matters is that every feature is strictly point-in-time: a
value must appear ONLY from its real `as_of` onward, never broadcast backwards
(which would be look-ahead). We also check the ratio math and the
estimates-vs-intrinsic comparison sign.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.analyst_features import _analyst_fields
from src.data_aggregate.utils.intrinsic import intrinsic_value_daily


def _synth_analyst():
    as_of = "2020-03-02"
    row = {
        "ticker": "AAA", "as_of": as_of,
        "eps_est_0y": 5.0, "eps_est_+1y": 6.0,
        "rev_est_0y": 1000.0, "rev_est_+1y": 1100.0,
        "price_target_mean": 12.0,
        "eps_revisions_+1y_upLast30days": 6.0,
        "eps_revisions_+1y_downLast30days": 2.0,
        "eps_trend_+1y_current": 6.0, "eps_trend_+1y_90daysAgo": 5.0,
        "rec_strongBuy": 3.0, "rec_buy": 2.0, "rec_hold": 1.0,
        "rec_sell": 0.0, "rec_strongSell": 0.0,
    }
    return pd.DataFrame([row])


def _synth_fund_for_intrinsic():
    return pd.DataFrame({
        "ticker": ["AAA"], "as_of": ["2020-03-02"],
        "freeCashflow": [100.0], "revenueGrowth": [0.05],
        "sharesOutstanding": [1000.0],
    })


def test_analyst_ratio_math_and_point_in_time():
    idx = pd.bdate_range("2020-01-01", "2020-06-01")
    close = pd.DataFrame({"AAA": 10.0}, index=idx)
    F = _analyst_fields(_synth_analyst(), idx, close,
                        fund_hist=_synth_fund_for_intrinsic(), intrinsic_cfg={})

    before = pd.Timestamp("2020-02-14")   # before the 2020-03-02 as_of
    after = pd.Timestamp("2020-04-01")    # after it is public

    # ---- point-in-time: nothing before the estimate's as_of ----
    for name in F:
        assert np.isnan(F[name].loc[before, "AAA"]), f"{name} leaked before as_of"

    # ---- ratio math on/after as_of ----
    assert abs(F["est_target_upside"].loc[after, "AAA"] - (12.0 / 10.0 - 1)) < 1e-9
    assert abs(F["est_eps_growth"].loc[after, "AAA"] - (6.0 / 5.0 - 1)) < 1e-9
    assert abs(F["est_rev_growth"].loc[after, "AAA"] - (1100.0 / 1000.0 - 1)) < 1e-9
    assert abs(F["est_fwd_earnings_yield"].loc[after, "AAA"] - (5.0 / 10.0)) < 1e-9
    # revisions: (6 up - 2 down)/(6+2) = 0.5
    assert abs(F["est_revision_ratio"].loc[after, "AAA"] - 0.5) < 1e-9
    # eps trend 3m: (6-5)/|5| = 0.2
    assert abs(F["est_eps_trend_3m"].loc[after, "AAA"] - 0.2) < 1e-9
    # rec score: (2*3 + 2 - 0 - 2*0)/(3+2+1) = 8/6
    assert abs(F["est_rec_score"].loc[after, "AAA"] - 8.0 / 6.0) < 1e-9

    print("\n=== SANITY CHECK: analyst ratios + point-in-time ===")
    print(f"  target upside={F['est_target_upside'].loc[after,'AAA']:.2f}  "
          f"eps growth={F['est_eps_growth'].loc[after,'AAA']:.2f}  "
          f"rev growth={F['est_rev_growth'].loc[after,'AAA']:.2f}")
    print(f"  revision ratio={F['est_revision_ratio'].loc[after,'AAA']:.2f}  "
          f"rec score={F['est_rec_score'].loc[after,'AAA']:.2f}")
    print("  All values NaN before the 2020-03-02 as_of -> strictly leak-free.")


def test_est_vs_intrinsic_matches_definition():
    idx = pd.bdate_range("2020-01-01", "2020-06-01")
    close = pd.DataFrame({"AAA": 10.0}, index=idx)
    fund = _synth_fund_for_intrinsic()
    F = _analyst_fields(_synth_analyst(), idx, close, fund_hist=fund, intrinsic_cfg={})

    d = pd.Timestamp("2020-04-01")
    iv = intrinsic_value_daily(fund, close, idx)
    ips = iv["per_share"].loc[d, "AAA"]
    intrinsic_up = ips / 10.0 - 1
    analyst_up = 12.0 / 10.0 - 1
    assert abs(F["est_vs_intrinsic"].loc[d, "AAA"] - (intrinsic_up - analyst_up)) < 1e-9

    print("\n=== SANITY CHECK: estimates vs intrinsic value ===")
    print(f"  intrinsic per-share={ips:.2f} -> intrinsic upside={intrinsic_up:+.1%}; "
          f"analyst target upside={analyst_up:+.1%}")
    print(f"  est_vs_intrinsic={F['est_vs_intrinsic'].loc[d,'AAA']:+.2f} "
          f"(>0: our DCF sees more value than the street's target).")
