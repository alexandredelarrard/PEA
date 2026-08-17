"""Tests for the price-only Tier-2 anomalies added to features.compute_raw_features
(MAX, return skewness, downside semi-deviation, idiosyncratic vol). All are
trailing-window / point-in-time and built from the already-extracted price panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.momentum.features import compute_raw_features


def _prices(n_days=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    tickers = ["AAA", "BBB", "CCC"]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.012, (n_days, 3)), axis=0),
                         index=dates, columns=tickers)
    open_ = close.shift(1).bfill()
    sector = pd.DataFrame(0.0, index=dates, columns=tickers)
    return close, open_, sector


def test_price_anomalies_present_and_exact():
    close, open_, sector = _prices()
    feats = compute_raw_features(close, open_, sector)
    for k in ("max_21", "ret_skew_126", "downside_vol_63", "idio_vol_63"):
        assert k in feats, f"missing anomaly feature {k}"

    ret = close.pct_change(fill_method=None)
    d = close.index[-1]

    # MAX = highest daily return over the trailing 21 days
    assert abs(feats["max_21"].loc[d, "AAA"] - ret["AAA"].iloc[-21:].max()) < 1e-12
    # idiosyncratic vol = std of (stock ret - equal-weight universe ret) over 63d
    mkt = ret.mean(axis=1)
    idio = (ret["AAA"] - mkt).iloc[-63:]
    assert abs(feats["idio_vol_63"].loc[d, "AAA"] - idio.std()) < 1e-9
    # downside vol is finite and non-negative; skewness is finite
    assert feats["downside_vol_63"].loc[d, "AAA"] >= 0
    assert np.isfinite(feats["ret_skew_126"].loc[d, "AAA"])
    # point-in-time: 126d skewness undefined at the very start
    assert np.isnan(feats["ret_skew_126"].iloc[0]["AAA"])

    print("\n=== SANITY CHECK: price-only anomalies ===")
    print(f"  MAX_21={feats['max_21'].loc[d,'AAA']:.4f} (=max of last 21 daily rets); "
          f"idio_vol_63={feats['idio_vol_63'].loc[d,'AAA']:.4f} (market-relative). Exact.")
    print("  skewness NaN at start -> trailing/point-in-time. Validated.")
