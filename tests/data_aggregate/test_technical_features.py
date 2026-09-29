"""Tests for the MACD / RSI(14) / ATR(14) technical features
(src/data_aggregate/utils/features.py).

Two things must hold:
  * correctness -- RSI/MACD/ATR match their textbook definitions;
  * NO LEAKAGE -- each indicator on date t is built from prices up to t-1 only
    (the features are lagged one day), so changing the close ON day t must not
    change the feature value ON day t.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.momentum.features import (
    _atr,
    _macd,
    _rsi,
    compute_raw_features,
)


def _number(value: object) -> float:
    assert not isinstance(value, pd.Series | pd.DataFrame)
    return float(cast(float, value))


def _ramp(n=60, start=100.0, step=1.0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"AAA": start + step * np.arange(n)}, index=idx)


def test_rsi_bounds_and_extremes():
    up = _ramp(60, step=1.0)  # strictly rising
    down = _ramp(60, step=-1.0)  # strictly falling
    rsi_up = _rsi(up, 14)
    rsi_down = _rsi(down, 14)

    assert abs(_number(rsi_up.iloc[-1, 0]) - 100.0) < 1e-9, "all-gains -> RSI 100"
    assert _number(rsi_down.iloc[-1, 0]) < 1e-6, "all-losses -> RSI 0"
    # mixed real-ish series stays within [0,100]
    rng = np.random.default_rng(0)
    wig = pd.DataFrame({"AAA": 100 + np.cumsum(rng.normal(0, 1, 200))}, index=pd.bdate_range("2020-01-01", periods=200))
    r = _rsi(wig, 14).dropna()
    assert r.to_numpy().min() >= 0.0 and r.to_numpy().max() <= 100.0

    print("\n=== SANITY CHECK: RSI(14) ===")
    print(f"  rising series -> RSI={rsi_up.iloc[-1, 0]:.1f} (100), falling -> RSI={rsi_down.iloc[-1, 0]:.1f} (0); random stays in [0,100].")


def test_macd_matches_definition():
    rng = np.random.default_rng(1)
    close = pd.DataFrame({"AAA": 100 + np.cumsum(rng.normal(0, 1, 120))}, index=pd.bdate_range("2020-01-01", periods=120))
    macd_norm, hist = _macd(close, 12, 26, 9)

    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    exp_line = (ema12 - ema26) / close
    exp_signal = (ema12 - ema26).ewm(span=9, min_periods=9, adjust=False).mean()
    exp_hist = ((ema12 - ema26) - exp_signal) / close

    d = pd.Timestamp(close.index[-1])
    assert abs(_number(macd_norm.loc[d, "AAA"]) - _number(exp_line.loc[d, "AAA"])) < 1e-9
    assert abs(_number(hist.loc[d, "AAA"]) - _number(exp_hist.loc[d, "AAA"])) < 1e-9

    print("\n=== SANITY CHECK: MACD ===")
    print(f"  MACD line (norm)={macd_norm.loc[d, 'AAA']:+.5f}, histogram={hist.loc[d, 'AAA']:+.5f} -- match EMA12-EMA26 definition.")


def test_atr_true_range_and_wilder():
    idx = pd.bdate_range("2020-01-01", periods=20)
    close = pd.DataFrame({"AAA": np.linspace(100, 110, 20)}, index=idx)
    high = close + 2.0
    low = close - 1.0
    atr = _atr(high, low, close, 14)

    # true range once prev close exists: max(H-L=3, |H-Cprev|, |L-Cprev|).
    prev = close.shift(1)
    tr = np.maximum.reduce([(high - low).to_numpy(), (high - prev).abs().to_numpy(), (low - prev).abs().to_numpy()])
    tr = pd.DataFrame(tr, index=idx, columns=["AAA"])
    exp = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / close
    d = pd.Timestamp(idx[-1])
    assert abs(_number(atr.loc[d, "AAA"]) - _number(exp.loc[d, "AAA"])) < 1e-9
    assert (atr.dropna() >= 0).to_numpy().all()

    print("\n=== SANITY CHECK: ATR(14) ===")
    print(f"  ATR%={atr.loc[d, 'AAA']:.4f} matches Wilder-smoothed true range / close.")


def test_technicals_are_lagged_no_same_day_leakage():
    rng = np.random.default_rng(2)
    base = pd.DataFrame({"AAA": 100 + np.cumsum(rng.normal(0, 1, 80))}, index=pd.bdate_range("2020-01-01", periods=80))
    high = base + 1.5
    low = base - 1.5
    sector = pd.DataFrame(0.0, index=base.index, columns=["AAA"])

    f1 = compute_raw_features(base, base, sector, high=high, low=low)
    # perturb ONLY the last close (and its high/low) by a large shock
    base2 = base.copy()
    base2.iloc[-1, 0] = _number(base2.iloc[-1, 0]) + 25.0
    high2 = high.copy()
    high2.iloc[-1, 0] = _number(high2.iloc[-1, 0]) + 25.0
    low2 = low.copy()
    low2.iloc[-1, 0] = _number(low2.iloc[-1, 0]) + 25.0
    f2 = compute_raw_features(base2, base, sector, high=high2, low=low2)

    d = pd.Timestamp(base.index[-1])
    for name in ["macd", "macd_hist", "rsi_14", "atr_14"]:
        v1, v2 = _number(f1[name].loc[d, "AAA"]), _number(f2[name].loc[d, "AAA"])
        assert (np.isnan(v1) and np.isnan(v2)) or abs(v1 - v2) < 1e-12, f"{name} on day t changed when close[t] changed -> leakage"

    print("\n=== SANITY CHECK: technicals exclude day t (no leakage) ===")
    print("  shocking close[t] by +25 leaves MACD/RSI/ATR on day t unchanged")
    print("  (features are lagged one day -> built only from data up to t-1).")
