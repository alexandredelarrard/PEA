"""Regression: the horizon-accuracy table must not drop nearly every date to NaN.

`forward_return` used rolling(horizon, min_periods=horizon): one NaN anywhere in
the 30/60/90-day forward window blanked a name's forward return; after dropna the
date had < 10 names and was skipped entirely -> the accuracy chart showed only a
handful of days even though the book trades daily. The fix requires only a
fraction of the window, so scattered gaps no longer blank the name/date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.post_processing.utils.accuracy import (
    forward_return, compute_horizon_accuracy, horizon_accuracy_summary,
)


class _BT:
    """Duck-typed StepBacktest: only .signal, .stock_ret, .spy_ret are used."""
    def __init__(self, signal, stock_ret, spy_ret):
        self.signal, self.stock_ret, self.spy_ret = signal, stock_ret, spy_ret


def _gappy_bt(T=400, N=40, seed=0, miss_prob=0.03):
    dates = pd.bdate_range("2023-01-02", periods=T)
    tickers = [f"S{i:02d}" for i in range(N)]
    rng = np.random.default_rng(seed)
    # persistent cross-sectional signal that genuinely predicts the drift
    alpha = rng.normal(0, 1, N)
    daily = alpha[None, :] * 0.0015 + rng.normal(0, 0.02, (T, N))
    stock = pd.DataFrame(daily, index=dates, columns=tickers)
    stock.iloc[0] = np.nan                      # pct_change leading NaN
    # realistic scattered missingness (suspensions / missing closes): ~3% of cells
    miss = rng.random((T, N)) < miss_prob
    stock = stock.mask(miss)
    spy = pd.Series(rng.normal(0.0003, 0.01, T), index=dates)
    signal = pd.DataFrame(np.tile(alpha, (T, 1)), index=dates, columns=tickers)
    return _BT(signal, stock, spy), dates


def test_forward_return_is_nan_tolerant():
    bt, dates = _gappy_bt()
    strict = _forward_full_window(bt.stock_ret, 60)   # old behaviour
    lenient = forward_return(bt.stock_ret, 60)          # new behaviour
    t = dates[150]
    assert np.isfinite(strict.loc[t]).sum() < 10, "setup: strict rule blanks most names"
    assert np.isfinite(lenient.loc[t]).sum() >= 30, "fix should recover most names"
    print("\n=== SANITY CHECK: forward_return NaN tolerance ===")
    print(f"  at {t.date()}: strict(min_periods=horizon) finite={int(np.isfinite(strict.loc[t]).sum())}/40; "
          f"lenient finite={int(np.isfinite(lenient.loc[t]).sum())}/40. Validated.")


def _forward_full_window(daily_ret, horizon):
    """Reproduce the OLD strict rule for contrast."""
    safe = daily_ret.clip(lower=-0.999999)
    logr = np.log1p(safe)
    return np.expm1(logr[::-1].rolling(horizon, min_periods=horizon).sum()[::-1].shift(-1))


def test_accuracy_table_covers_the_period_not_a_few_days():
    bt, dates = _gappy_bt()
    acc = compute_horizon_accuracy(bt, 60)

    # a 400-day window minus ~60d tail -> hundreds of scored dates, not a handful
    assert len(acc) > 250, f"accuracy table collapsed to {len(acc)} dates (NaN drop bug)"
    # the persistent skilled signal should score clearly above coin-flip
    assert acc["hit_rate_%"].mean() > 55, acc["hit_rate_%"].mean()

    summ = horizon_accuracy_summary(bt, [30, 60, 90])
    assert (summ["n_dates"] > 200).all(), summ

    print("\n=== SANITY CHECK: accuracy table covers the whole period ===")
    print(f"  h=60 scored on {len(acc)} dates (avg hit {acc['hit_rate_%'].mean():.1f}%). "
          f"n_dates by horizon: {summ['n_dates'].to_dict()}. No more 'few days only'. Validated.")


if __name__ == "__main__":
    test_forward_return_is_nan_tolerant()
    test_accuracy_table_covers_the_period_not_a_few_days()
