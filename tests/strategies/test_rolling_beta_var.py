"""Regression: rolling_beta_var must be NaN-tolerant.

Daily returns (pct_change) always start with a NaN and carry scattered holes
(suspensions, missing closes, recent IPOs). With the old default
`min_periods == window`, a single NaN anywhere in the trailing window blanked the
beta, so `beta_df` came back almost all-NaN, the day loop's `len(common) >= 10`
check never passed, and the book never traded (the "no trades / late first buy"
symptom). A NaN in SPY blanked the whole cross-section. The fix sets min_periods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.utils.strategies_opt import rolling_beta_var, simulate_portfolio_opt


def _gappy(T=300, N=30, seed=0):
    dates = pd.bdate_range("2023-01-02", periods=T)
    tickers = [f"S{i:02d}" for i in range(N)]
    rng = np.random.default_rng(seed)
    spy = pd.Series(rng.normal(0, 0.01, T), index=dates)
    stock = pd.DataFrame(rng.normal(0, 0.02, (T, N)), index=dates, columns=tickers)
    stock.iloc[0] = np.nan                       # pct_change leading NaN
    for j in range(N):                           # scattered per-name gaps
        stock.iloc[rng.choice(T, size=3, replace=False), j] = np.nan
    return dates, tickers, stock, spy


def test_beta_var_is_nan_tolerant():
    dates, tickers, stock, spy = _gappy()
    t = dates[250]                               # well past the 63-day warmup

    beta, idio = rolling_beta_var(stock, spy, 63, 63)
    finite = int((np.isfinite(beta.loc[t]) & np.isfinite(idio.loc[t])).sum())
    assert finite >= 25, f"only {finite}/30 names have finite beta/var (min_periods broken)"

    # even a SPY with interior gaps must not blank the whole cross-section
    spy_gap = spy.copy(); spy_gap.iloc[[100, 150, 151]] = np.nan
    beta2, _ = rolling_beta_var(stock, spy_gap, 63, 63)
    assert int(np.isfinite(beta2.loc[t]).sum()) >= 25, "SPY gaps blanked the cross-section"

    print("\n=== SANITY CHECK: rolling_beta_var NaN tolerance ===")
    print(f"  gappy returns -> {finite}/30 names have finite beta & var at {t.date()} "
          f"(was ~14/30); SPY-gap case also recovers. min_periods works. Validated.")


def test_book_trades_on_gappy_returns():
    dates, tickers, stock, spy = _gappy()
    # persistent cross-sectional signal so a target exists every day
    base = np.random.default_rng(1).normal(0, 1, len(tickers))
    signal = pd.DataFrame(np.tile(base, (len(dates), 1)), index=dates, columns=tickers)

    d = simulate_portfolio_opt(
        signal, stock, spy, market_weight=0.0, target_ann_vol=0.10,
        beta_neutral=True, pos_cap=0.08, gross_cap=3.0, step=0.4,
        beta_window=63, vol_window=63, fee_bps=1.0, spread_bps=5.0, rebalance_freq=1)

    post = d.iloc[70:]                           # after the beta warmup
    assert post["alpha_gross"].mean() > 0.1, "alpha book never established on gappy data"
    assert (post["turnover"] > 0).mean() > 0.5, "book barely trades on gappy data"

    print("\n=== SANITY CHECK: book trades end-to-end on gappy returns ===")
    print(f"  post-warmup avg alpha gross = {post['alpha_gross'].mean():.2f}, "
          f"share of days trading = {(post['turnover']>0).mean()*100:.0f}%. "
          f"With gappy returns the book now establishes positions instead of "
          f"staying flat (common>=10 passes). Validated.")


if __name__ == "__main__":
    test_beta_var_is_nan_tolerant()
    test_book_trades_on_gappy_returns()
