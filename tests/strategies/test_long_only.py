"""
Long-only top-N book (src/strategies/utils/long_only.py). Validates: weights are long-only and
fully invested (sum≈1, all ≥0, no shorts); the book holds ~top_n names (≤ the hold-band cap); the
HOLD-BAND buffer reduces turnover; ERC weighting also yields valid long-only weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.utils.long_only import long_only_book


def _synthetic(seed=0, n_days=250, n_names=20):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_names)]
    stock_ret = pd.DataFrame(rng.normal(0.0003, 0.02, (n_days, n_names)), index=idx, columns=tickers)
    # a signal that wiggles day to day (ranks churn) so the hold-band has something to buffer
    base = rng.normal(0, 1, (n_days, n_names)).cumsum(axis=0) * 0.1
    signal = pd.DataFrame(base + rng.normal(0, 1, (n_days, n_names)), index=idx, columns=tickers)
    signal = signal.iloc[60:]                                   # leave vol warmup
    return signal, stock_ret


def test_long_only_is_long_and_fully_invested():
    signal, stock_ret = _synthetic()
    for weighting in ("inverse_vol", "erc", "equal"):
        bk = long_only_book(signal, stock_ret, 1_000_000, top_n=5, buffer_mult=2.0,
                            weighting=weighting, vol_window=40, rebalance_freq=1)
        w = bk["weights"]
        row_sums = w.sum(axis=1)
        assert (row_sums.round(6).between(0.999, 1.001)).all(), f"{weighting}: weights must sum to 1"
        assert (w.values >= -1e-9).all(), f"{weighting}: NO shorts (all weights >= 0)"
        assert bk["n_holdings"].max() <= 10, "holdings capped at top_n*buffer_mult (=10)"
        assert bk["n_holdings"].min() >= 1 and np.isfinite(bk["net_ret"]).all()
    print("\n=== SANITY CHECK: long-only top-N book ===")
    print(f"  weights sum to 1, all >=0 (no shorts), holdings ≤ hold-band cap; erc/inverse_vol/equal "
          f"all valid. Validated.")


def test_hold_band_reduces_turnover():
    signal, stock_ret = _synthetic(seed=1)
    tight = long_only_book(signal, stock_ret, 1e6, top_n=5, buffer_mult=1.0, weighting="equal",
                           vol_window=40, rebalance_freq=1)   # no buffer -> churns at the top_n edge
    wide = long_only_book(signal, stock_ret, 1e6, top_n=5, buffer_mult=3.0, weighting="equal",
                          vol_window=40, rebalance_freq=1)     # wide hold-band -> names stick
    t_tight, t_wide = float(tight["turnover"].mean()), float(wide["turnover"].mean())
    assert t_wide < t_tight, "a wider hold-band must reduce average turnover"
    print("\n=== SANITY CHECK: hold-band cuts turnover ===")
    print(f"  avg daily turnover: no-buffer={t_tight:.3f}  wide-buffer={t_wide:.3f} "
          f"({(1-t_wide/t_tight)*100:.0f}% lower). Validated.")


if __name__ == "__main__":
    test_long_only_is_long_and_fully_invested()
    test_hold_band_reduces_turnover()
