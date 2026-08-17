"""
Unified 3-strategy portfolio blend (src/post_processing/backtest/step_portfolio_backtest.py).

The orchestrator blends the sleeve return streams with base_weights (ERC/EWMA across sleeves)
+ blend_to_vol_target (global vol/leverage). This validates the two behaviours that matter for
the blend: (1) NaN-aware weighting — a late-starting sleeve (L/S is OOS only) gets ~0 weight
until it has history, the others carry the book; (2) the blended book hits ~the global vol target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.risk_parity import base_weights, series_metrics
from src.portfolio.utils.blend import blend_to_vol_target


def test_sleeve_blend_nan_aware_and_global_vol_target():
    idx = pd.bdate_range("2020-01-01", periods=900)
    rng = np.random.default_rng(0)
    long_book = pd.Series(rng.normal(0.0003, 0.006, len(idx)), index=idx)
    trend = pd.Series(rng.normal(0.0002, 0.008, len(idx)), index=idx)
    ls = pd.Series(np.nan, index=idx)
    ls.iloc[400:] = rng.normal(0.0004, 0.005, len(idx) - 400)     # L/S joins late (OOS start)
    rets = pd.DataFrame({"ls_equity": ls, "long_book": long_book, "trend_cta": trend})

    W = base_weights(rets, window=63, scheme="erc", rebalance_freq=21,
                     cov_mode="ewma", cov_halflife=63)
    early = W.iloc[200]
    assert not (early["ls_equity"] > 1e-6), "L/S must carry ~no weight before it has history"
    assert abs(early[["long_book", "trend_cta"]].sum() - 1.0) < 1e-6, "live sleeves sum to 1"
    late = W.iloc[880]
    assert (late[["ls_equity", "long_book", "trend_cta"]] > 0).all(), "all 3 weighted once L/S is live"

    blended = blend_to_vol_target(rets, W, portfolio_vol_target=0.10, vol_window=63, max_leverage=2.0)
    net = blended["ret"].dropna()
    assert np.isfinite(net).all() and len(net) > 800
    vol = float(net.std() * np.sqrt(252))
    assert 0.04 < vol < 0.20, f"blended vol {vol:.3f} not near the 10% global target"

    print("\n=== SANITY CHECK: 3-strategy blend (NaN-aware + global vol target) ===")
    print(f"  before L/S: weights long_book={early['long_book']:.2f} trend_cta={early['trend_cta']:.2f} "
          f"(L/S ~0); after L/S live: {late[['ls_equity','long_book','trend_cta']].round(2).to_dict()}")
    print(f"  blended ann-vol {vol*100:.1f}% (target 10%), Sharpe {series_metrics(net)['sharpe']:.2f}. Validated.")


if __name__ == "__main__":
    test_sleeve_blend_nan_aware_and_global_vol_target()
