"""Step 1 — Liquidity features (dollar volume, Amihud illiquidity, relative
volume). Verify they are produced only when volume is supplied, are point-in-time
(no look-ahead), rank the universe sensibly, and are cross-sectionally
standardized in the built panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.momentum.features import compute_raw_features, build_feature_panel

_LIQ = ["dollar_volume_63", "amihud_63", "rel_volume_5_63"]


def _synth(T=260, N=25, seed=0):
    dates = pd.bdate_range("2022-01-03", periods=T)
    tickers = [f"S{i:02d}" for i in range(N)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.015, (T, N)), axis=0),
                         index=dates, columns=tickers)
    open_ = close.shift(1).fillna(close.iloc[0])
    # volume: each stock a different base level -> a clear liquidity ranking
    base = np.linspace(1e5, 1e7, N)
    volume = pd.DataFrame(base[None, :] * rng.lognormal(0, 0.3, (T, N)),
                          index=dates, columns=tickers)
    sector = pd.DataFrame(rng.normal(0, 0.01, (T, N)), index=dates, columns=tickers)
    return dates, tickers, close, open_, volume, sector


def test_liquidity_skipped_without_volume_present_with():
    dates, tickers, close, open_, volume, sector = _synth()
    no_vol = compute_raw_features(close, open_, sector)
    assert not any(k in no_vol for k in _LIQ), "liquidity must be skipped without volume"

    with_vol = compute_raw_features(close, open_, sector, volume=volume)
    for k in _LIQ:
        assert k in with_vol, f"{k} missing when volume supplied"
        # finite after warmup (need >=20 obs for the 63d windows)
        assert np.isfinite(with_vol[k].iloc[80]).sum() >= 20, f"{k} mostly NaN post-warmup"

    print("\n=== SANITY CHECK: liquidity present iff volume supplied ===")
    print(f"  no volume -> none of {_LIQ}; with volume -> all present & finite post-warmup. Validated.")


def test_no_lookahead():
    """A feature at date t must not change if FUTURE rows are altered."""
    dates, tickers, close, open_, volume, sector = _synth()
    t_idx = 150
    base = compute_raw_features(close, open_, sector, volume=volume)

    close2, volume2 = close.copy(), volume.copy()
    close2.iloc[t_idx + 1:] *= 1.5          # perturb everything AFTER t
    volume2.iloc[t_idx + 1:] *= 3.0
    pert = compute_raw_features(close2, open_, sector, volume=volume2)

    for k in _LIQ:
        a = base[k].iloc[t_idx].to_numpy()
        b = pert[k].iloc[t_idx].to_numpy()
        assert np.allclose(a, b, equal_nan=True), f"{k} leaks future data at t"
    print("\n=== SANITY CHECK: liquidity features are point-in-time ===")
    print("  perturbing all rows AFTER t left every liquidity value at t unchanged. Validated.")


def test_dollar_volume_ranks_liquidity():
    dates, tickers, close, open_, volume, sector = _synth()
    raw = compute_raw_features(close, open_, sector, volume=volume)
    dv = raw["dollar_volume_63"].iloc[-1]      # last date
    # S24 has ~100x the base volume of S00 -> must rank far higher on $ volume
    assert dv[tickers[-1]] > dv[tickers[0]], (dv[tickers[0]], dv[tickers[-1]])

    panel = build_feature_panel(close, open_, sector, method="rank", volume=volume)
    for k in _LIQ:
        assert k in panel.columns, f"{k} not in built panel"
        col = panel[k].dropna()
        assert col.between(0, 1).all(), f"{k} not rank-standardized to [0,1]"
    print("\n=== SANITY CHECK: dollar_volume ranks liquidity; panel standardized ===")
    print(f"  high-volume name outranks low-volume name on dollar_volume_63; "
          f"all liquidity cols rank-standardized to [0,1] in the panel. Validated.")


if __name__ == "__main__":
    test_liquidity_skipped_without_volume_present_with()
    test_no_lookahead()
    test_dollar_volume_ranks_liquidity()
