"""Step 7 — Volume-flow dynamics.

Checks: signed-volume imbalance reflects buying vs selling pressure; volume trend
/ dispersion are produced; and all are point-in-time.

`tax_loss_pressure` used to live here. It was DROPPED: its encoding is non-stationary, so the
identical economic state ranks differently across years (the zero block reads 0.039 in 2008 vs
0.455 in 2013), which is a year effect masquerading as a stock signal. `test_the_dropped_
tax_loss_pressure_stays_dropped` below pins the removal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.momentum.features import compute_raw_features, build_feature_panel

_FLOW = ["signed_vol_63", "volume_trend_63", "volume_cv_63"]


def _synth(years=3, N=8, seed=0):
    T = years * 252
    dates = pd.bdate_range("2021-01-04", periods=T)
    tickers = [f"S{i}" for i in range(N)]
    rng = np.random.default_rng(seed)
    ret = pd.DataFrame(rng.normal(0.0003, 0.015, (T, N)), index=dates, columns=tickers)
    # S0 = a big YTD loser (persistent negative drift) -> tax-loss candidate
    ret["S0"] = rng.normal(-0.004, 0.015, T)
    close = 100 * (1 + ret).cumprod()
    open_ = close.shift(1).bfill()
    # volume: put MORE volume on up-days for S1 (accumulation), down-days for S2
    volume = pd.DataFrame(rng.uniform(1e6, 2e6, (T, N)), index=dates, columns=tickers)
    up = ret > 0
    volume["S1"] = volume["S1"] * np.where(up["S1"], 3.0, 1.0)
    volume["S2"] = volume["S2"] * np.where(up["S2"], 1.0, 3.0)
    sector = pd.DataFrame(0.0, index=dates, columns=tickers)
    return dates, tickers, close, open_, volume, sector


def test_flow_features_present_and_signed_volume():
    dates, tickers, close, open_, volume, sector = _synth()
    raw = compute_raw_features(close, open_, sector, volume=volume, seasonal_horizons=None)
    for k in _FLOW:
        assert k in raw, f"{k} missing"
    t = dates[-1]
    sv = raw["signed_vol_63"].loc[t]
    # S1 (volume on up-days) net-positive; S2 (volume on down-days) net-negative
    assert sv["S1"] > 0 > sv["S2"], (sv["S1"], sv["S2"])
    print("\n=== SANITY CHECK: flow features + signed volume ===")
    print(f"  all of {_FLOW} built; signed_vol_63 S1={sv['S1']:+.2f} (buys) vs "
          f"S2={sv['S2']:+.2f} (sells). Validated.")


def test_the_dropped_tax_loss_pressure_stays_dropped():
    """`tax_loss_pressure` was removed from the feature set, the stored part, `schema.sql` and
    all three model configs. Pinned here so it cannot come back by accident: the anomaly is
    real, but this ENCODING is not stationary -- off-window the value was NaN rather than 0
    precisely because a 0 fed the per-day ranker a sea of equal zeros (9 of 12 months plus
    every in-season YTD winner) and collapsed the feature to a near-constant ~0.5 whose only
    variation was the rank's +1/(2N) small-sample bias, manufacturing a spurious downward trend
    linear in the year as the universe grew. NaN fixed the drift but not the underlying
    non-stationarity: the zero block still ranks 0.039 in 2008 against 0.455 in 2013 for the
    identical economic state. If the anomaly is wanted back, it belongs in an Oct-Dec
    specialist sleeve, not as a column the ranker sees all year.
    """
    dates, tickers, close, open_, volume, sector = _synth()
    raw = compute_raw_features(close, open_, sector, volume=volume)
    panel = build_feature_panel(close, open_, sector, method="rank", volume=volume)

    assert "tax_loss_pressure" not in raw
    assert "tax_loss_pressure" not in panel.columns
    assert not any("tax_loss" in name for name in raw), sorted(raw)

    print("\n=== SANITY CHECK: tax_loss_pressure is gone ===")
    print(f"  {len(raw)} raw features built, none matching 'tax_loss'; the long panel has "
          f"{len(panel.columns) - 2} feature columns and no tax_loss_pressure. Validated.")


def test_flow_is_leak_free_and_panel():
    dates, tickers, close, open_, volume, sector = _synth()
    base = compute_raw_features(close, open_, sector, volume=volume)
    t_idx = 252 * 2
    close2, vol2 = close.copy(), volume.copy()
    close2.iloc[t_idx + 1:] *= 1.5
    vol2.iloc[t_idx + 1:] *= 4.0
    pert = compute_raw_features(close2, open_, sector, volume=vol2)
    for k in _FLOW:
        assert np.allclose(base[k].iloc[t_idx].to_numpy(),
                           pert[k].iloc[t_idx].to_numpy(), equal_nan=True), f"{k} leaks"

    panel = build_feature_panel(close, open_, sector, method="rank", volume=volume)
    for k in _FLOW:
        assert k in panel.columns and panel[k].dropna().between(0, 1).all()
    print("\n=== SANITY CHECK: flow features point-in-time + panel ===")
    print("  perturbing all data AFTER t left every flow feature at t unchanged; "
          "panel columns rank-standardized to [0,1]. Validated.")


if __name__ == "__main__":
    test_flow_features_present_and_signed_volume()
    test_the_dropped_tax_loss_pressure_stays_dropped()
    test_flow_is_leak_free_and_panel()
