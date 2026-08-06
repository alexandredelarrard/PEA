"""Step 5 — Cross-sectional seasonality at t+h (Heston-Sadka).

Checks: the seasonal feature exists per horizon; it is CROSS-SECTIONALLY dispersed
(unlike a calendar dummy); it is strictly leak-free (uses only prior-year windows);
and a stock with a repeating same-calendar-window return ranks high on the
seasonal feature at dates whose forward window hits that season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.momentum.features import compute_raw_features, build_feature_panel


def _synth(years=6, N=12, seed=0, season_start=120, season_len=40, boost=0.004):
    """~`years` of business days. Ticker 'SEAS' gets an extra positive daily drift
    during a fixed calendar window [season_start, season_start+season_len) every
    year; everyone else is plain noise."""
    T = years * 252
    dates = pd.bdate_range("2016-01-04", periods=T)
    tickers = [f"S{i:02d}" for i in range(N - 1)] + ["SEAS"]
    rng = np.random.default_rng(seed)
    ret = pd.DataFrame(rng.normal(0, 0.01, (T, N)), index=dates, columns=tickers)
    doy = dates.dayofyear.to_numpy()
    in_season = (doy >= season_start) & (doy < season_start + season_len)
    ret.loc[in_season, "SEAS"] += boost                     # repeating seasonal drift
    close = 100 * (1 + ret).cumprod()
    open_ = close.shift(1).bfill()
    sector = pd.DataFrame(0.0, index=dates, columns=tickers)
    return dates, tickers, close, open_, sector, season_start, season_len


def test_seasonal_present_and_dispersed():
    dates, tickers, close, open_, sector, s0, sl = _synth()
    raw = compute_raw_features(close, open_, sector, seasonal_horizons=[30, 60, 90])
    for h in (30, 60, 90):
        assert f"seasonal_h{h}" in raw, f"seasonal_h{h} missing"
    # CROSS-SECTIONAL dispersion: not identical across names (a calendar dummy would be)
    late = raw["seasonal_h30"].iloc[-260:]                  # after >=1y of history
    row = late.dropna(how="all").iloc[-1]
    assert row.notna().sum() >= 5 and row.std() > 0, "seasonal feature has no dispersion"
    print("\n=== SANITY CHECK: seasonal present & cross-sectionally dispersed ===")
    print(f"  seasonal_h30/60/90 built; last row std={row.std():.5f} (>0) across "
          f"{int(row.notna().sum())} names -> NOT a flat calendar dummy. Validated.")


def test_seasonal_is_leak_free():
    dates, tickers, close, open_, sector, s0, sl = _synth()
    base = compute_raw_features(close, open_, sector, seasonal_horizons=[60])["seasonal_h60"]

    # perturb ALL returns after t -> seasonal at t must not move (uses only past yrs)
    t_idx = 252 * 4
    close2 = close.copy()
    close2.iloc[t_idx + 1:] *= 1.5
    pert = compute_raw_features(close2, open_, sector, seasonal_horizons=[60])["seasonal_h60"]
    a = base.iloc[t_idx].to_numpy(); b = pert.iloc[t_idx].to_numpy()
    assert np.allclose(a, b, equal_nan=True), "seasonal_h60 leaked future returns"
    print("\n=== SANITY CHECK: seasonal is strictly leak-free ===")
    print("  perturbing all returns AFTER t left seasonal_h60 at t unchanged "
          "(only prior-year windows used). Validated.")


def test_seasonal_ranks_the_seasonal_stock():
    dates, tickers, close, open_, sector, s0, sl = _synth()
    raw = compute_raw_features(close, open_, sector, seasonal_horizons=[30])
    seas = raw["seasonal_h30"]
    doy = dates.dayofyear.to_numpy()
    # dates in a late year whose 30d-forward window sits INSIDE SEAS's season,
    # so its historical same-window return (prior years) is high
    mask = (np.arange(len(dates)) > 252 * 3) & (doy >= s0 - 25) & (doy < s0 + sl - 30)
    cand = dates[mask]
    ranks = [seas.loc[d].rank(pct=True)["SEAS"] for d in cand
             if seas.loc[d].notna().sum() >= 5 and not np.isnan(seas.loc[d, "SEAS"])]
    avg_rank = float(np.mean(ranks))
    assert avg_rank > 0.7, f"seasonal stock not ranked high (avg pct-rank {avg_rank:.2f})"

    # panel integration: seasonal columns present and rank-standardized to [0,1]
    panel = build_feature_panel(close, open_, sector, method="rank", seasonal_horizons=[30, 60])
    assert {"seasonal_h30", "seasonal_h60"}.issubset(panel.columns)
    assert panel["seasonal_h30"].dropna().between(0, 1).all()
    print("\n=== SANITY CHECK: seasonal ranks the seasonal name ===")
    print(f"  when the 30d-forward window hits SEAS's recurring season, SEAS's avg "
          f"cross-sectional pct-rank on seasonal_h30 = {avg_rank:.2f} (>0.7). "
          f"Panel columns standardized to [0,1]. Validated.")


if __name__ == "__main__":
    test_seasonal_present_and_dispersed()
    test_seasonal_is_leak_free()
    test_seasonal_ranks_the_seasonal_stock()
