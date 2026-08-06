"""Step 7 — Volume-flow dynamics + forced year-end flow.

Checks: signed-volume imbalance reflects buying vs selling pressure; volume trend
/ dispersion are produced; tax_loss_pressure is a cross-sectional (dispersed)
year-end effect that is zero off-season; and all are point-in-time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.momentum.features import compute_raw_features, build_feature_panel

_FLOW = ["signed_vol_63", "volume_trend_63", "volume_cv_63", "tax_loss_pressure"]


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


def test_tax_loss_pressure_is_seasonal_and_cross_sectional():
    dates, tickers, close, open_, volume, sector = _synth()
    tlp = compute_raw_features(close, open_, sector, volume=volume)["tax_loss_pressure"]

    # off-season (e.g. June) -> NaN for everyone (NOT 0): a 0 here would feed the per-day
    # ranker a sea of equal zeros -> a ~0.5-constant feature with a spurious universe-size drift.
    june = [d for d in dates if d.month == 6]
    assert tlp.loc[june].isna().all().all(), "tax_loss must be NaN off-season (not 0)"
    # in-season (Nov/Dec) -> non-zero and DISPERSED (S0 the loser highest)
    dsel = [d for d in dates if d.month == 12 and tlp.loc[d].notna().any()][-1]
    row = tlp.loc[dsel]
    assert row.max() > 0 and row.std() > 0, "tax_loss flat/zero in-season"
    assert row.idxmax() == "S0", (dsel, row.to_dict())
    print("\n=== SANITY CHECK: tax-loss pressure (year-end, cross-sectional) ===")
    print(f"  NaN in June (absent off-season); in {dsel.date()} the YTD loser S0 tops "
          f"tax_loss_pressure (std={row.std():.3f}>0 -> dispersed, not a flat calendar dummy). Validated.")


def test_tax_loss_pressure_absent_off_season_and_dispersed_in_season():
    """Regression for the 'linear downward trend with the years' bug. Root cause: the feature was
    0 for ~75% of cells (off-season + YTD winners), so the per-day ranker collapsed it to a
    near-constant ~0.5 whose ONLY variation was the rank's universal +1/(2N) mean drift (which
    grows a spurious year-trend as the universe expands). The fix makes it NaN off-season, so the
    standardized feature is ABSENT off-season and carries a REAL loser-vs-winner spread in-season.
    (The residual mean-level drift is the same benign artifact every rank feature has and is NOT
    asserted here — the model uses within-day ordering, not the mean.)"""
    dates = pd.bdate_range("2019-01-02", periods=4 * 252)
    tickers = ["LOSER", "WINNER"] + [f"S{i}" for i in range(6)]
    rng = np.random.default_rng(1)
    ret = pd.DataFrame(rng.normal(0.0003, 0.015, (len(dates), len(tickers))),
                       index=dates, columns=tickers)
    ret["LOSER"] = -0.004                                  # persistent YTD loser
    ret["WINNER"] = +0.004                                 # persistent YTD winner
    close = 100 * (1 + ret).cumprod()
    open_ = close.shift(1).bfill()
    sector = pd.DataFrame(0.0, index=dates, columns=tickers)

    panel = build_feature_panel(close, open_, sector, method="rank")
    p = panel.assign(month=pd.to_datetime(panel["date"]).dt.month)
    tlp = p["tax_loss_pressure"]
    # (1) ABSENT off-season: all-NaN (was a ~0.5 constant that manufactured the year-trend)
    assert tlp[~p["month"].isin([10, 11, 12])].notna().sum() == 0, "must be all-NaN off-season"
    # (2) present + DISPERSED in-season, with the real signal: the YTD loser ranks ABOVE the winner
    ins = p[p["month"].isin([10, 11, 12]) & tlp.notna()]
    assert len(ins) > 0 and ins["tax_loss_pressure"].std() > 0, "in-season must be dispersed, not flat"
    day = ins["date"].iloc[-1]
    d = ins[ins["date"] == day].set_index("ticker")["tax_loss_pressure"]
    assert d.get("LOSER", 0) > d.get("WINNER", 1), f"loser must rank above winner: {d.to_dict()}"
    print("\n=== SANITY CHECK: tax-loss pressure absent off-season, real signal in-season ===")
    print(f"  standardized feature all-NaN off-season (no more ~0.5 constant / year drift); "
          f"in {pd.Timestamp(day).date()} LOSER rank {d.get('LOSER'):.2f} > WINNER {d.get('WINNER'):.2f} "
          "-> real cross-sectional loser signal. Bug fixed.")


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
    test_tax_loss_pressure_is_seasonal_and_cross_sectional()
    test_flow_is_leak_free_and_panel()
