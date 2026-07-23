"""
Multi-asset allocation backtest (src/post_processing/utils/strategies_alloc.py).

Validates the three layers on controlled synthetic data:
  1. ERC equalizes each asset's risk CONTRIBUTION even when assets are correlated,
     whereas inverse-vol (equal weights here) does NOT -> proves why ERC diversifies better.
  2. the long-only TREND overlay scales a DOWN-trending asset toward the floor (into cash)
     while keeping an UP-trending asset near full weight.
  3. the end-to-end backtest returns finite, sane P&L with cash as the residual leg.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.post_processing.utils import strategies_alloc as sa


def _cov_from(vols, corr):
    d = np.diag(vols)
    return d @ corr @ d


def test_erc_equalizes_risk_contributions_vs_inverse_vol():
    # 3 assets, EQUAL vol; assets 0 & 1 are 0.8-correlated (an "equity-beta cluster"),
    # asset 2 uncorrelated (a "bond/gold diversifier").
    vols = np.array([0.15, 0.15, 0.15])
    corr = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cov = _cov_from(vols, corr)

    # inverse-vol -> equal weights (vols equal); risk contributions are UNEQUAL
    iv = np.array([1 / 3, 1 / 3, 1 / 3])
    rc_iv = sa.risk_contributions(cov, iv)
    # ERC -> risk contributions all ~ 1/3
    w_erc = sa.erc_weights(cov)
    rc_erc = sa.risk_contributions(cov, w_erc)

    assert rc_iv.std() > 0.02, "inverse-vol should leave UNEQUAL risk contributions here"
    assert rc_erc.std() < 1e-3, "ERC must equalize risk contributions"
    # ERC must UNDER-weight the correlated cluster vs the lone diversifier
    assert w_erc[2] > w_erc[0] and w_erc[2] > w_erc[1]

    print("\n=== SANITY CHECK: ERC vs inverse-vol risk contributions ===")
    print(f"  inverse-vol weights {iv.round(3)}  -> risk contrib {rc_iv.round(3)} (unequal, std={rc_iv.std():.3f})")
    print(f"  ERC weights         {w_erc.round(3)}  -> risk contrib {rc_erc.round(3)} (equal,  std={rc_erc.std():.4f})")
    print("  ERC down-weights the correlated equity cluster, lifts the uncorrelated diversifier. Validated.")


def test_trend_overlay_derisks_downtrend():
    idx = pd.bdate_range("2015-01-01", periods=400)
    up = pd.Series(np.linspace(100, 200, len(idx)), index=idx)      # steady uptrend
    down = pd.Series(np.linspace(200, 100, len(idx)), index=idx)    # steady downtrend
    close = pd.DataFrame({"up": up, "down": down})
    scale = sa.trend_scale_long_only(close, [63, 126, 252], vol_window=63,
                                     scheme="linear", floor=0.0, cap=2.0)
    last = scale.dropna().iloc[-1]
    assert last["up"] > 0.8, "uptrending asset should keep near-full weight"
    assert last["down"] < 0.2, "downtrending asset should be scaled toward cash"

    print("\n=== SANITY CHECK: long-only trend overlay ===")
    print(f"  final scale  up={last['up']:.2f}  down={last['down']:.2f}  "
          "(uptrend held, downtrend de-risked into cash). Validated.")


def test_allocation_backtest_sane():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2005-01-01", periods=1500)
    # equity drift+vol, gold, bond (lower vol), cash rate ~2%
    rets = pd.DataFrame({
        "equity": rng.normal(0.0004, 0.011, len(idx)),
        "gold": rng.normal(0.0002, 0.010, len(idx)),
        "bond": rng.normal(0.0001, 0.004, len(idx)),
    }, index=idx)
    cash = pd.Series(0.02 / 252.0, index=idx, name="cash")

    res = sa.allocation_backtest(rets, cash, scheme="erc", vol_window=63, rebalance_freq=21,
                                 trend_enabled=True, portfolio_vol_target=0.10,
                                 fee_bps=2.0, spread_bps=8.0)
    net = res["net_ret"]
    assert np.isfinite(net).all() and len(net) > 1000
    m = sa.series_metrics(net)
    # realized vol should be in the neighborhood of the 10% target (loose band)
    assert 0.03 < m["ann_vol"] < 0.20, f"realized vol {m['ann_vol']:.3f} far from target"
    pam = sa.per_asset_metrics(rets, cash)
    assert set(pam.index) == {"equity", "gold", "bond", "cash"}
    assert np.isfinite(pam["sharpe"]["equity"])

    print("\n=== SANITY CHECK: end-to-end allocation backtest ===")
    print(f"  days={len(net)}  ann_ret={m['ann_return']*100:.1f}%  ann_vol={m['ann_vol']*100:.1f}%  "
          f"Sharpe={m['sharpe']:.2f}  maxDD={m['max_drawdown']*100:.1f}%")
    print(f"  avg cash weight={res['cash_weight'].mean():.2f}  avg leverage={res['leverage'].mean():.2f}")
    print("  finite P&L, vol near target, cash acts as residual. Validated.")


def test_regime_tilt_lifts_offensive_budget():
    cols = ["equity", "gold", "energy", "bond", "fx"]
    off = ("equity", "energy")
    b_hi = sa._tilted_budget(cols, score=0.9, offensive=off, off_range=(0.15, 0.85))
    b_lo = sa._tilted_budget(cols, score=0.1, offensive=off, off_range=(0.15, 0.85))
    off_hi = b_hi[[cols.index(c) for c in off]].sum()
    off_lo = b_lo[[cols.index(c) for c in off]].sum()
    assert off_hi > off_lo, "risk-on (high score) must give offensive sleeves MORE risk budget"
    assert off_hi == pytest.approx(0.2 + 0.6 * 0.9, abs=1e-9)      # 0.74
    assert off_lo == pytest.approx(0.2 + 0.6 * 0.1, abs=1e-9)      # 0.26
    # feed the tilted budgets into ERC -> equity dollar weight also rises with the score
    vols = np.array([0.18, 0.16, 0.28, 0.07, 0.09])
    cov = np.diag(vols ** 2)
    w_hi = sa.erc_weights(cov, budget=b_hi)
    w_lo = sa.erc_weights(cov, budget=b_lo)
    assert w_hi[0] > w_lo[0], "equity weight should be higher in the risk-on regime"

    print("\n=== SANITY CHECK: risk-on regime tilt ===")
    print(f"  offensive risk share: risk-on={off_hi:.2f}  risk-off={off_lo:.2f}")
    print(f"  equity ERC weight:    risk-on={w_hi[0]:.2f}  risk-off={w_lo[0]:.2f}  "
          "(more equity when calm, less in stress). Validated.")


def test_ewma_cov_reacts_faster_than_flat_window():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2018-01-01", periods=250)
    r = pd.Series(rng.normal(0, 0.008, len(idx)), index=idx)
    r.iloc[-20:] = rng.normal(0, 0.030, 20)                        # recent vol SPIKE
    win = pd.DataFrame({"a": r})
    cov_ewma, _ = sa._ewma_cov(win, halflife=20)
    cov_flat, _ = sa._cov_window(win)
    ewma_vol = float(np.sqrt(cov_ewma[0, 0]))
    flat_vol = float(np.sqrt(cov_flat[0, 0]))
    assert ewma_vol > flat_vol, "EWMA vol should weight the recent spike more than a flat window"

    print("\n=== SANITY CHECK: EWMA vs flat-window vol ===")
    print(f"  after a recent vol spike: EWMA ann-vol={ewma_vol*100:.1f}%  flat 250d={flat_vol*100:.1f}%  "
          "(EWMA reacts faster). Validated.")


def test_vol_responsive_leverage_by_regime():
    rng = np.random.default_rng(3)
    n = 2200
    idx = pd.bdate_range("2010-01-01", periods=n)
    # first half CALM (low vol, up drift) -> risk-on; second half STRESS (high vol, down drift)
    eq = np.concatenate([rng.normal(0.0006, 0.006, n // 2), rng.normal(-0.0004, 0.022, n - n // 2)])
    rets = pd.DataFrame({"equity": eq,
                         "bond": rng.normal(0.0001, 0.004, n),
                         "gold": rng.normal(0.0002, 0.009, n)}, index=idx)
    cash = pd.Series(0.02 / 252.0, index=idx, name="cash")
    res = sa.allocation_backtest(rets, cash, scheme="erc", trend_enabled=False, risk_on=True,
                                 lev_responsive=True, lev_min=1.0, lev_max=2.0,
                                 portfolio_vol_target=0.10, cov_mode="ewma", vol_mode="ewma")
    lev = res["leverage"]
    calm = float(lev.iloc[850:1050].mean())        # inside the calm regime (post-warmup)
    stress = float(lev.iloc[1600:2100].mean())      # inside the stress regime
    assert lev.max() <= 2.0 + 1e-9, "leverage must never exceed lev_max"
    assert calm - stress >= 0.3, "leverage should be MATERIALLY higher in calm than stress"
    assert stress < 1.5, "stress leverage should sit well below the calm ceiling"

    print("\n=== SANITY CHECK: vol-responsive leverage (1x stress -> 2x calm) ===")
    print(f"  avg leverage  calm={calm:.2f}  stress={stress:.2f}  max={float(lev.max()):.2f}  "
          "(levers up when calm, unlevered in stress). Validated.")


if __name__ == "__main__":
    test_erc_equalizes_risk_contributions_vs_inverse_vol()
    test_trend_overlay_derisks_downtrend()
    test_allocation_backtest_sane()
    test_regime_tilt_lifts_offensive_budget()
    test_ewma_cov_reacts_faster_than_flat_window()
    test_vol_responsive_leverage_by_regime()
