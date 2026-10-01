"""
Shared risk-parity primitives (src/utils/risk_parity.py), on controlled synthetic inputs:
  1. ERC equalizes each asset's risk CONTRIBUTION even when assets are correlated, whereas
     inverse-vol (equal weights here) does NOT -> why ERC diversifies better;
  2. EWMA covariance reacts to a recent volatility spike faster than a flat window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import risk_parity as rp


def _cov_from(vols: np.ndarray, corr: np.ndarray) -> np.ndarray:
    d = np.diag(vols)
    return d @ corr @ d


def test_erc_equalizes_risk_contributions_vs_inverse_vol() -> None:
    # 3 assets, EQUAL vol; assets 0 & 1 are 0.8-correlated (a beta cluster), asset 2 uncorrelated.
    vols = np.array([0.15, 0.15, 0.15])
    corr = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cov = _cov_from(vols, corr)

    iv = np.array([1 / 3, 1 / 3, 1 / 3])  # inverse-vol == equal weights (vols equal)
    rc_iv = rp.risk_contributions(cov, iv)
    w_erc = rp.erc_weights(cov)
    rc_erc = rp.risk_contributions(cov, w_erc)

    assert rc_iv.std() > 0.02, "inverse-vol should leave UNEQUAL risk contributions here"
    assert rc_erc.std() < 1e-3, "ERC must equalize risk contributions"
    assert w_erc[2] > w_erc[0] and w_erc[2] > w_erc[1], "ERC must under-weight the correlated cluster"

    print("\n=== SANITY CHECK: ERC vs inverse-vol risk contributions ===")
    print(f"  inverse-vol weights {iv.round(3)}  -> risk contrib {rc_iv.round(3)} (unequal, std={rc_iv.std():.3f})")
    print(f"  ERC weights         {w_erc.round(3)}  -> risk contrib {rc_erc.round(3)} (equal,  std={rc_erc.std():.4f})")
    print("  ERC down-weights the correlated cluster and lifts the uncorrelated diversifier. Validated.")


def test_ewma_cov_reacts_faster_than_flat_window() -> None:
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2018-01-01", periods=250)
    r = pd.Series(rng.normal(0, 0.008, len(idx)), index=idx)
    r.iloc[-20:] = pd.Series(rng.normal(0, 0.030, 20), index=idx[-20:])  # recent vol SPIKE
    win = pd.DataFrame({"a": r})
    cov_ewma, _ = rp.ewma_cov(win, halflife=20)
    cov_flat, _ = rp.cov_window(win)
    ewma_vol = float(np.sqrt(cov_ewma[0, 0]))
    flat_vol = float(np.sqrt(cov_flat[0, 0]))
    assert ewma_vol > flat_vol, "EWMA vol should weight the recent spike more than a flat window"

    print("\n=== SANITY CHECK: EWMA vs flat-window vol ===")
    print(f"  after a recent vol spike: EWMA ann-vol={ewma_vol * 100:.1f}%  flat 250d={flat_vol * 100:.1f}%  (EWMA reacts faster). Validated.")
