"""Tests for the rolling multi-factor beta estimator (src/data_aggregate/utils/betas.py).

Covers:
  1. Correctness   -> ridge recovers KNOWN loadings on synthetic data.
  2. Ridge vs OLS  -> ridge betas are materially more stable under collinearity.
  3. No look-ahead -> betas at date t are invariant to data after t.
  4. Coverage      -> a sparse regressor must not silently truncate the whole
                      history (regression test for the sector-NaN bug).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.target.betas import estimate_betas_for_stock


# --------------------------------------------------------------------------- #
# 1. Correctness: recover known loadings                                       #
# --------------------------------------------------------------------------- #
def test_ridge_recovers_known_betas(synthetic_factor_model):
    y, shared, sector, true_betas = synthetic_factor_model

    # near-OLS (tiny ridge) + long window => estimator should recover truth.
    out = estimate_betas_for_stock(
        y, shared, sector, window=250, min_obs=200, ridge=0.05, step=1,
    )
    last = out.dropna().iloc[-1]

    recovered = {
        "market": last["beta_market"],
        "momentum": last["beta_momentum"],
        "value": last["beta_value"],
        "sector": last["beta_sector"],
    }
    for name, truth in true_betas.items():
        assert abs(recovered[name] - truth) < 0.15, (
            f"beta_{name}={recovered[name]:.3f} not within 0.15 of true {truth}"
        )

    # beta_market_simple (cov/var) should also be close to the true market beta.
    assert abs(last["beta_market_simple"] - true_betas["market"]) < 0.30

    print("\n=== SANITY CHECK: ridge recovers known betas ===")
    for name, truth in true_betas.items():
        print(f"  beta_{name:<9} recovered={recovered[name]:+.3f}  true={truth:+.2f}")
    print("  -> all loadings recovered within tolerance. Estimator is correct.")


# --------------------------------------------------------------------------- #
# 2. Ridge vs OLS stability under collinearity                                 #
# --------------------------------------------------------------------------- #
def test_ridge_more_stable_than_ols_under_collinearity():
    """With collinear regressors the OLS split between them is unstable over
    time; ridge should reduce that temporal variance."""
    rng = np.random.default_rng(7)
    n = 600
    dates = pd.bdate_range("2017-01-01", periods=n)

    market = pd.Series(rng.normal(0, 0.01, n), index=dates, name="market")
    # sector strongly collinear with market (corr ~0.9)
    sector = pd.Series(0.9 * market.to_numpy() + rng.normal(0, 0.004, n),
                       index=dates, name="sector")
    shared = market.to_frame()

    y = 0.8 * market + 0.5 * sector + pd.Series(rng.normal(0, 0.004, n), index=dates)
    y.name = "STOCK"

    ols = estimate_betas_for_stock(y, shared, sector, window=63, min_obs=40,
                                   ridge=0.0, step=5)
    rdg = estimate_betas_for_stock(y, shared, sector, window=63, min_obs=40,
                                   ridge=5.0, step=5)

    ols_std = ols["beta_market"].std()
    rdg_std = rdg["beta_market"].std()

    assert rdg_std < ols_std, (
        f"ridge should stabilize collinear loadings: ridge std={rdg_std:.3f} "
        f"vs ols std={ols_std:.3f}"
    )

    print("\n=== SANITY CHECK: ridge vs OLS stability (collinear market/sector) ===")
    print(f"  temporal std(beta_market)  OLS={ols_std:.3f}   RIDGE={rdg_std:.3f}")
    print(f"  -> ridge cut loading volatility by {100*(1-rdg_std/ols_std):.0f}%. "
          "Ridge is the better choice here.")


# --------------------------------------------------------------------------- #
# 3. No look-ahead                                                             #
# --------------------------------------------------------------------------- #
def test_betas_have_no_lookahead(synthetic_factor_model):
    """Betas dated <= t must not depend on any observation after t."""
    y, shared, sector, _ = synthetic_factor_model
    cutoff = y.index[300]

    base = estimate_betas_for_stock(y, shared, sector, window=120, min_obs=60, step=5)

    # Corrupt everything strictly AFTER the cutoff and recompute.
    y2 = y.copy()
    y2.loc[y2.index > cutoff] = y2.loc[y2.index > cutoff] * 5.0 + 1.0
    shared2 = shared.copy()
    shared2.loc[shared2.index > cutoff] += 3.0
    sector2 = sector.copy()
    sector2.loc[sector2.index > cutoff] += 3.0

    corrupted = estimate_betas_for_stock(y2, shared2, sector2, window=120,
                                         min_obs=60, step=5)

    joint_cols = [c for c in base.columns if c != "beta_market_simple"]
    a = base.loc[base.index <= cutoff, joint_cols]
    b = corrupted.loc[corrupted.index <= cutoff, joint_cols]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-10)

    print("\n=== SANITY CHECK: no look-ahead in betas ===")
    print(f"  betas up to {cutoff.date()} unchanged after corrupting all future data.")
    print("  -> timing rule holds: betas at t use only data up to t.")


# --------------------------------------------------------------------------- #
# 4. A sparse regressor must not truncate the whole history                    #
# --------------------------------------------------------------------------- #
def test_sparse_sector_does_not_truncate_history(synthetic_factor_model):
    """Regression test for the real bug: when the sector series is NaN for the
    early part of history (a late-IPO peer), betas should still be estimated
    over the long stretch where it IS available -- not collapse to a tiny
    recent window (the estimator drops per-date NaNs, which is correct; the
    upstream sector series must simply not be all-NaN)."""
    y, shared, sector, _ = synthetic_factor_model

    sparse = sector.copy()
    sparse.iloc[: int(0.6 * len(sparse))] = np.nan  # first 60% missing

    out = estimate_betas_for_stock(y, shared, sparse, window=63, min_obs=40, step=5)
    valid = out["beta_sector"].notna()

    # ~40% of dates have a sector value; after warmup we expect a healthy chunk.
    assert valid.mean() > 0.25, (
        f"betas collapsed: only {valid.mean():.1%} of dates estimated"
    )
    first_valid = out["beta_sector"].first_valid_index()
    span_days = (out.index[-1] - first_valid).days
    assert span_days > 200, "beta coverage should span the available history"

    print("\n=== SANITY CHECK: sparse regressor does not truncate history ===")
    print(f"  sector available on last 40% of dates -> betas valid on "
          f"{valid.mean():.0%} of dates, spanning {span_days} days.")
    print("  -> estimator uses all available data instead of collapsing.")
