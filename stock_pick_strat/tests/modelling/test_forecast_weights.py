"""Tests for the correlation-aware, shrinkage horizon-blend weights
(model.optimal_forecast_weights) that replace naive w ∝ max(0, IR)/ΣIR.

Locks in the two behaviours the review targeted:
  * a horizon whose CV IR is NaN is NOT silently dropped (the 90d bug);
  * highly-correlated horizons share weight, a diversifying horizon earns more.
"""
from __future__ import annotations

import numpy as np

from src.modelling.long_short.utils import model as ml


def _corr_signals(n=4000, seed=0):
    """h30 & h60 highly correlated (same latent), h90 independent (diversifier)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    return {
        30: base + rng.normal(0, 0.2, n),
        60: base + rng.normal(0, 0.2, n),
        90: rng.normal(0, 1, n),
    }


def test_nan_ir_horizon_is_not_dropped():
    sig = _corr_signals()
    ir = {30: 2.0, 60: 1.5, 90: float("nan")}     # 90d IR unestimable
    w = ml.optimal_forecast_weights(sig, ir, shrink=0.5)

    assert set(w) == {30, 60, 90}
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w[90] > 0.05, f"NaN-IR horizon must still participate, got {w[90]:.3f}"

    # contrast with the OLD rule: NaN/0 IR -> weight 0
    old = {h: max(0.0, (ir[h] if np.isfinite(ir[h]) else 0.0)) for h in ir}
    tot = sum(old.values()); old = {h: v / tot for h, v in old.items()}
    print("\n=== SANITY CHECK: NaN-IR horizon no longer dropped ===")
    print(f"  old rule w[90]={old[90]:.3f} (dropped); new corr-aware w[90]={w[90]:.3f} "
          f"(kept via neutral prior). Full new weights: "
          f"{ {h: round(v,3) for h,v in w.items()} }. Validated.")


def test_diversifying_horizon_earns_more_than_each_correlated_one():
    sig = _corr_signals()
    ir = {30: 1.0, 60: 1.0, 90: 1.0}              # equal skill
    w = ml.optimal_forecast_weights(sig, ir, shrink=0.3)

    # h90 diversifies the correlated h30/h60 block -> higher weight than either
    assert w[90] > w[30] and w[90] > w[60], w
    assert abs(w[30] - w[60]) < 0.05, "the two correlated horizons should be ~symmetric"

    print("\n=== SANITY CHECK: correlated horizons share weight ===")
    print(f"  equal IR, but h30/h60 correlated & h90 independent -> "
          f"w={ {h: round(v,3) for h,v in w.items()} }; the diversifier (h90) is "
          f"up-weighted instead of triple-counting the correlated block. Validated.")


def test_degenerate_cases():
    sig = _corr_signals()
    assert ml.optimal_forecast_weights({30: sig[30]}, {30: 1.0}) == {30: 1.0}
    # all-NaN IR -> equal weights
    w = ml.optimal_forecast_weights(sig, {30: np.nan, 60: np.nan, 90: np.nan})
    assert all(abs(v - 1/3) < 1e-9 for v in w.values())
    # all non-positive IR -> equal weights
    w2 = ml.optimal_forecast_weights(sig, {30: -1.0, 60: 0.0, 90: -2.0})
    assert all(abs(v - 1/3) < 1e-9 for v in w2.values())
    print("\n=== SANITY CHECK: degenerate cases fall back to equal weights ===")
    print("  single horizon -> 1.0; all-NaN IR -> equal; all non-positive IR -> equal. "
          "Validated.")


if __name__ == "__main__":
    test_nan_ir_horizon_is_not_dropped()
    test_diversifying_horizon_earns_more_than_each_correlated_one()
    test_degenerate_cases()
