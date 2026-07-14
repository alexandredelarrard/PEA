"""Tests for ensemble transparency:

  * ensemble_predict(return_members=True) returns the blended score AND each
    model's per-day-standardized prediction, with blended == nanmean(members).
  * daily_ic can be computed per member (the CV logging path) and separates a
    skilled member from a noise member.

Uses lightweight mock models (anything with a .predict) so no LightGBM/training
is needed -- the logic under test is pure numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modelling.utils_model import model as ml


class _LinearMock:
    """A stand-in 'model': prediction = X @ w (+ optional noise). ``predict`` is
    PURE (deterministic in X) -- the noise is reseeded each call -- so repeated
    calls return identical output (needed for the backward-compat assertion)."""
    def __init__(self, w: np.ndarray, noise: float = 0.0, seed: int = 0):
        self.w = np.asarray(w, float)
        self.noise = noise
        self.seed = seed

    def predict(self, X):
        X = np.asarray(X, float)
        out = X @ self.w
        if self.noise:
            out = out + np.random.default_rng(self.seed).normal(0, self.noise, len(out))
        return out


def _panel(T: int = 40, N: int = 30, seed: int = 1):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2021-01-01", periods=T), N)
    tickers = np.tile([f"T{i:02d}" for i in range(N)], T)
    f0 = rng.normal(0, 1, T * N)
    f1 = rng.normal(0, 1, T * N)
    # label correlates with f0 (so a model that reads f0 has real IC)
    y = f0 + rng.normal(0, 1.0, T * N)
    return pd.DataFrame({"date": dates, "ticker": tickers, "f0": f0, "f1": f1, "y": y})


def test_ensemble_returns_members_and_blend_is_their_mean():
    panel = _panel()
    feats = ["f0", "f1"]
    models = {
        "elasticnet": _LinearMock([1.0, 0.0]),          # reads the useful feature
        "lightgbm":   _LinearMock([1.0, 0.0], noise=2.0, seed=7),  # noisier copy
    }

    blended, members = ml.ensemble_predict(models, panel, feats, return_members=True)

    # members keyed by model name, aligned to panel
    assert set(members) == {"elasticnet", "lightgbm"}
    for name, s in members.items():
        assert isinstance(s, pd.Series) and len(s) == len(panel)
        assert s.index.equals(panel.index)

    # each member is per-day standardized: mean ~0, std ~1 within each date
    for s in members.values():
        by_day = pd.DataFrame({"date": panel["date"].to_numpy(), "v": s.to_numpy()})
        stats = by_day.groupby("date")["v"].agg(["mean", "std"])
        assert np.allclose(stats["mean"].to_numpy(), 0.0, atol=1e-9)
        assert np.allclose(stats["std"].to_numpy(), 1.0, atol=1e-6)

    # blended score == nanmean of the member z-scores, elementwise
    stacked = np.column_stack([members[n].to_numpy() for n in members])
    assert np.allclose(blended.to_numpy(), np.nanmean(stacked, axis=1), atol=1e-9)

    # backward compat: without the flag we still get just the Series
    only = ml.ensemble_predict(models, panel, feats)
    assert isinstance(only, pd.Series)
    assert np.allclose(only.to_numpy(), blended.to_numpy(), atol=1e-12)

    print("\n=== SANITY CHECK: ensemble_predict returns blend + members ===")
    print(f"  members={list(members)}; each per-day z (mean~0,std~1); "
          f"blend == nanmean(members) (max abs diff "
          f"{np.max(np.abs(blended.to_numpy() - np.nanmean(stacked,axis=1))):.2e}). "
          f"Backward-compatible single-return preserved. Validated.")


def test_per_member_ic_separates_skill_from_noise():
    panel = _panel()
    feats = ["f0", "f1"]
    models = {
        "skilled": _LinearMock([1.0, 0.0]),                    # reads f0 -> high IC
        "noise":   _LinearMock([0.0, 0.0], noise=1.0, seed=3),  # pure noise -> IC ~0
    }
    _, members = ml.ensemble_predict(models, panel, feats, return_members=True)

    ic_skilled = ml.daily_ic(panel, members["skilled"], "y", horizon=1)
    ic_noise = ml.daily_ic(panel, members["noise"], "y", horizon=1)

    assert ic_skilled["mean_ic"] > 0.3, ic_skilled
    assert abs(ic_noise["mean_ic"]) < 0.15, ic_noise
    assert ic_skilled["ic_ir"] > ic_noise["ic_ir"]

    print("\n=== SANITY CHECK: per-member CV IC / IC_IR is computable & meaningful ===")
    print(f"  skilled: mean_IC={ic_skilled['mean_ic']:+.3f} IC_IR={ic_skilled['ic_ir']:+.2f}; "
          f"noise: mean_IC={ic_noise['mean_ic']:+.3f} IC_IR={ic_noise['ic_ir']:+.2f}. "
          f"The CV loop can now log IC/IC_IR for each ensemble member. Validated.")


if __name__ == "__main__":
    test_ensemble_returns_members_and_blend_is_their_mean()
    test_per_member_ic_separates_skill_from_noise()
