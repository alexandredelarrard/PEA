"""Tests for the two modelling fixes:

  1. daily_ic annualizes the IC information ratio by the HORIZON (overlapping
     labels), so long horizons are no longer inflated ~sqrt(horizon).
  2. bagging_freq>0 actually activates row subsampling (`subsample`), which was
     previously a silent no-op, while staying deterministic per seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modelling.long_short.utils.model import daily_ic, train_ranker, predict


def _ic_panel(n_days: int = 80, n_tickers: int = 40, seed: int = 3):
    """Panel + prediction series whose per-day IC varies (nonzero IC std)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    frames, preds = [], []
    for d in dates:
        y = rng.normal(size=n_tickers)
        p = y * rng.uniform(0.1, 0.9) + rng.normal(scale=1.0, size=n_tickers)  # day-varying skill
        frames.append(pd.DataFrame({"date": d,
                                    "ticker": [f"T{i:03d}" for i in range(n_tickers)],
                                    "y": y}))
        preds.append(p)
    panel = pd.concat(frames, ignore_index=True)
    return panel, pd.Series(np.concatenate(preds), index=panel.index)


def _synth_panel(n_days: int = 120, n_tickers: int = 70, n_feats: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    feats = [f"f{j}" for j in range(n_feats)]
    frames = []
    for d in dates:
        X = rng.normal(size=(n_tickers, n_feats))
        sig = X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.normal(scale=0.5, size=n_tickers)
        y = sig.argsort().argsort() / (n_tickers - 1)
        block = pd.DataFrame(X, columns=feats)
        block.insert(0, "y", y)
        block.insert(0, "ticker", [f"T{i:03d}" for i in range(n_tickers)])
        block.insert(0, "date", d)
        frames.append(block)
    return pd.concat(frames, ignore_index=True), feats


def test_ic_ir_annualization_scales_with_horizon():
    panel, preds = _ic_panel()

    r1 = daily_ic(panel, preds, "y", horizon=1)
    r5 = daily_ic(panel, preds, "y", horizon=5)
    r20 = daily_ic(panel, preds, "y", horizon=20)

    # mean and std of daily IC are horizon-independent (only annualization changes)
    assert r1["mean_ic"] == r5["mean_ic"] == r20["mean_ic"]
    assert r1["ic_std"] == r5["ic_std"] == r20["ic_std"]

    # IR scales by sqrt(252/horizon): longer horizon -> SMALLER annualized IR
    assert abs(r5["ic_ir"] / r1["ic_ir"] - 1 / np.sqrt(5)) < 1e-9
    assert abs(r20["ic_ir"] / r1["ic_ir"] - 1 / np.sqrt(20)) < 1e-9
    # horizon=1 reduces to the classic sqrt(252) daily IR
    assert abs(r1["ic_ir"] - r1["mean_ic"] / r1["ic_std"] * np.sqrt(252)) < 1e-9
    assert abs(r20["ic_ir"] - r1["mean_ic"] / r1["ic_std"] * np.sqrt(252 / 20)) < 1e-9

    print("\n=== SANITY CHECK: IC_IR annualization by horizon ===")
    print(f"  mean_IC={r1['mean_ic']:+.4f} (same for all horizons); "
          f"IR: h1={r1['ic_ir']:+.2f}  h5={r5['ic_ir']:+.2f}  h20={r20['ic_ir']:+.2f}")
    print(f"  h20/h1 ratio={r20['ic_ir']/r1['ic_ir']:.4f} == 1/sqrt(20)={1/np.sqrt(20):.4f} "
          "-> long-horizon inflation removed. Validated.")


def test_bagging_freq_activates_subsample():
    panel, feats = _synth_panel()
    base = {"colsample_bytree": 1.0, "subsample": 0.6}   # isolate bagging (no feature sampling)

    def fit(seed, freq):
        b = train_ranker(panel, feats, "y", num_boost_round=40,
                         params={**base, "bagging_freq": freq, "seed": seed})
        return predict(b, panel, feats).to_numpy()

    # bagging_freq=0 -> subsample is a NO-OP -> changing the seed changes nothing
    assert np.array_equal(fit(1, 0), fit(2, 0)), "with freq=0 the model must ignore the seed"

    # bagging_freq=1 -> subsample active -> different seed samples different rows -> different model
    f1a, f1b = fit(1, 1), fit(2, 1)
    assert not np.array_equal(f1a, f1b), "bagging_freq>0 did not activate row subsampling"
    # ...but still bit-for-bit deterministic for a FIXED seed
    assert np.array_equal(f1a, fit(1, 1)), "bagging broke per-seed determinism"

    print("\n=== SANITY CHECK: bagging_freq activates subsample ===")
    print("  freq=0 -> seed ignored (subsample was a no-op); freq=1 -> seed changes the "
          "row sample (bagging live) yet stays deterministic per seed. Validated.")
