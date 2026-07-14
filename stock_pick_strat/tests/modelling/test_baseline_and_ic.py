"""Tests for the low-SNR-target training changes:

  * IC-based early stopping (eval_metric="ic") -> LightGBM keeps training past a
    couple of rounds and picks a reproducible best_iteration (RMSE stopped ~1).
  * ridge linear baseline achieves OOS IC in the same ballpark as LightGBM on a
    (weak, linear) cross-sectional signal -- the "is the GBDT even worth it?"
    benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import warnings

from src.modelling.utils_model import model as ml
from src.modelling.utils_model import baselines


def test_standardize_no_warning_on_all_nan_column():
    """A feature that is entirely NaN in a fold must not emit numpy
    'Mean of empty slice' / 'Degrees of freedom <= 0' RuntimeWarnings, and its
    standardized column becomes all zeros (contributes nothing)."""
    X = np.array([[1.0, np.nan, 5.0],
                  [2.0, np.nan, 6.0],
                  [3.0, np.nan, 7.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # any RuntimeWarning -> failure
        Xs, mean, std = baselines._standardize(X)

    assert np.allclose(Xs[:, 1], 0.0), "all-NaN column should standardize to zeros"
    assert mean[1] == 0.0 and std[1] == 1.0
    assert np.isfinite(Xs).all()

    print("\n=== SANITY CHECK: standardize handles all-NaN column ===")
    print("  all-NaN feature -> no RuntimeWarning, mean=0/std=1, column -> zeros. Validated.")


def _panel(T: int = 300, N: int = 80, K: int = 8, noise: float = 1.5, seed: int = 0):
    """Daily cross-sections with a weak LINEAR signal + heavy noise -> a realistic
    low-IC target both a linear model and a GBDT can partly capture."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=T)
    tickers = [f"T{i:02d}" for i in range(N)]
    feats = [f"f{j}" for j in range(K)]
    w = np.array([0.6, 0.35, -0.25, 0.15, 0.0, 0.0, 0.0, 0.0])[:K]
    frames = []
    for d in dates:
        X = rng.normal(0, 1, (N, K))
        raw = X @ w + rng.normal(0, noise, N)
        y = pd.Series(raw).rank(pct=True).to_numpy()      # rank target in [0,1]
        block = pd.DataFrame(X, columns=feats)
        block.insert(0, "y", y)
        block.insert(0, "ticker", tickers)
        block.insert(0, "date", d)
        frames.append(block)
    return pd.concat(frames, ignore_index=True), feats


def test_ic_early_stopping_does_not_collapse():
    panel, feats = _panel()
    tr, val = ml.temporal_valid_split(panel, train_frac=0.8)

    b = ml.train_ranker(tr, feats, "y", valid_panel=val,
                        num_boost_round=400, early_stopping_rounds=50, eval_metric="ic")
    b2 = ml.train_ranker(tr, feats, "y", valid_panel=val,
                         num_boost_round=400, early_stopping_rounds=50, eval_metric="ic")

    assert b.best_iteration > 1, "IC early stopping collapsed to ~1 round"
    assert b.best_iteration == b2.best_iteration, "IC early stopping not reproducible"

    print("\n=== SANITY CHECK: IC-based early stopping ===")
    print(f"  best_iteration={b.best_iteration} (>1, reproducible) -> training no longer "
          "stops after ~1 RMSE-flat round.")


def _cv_ic(panel, feats, fit_fn, n_splits=4, embargo=5):
    ics = []
    for tr_d, te_d in ml.purged_wf_splits(panel["date"], n_splits=n_splits, embargo=embargo):
        tr = panel[panel["date"].isin(tr_d)]
        te = panel[panel["date"].isin(te_d)]
        if tr.empty or te.empty:
            continue
        sub_tr, sub_val = ml.temporal_valid_split(tr)
        model = fit_fn(sub_tr, sub_val)
        preds = ml.predict(model, te, feats)
        ics.append(ml.daily_ic(te, preds, "y", horizon=1)["mean_ic"])
    return float(np.nanmean(ics))


def test_ridge_baseline_close_to_lightgbm():
    panel, feats = _panel()

    lgb_ic = _cv_ic(panel, feats, lambda tr, val: ml.train_ranker(
        tr, feats, "y", valid_panel=val, num_boost_round=400,
        early_stopping_rounds=50, eval_metric="ic"))
    ridge_ic = _cv_ic(panel, feats, lambda tr, val: baselines.train_ridge(
        tr, feats, "y", alpha=10.0))

    assert lgb_ic > 0 and ridge_ic > 0, "both models should show positive OOS IC on real signal"
    # "close": the ridge baseline is within striking distance of the GBDT
    assert abs(ridge_ic - lgb_ic) < 0.05, f"ridge {ridge_ic:.4f} vs lgb {lgb_ic:.4f} not close"
    assert ridge_ic >= 0.7 * lgb_ic, "ridge much worse than lgb -> baseline not competitive"

    print("\n=== SANITY CHECK: ridge baseline vs LightGBM (OOS IC) ===")
    print(f"  purged-CV mean IC:  LightGBM={lgb_ic:+.4f}   ridge={ridge_ic:+.4f}   "
          f"|diff|={abs(ridge_ic - lgb_ic):.4f}")
    print("  -> the linear baseline is competitive with the GBDT on this target. Validated.")


def test_elasticnet_selects_where_ridge_keeps_everything():
    """With a correlated + mostly-irrelevant feature set, elastic net's L1 drives
    weak coefficients to EXACTLY zero (selection) while ridge keeps them all."""
    rng = np.random.default_rng(0)
    n, K = 3000, 10
    X = rng.normal(0, 1, (n, K))
    X[:, 1] = X[:, 0] + rng.normal(0, 0.05, n)        # f1 nearly collinear with f0
    y = 1.5 * X[:, 0] + rng.normal(0, 1.0, n)          # only the f0/f1 cluster matters
    feats = [f"f{j}" for j in range(K)]
    panel = pd.DataFrame(X, columns=feats)
    panel["y"] = y

    en = baselines.train_elasticnet(panel, feats, "y", alpha=0.1, l1_ratio=0.7,
                                    max_iter=3000, tol=1e-9)
    rg = baselines.train_ridge(panel, feats, "y", alpha=10.0)

    en_zeros = int((np.abs(en.coef) < 1e-8).sum())
    rg_zeros = int((np.abs(rg.coef) < 1e-8).sum())

    assert en_zeros >= 4, f"elastic net should zero irrelevant features (got {en_zeros})"
    assert rg_zeros == 0, "ridge should keep every coefficient nonzero (no selection)"
    # the informative (correlated) cluster carries essentially all the weight
    assert abs(en.coef[0]) + abs(en.coef[1]) > np.abs(en.coef[2:]).sum()

    print("\n=== SANITY CHECK: elastic net selection vs ridge ===")
    print(f"  zero coefficients: elasticnet={en_zeros}/{K}, ridge={rg_zeros}/{K}")
    print(f"  informative f0/f1 |coef| = {abs(en.coef[0]) + abs(en.coef[1]):.2f} "
          f"vs noise sum {np.abs(en.coef[2:]).sum():.3f} -> L1 selects, L2 shares. Validated.")


def test_elasticnet_baseline_close_to_lightgbm():
    panel, feats = _panel()

    lgb_ic = _cv_ic(panel, feats, lambda tr, val: ml.train_ranker(
        tr, feats, "y", valid_panel=val, num_boost_round=400,
        early_stopping_rounds=50, eval_metric="ic"))
    en_ic = _cv_ic(panel, feats, lambda tr, val: baselines.train_elasticnet(
        tr, feats, "y", alpha=0.001, l1_ratio=0.5, max_iter=1000))

    assert lgb_ic > 0 and en_ic > 0
    assert abs(en_ic - lgb_ic) < 0.05, f"elasticnet {en_ic:.4f} vs lgb {lgb_ic:.4f} not close"
    assert en_ic >= 0.7 * lgb_ic

    print("\n=== SANITY CHECK: elastic-net baseline vs LightGBM (OOS IC) ===")
    print(f"  purged-CV mean IC:  LightGBM={lgb_ic:+.4f}   elasticnet={en_ic:+.4f}   "
          f"|diff|={abs(en_ic - lgb_ic):.4f}  -> competitive. Validated.")
