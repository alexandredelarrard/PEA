"""Reproducibility of the modelling step: a rerun must give IDENTICAL results.

Two independent guarantees are checked:
  * purged_wf_splits produces the SAME folds every call (rules out the
    "different fold composition" hypothesis -- the CV split is deterministic).
  * train_ranker trains bit-for-bit identically on a rerun (same predictions,
    same model dump, same early-stopping best_iteration) thanks to the
    seed + deterministic + force_row_wise params. Without them, multithreaded
    LightGBM drifts and early stopping flips between 1800 and ~1 round.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modelling.long_short.utils.model import (
    train_ranker, predict, purged_wf_splits, temporal_valid_split,
)


def _synth_panel(n_days: int = 150, n_tickers: int = 80, n_feats: int = 8,
                 seed: int = 0) -> pd.DataFrame:
    """A realistic-shaped ranking panel: daily cross-sections, rank label in
    [0,1] driven by a few features plus noise. Big enough that multithreaded
    training would drift if it were not made deterministic."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    feats = [f"f{j}" for j in range(n_feats)]
    frames = []
    for d in dates:
        X = rng.normal(size=(n_tickers, n_feats))
        sig = X[:, 0] * 0.5 + X[:, 1] * 0.3 - X[:, 2] * 0.2 + rng.normal(scale=0.5, size=n_tickers)
        y = sig.argsort().argsort() / (n_tickers - 1)          # rank -> [0,1]
        block = pd.DataFrame(X, columns=feats)
        block.insert(0, "y", y)
        block.insert(0, "ticker", [f"T{i:03d}" for i in range(n_tickers)])
        block.insert(0, "date", d)
        frames.append(block)
    return pd.concat(frames, ignore_index=True), feats


def test_purged_wf_splits_is_deterministic():
    dates = pd.Series(np.repeat(pd.bdate_range("2015-01-01", periods=300), 4))
    f1 = list(purged_wf_splits(dates, n_splits=5, embargo=20))
    f2 = list(purged_wf_splits(dates, n_splits=5, embargo=20))

    assert len(f1) == len(f2) and len(f1) > 0
    for (tr1, te1), (tr2, te2) in zip(f1, f2):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)

    print("\n=== SANITY CHECK: CV fold composition is deterministic ===")
    print(f"  {len(f1)} folds identical across two calls (train/test day arrays equal). "
          "Fold composition is NOT a source of run-to-run difference. Validated.")


def test_train_ranker_reproducible_no_validation():
    panel, feats = _synth_panel()

    b1 = train_ranker(panel, feats, "y", num_boost_round=60)
    b2 = train_ranker(panel, feats, "y", num_boost_round=60)

    p1 = predict(b1, panel, feats).to_numpy()
    p2 = predict(b2, panel, feats).to_numpy()

    assert np.array_equal(p1, p2), "predictions differ between identical trainings"
    assert b1.model_to_string() == b2.model_to_string(), "model dumps differ"

    print("\n=== SANITY CHECK: training is bit-for-bit reproducible ===")
    print(f"  two trainings on identical data -> identical predictions "
          f"(max abs diff={np.abs(p1 - p2).max():.2e}) and identical model dump. Validated.")


def test_early_stopping_best_iteration_stable():
    panel, feats = _synth_panel(seed=1)
    train, valid = temporal_valid_split(panel, train_frac=0.8)

    b1 = train_ranker(train, feats, "y", valid_panel=valid, num_boost_round=300)
    b2 = train_ranker(train, feats, "y", valid_panel=valid, num_boost_round=300)

    assert b1.best_iteration == b2.best_iteration, "early-stopping round is not reproducible"
    v1 = predict(b1, valid, feats).to_numpy()
    v2 = predict(b2, valid, feats).to_numpy()
    assert np.array_equal(v1, v2)

    print("\n=== SANITY CHECK: early stopping is reproducible ===")
    print(f"  best_iteration identical on rerun (={b1.best_iteration}); "
          "no more 'sometimes 1800, sometimes 1' drift. Validated.")
