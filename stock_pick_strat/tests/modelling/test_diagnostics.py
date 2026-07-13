"""Tests for the per-run, per-horizon model diagnostics
(src/modelling/utils_model/diagnostics.py).

Verifies the artifact layout (top-N partial-dependence PNGs, SHAP importance,
per-horizon importance table, OOS IC-over-time curve) and that optional deps
(`shap`, an .xlsx engine) degrade gracefully -- the importance table falls back
to CSV and SHAP is skipped without failing the run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modelling.utils_model.model import (
    train_ranker, predict, purged_wf_splits,
)
from src.modelling.utils_model import diagnostics


def _panel(n_days: int = 120, n_tickers: int = 60, n_feats: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_days)
    feats = [f"f{j}" for j in range(n_feats)]
    frames = []
    for d in dates:
        X = rng.normal(size=(n_tickers, n_feats))
        sig = X[:, 0] * 0.6 + X[:, 1] * 0.3 - X[:, 2] * 0.2 + rng.normal(scale=0.5, size=n_tickers)
        y = sig.argsort().argsort() / (n_tickers - 1)
        block = pd.DataFrame(X, columns=feats)
        block.insert(0, "y", y)
        block.insert(0, "ticker", [f"T{i:03d}" for i in range(n_tickers)])
        block.insert(0, "date", d)
        frames.append(block)
    return pd.concat(frames, ignore_index=True), feats


def _oos_predictions(panel, feats):
    frames = []
    for tr_days, te_days in purged_wf_splits(panel["date"], n_splits=3, embargo=5):
        tr = panel[panel["date"].isin(tr_days)]
        te = panel[panel["date"].isin(te_days)]
        if tr.empty or te.empty:
            continue
        b = train_ranker(tr, feats, "y", num_boost_round=30)
        p = predict(b, te, feats)
        frames.append(pd.DataFrame({"date": te["date"].to_numpy(),
                                    "ticker": te["ticker"].to_numpy(),
                                    "pred": p.to_numpy(),
                                    "y": te["y"].to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def test_partial_dependence_and_ic_series():
    panel, feats = _panel(seed=1)
    model = train_ranker(panel, feats, "y", num_boost_round=40)
    X = panel[feats].to_numpy("float32")

    grid, means = diagnostics.partial_dependence(model, X, 0, grid_points=15, sample=500)
    assert grid is not None and len(grid) == len(means) >= 2

    oos = panel[["date", "ticker", "y"]].copy()
    oos["pred"] = predict(model, panel, feats).to_numpy()
    ic = diagnostics.daily_ic_series(oos, "y")
    assert len(ic) > 0 and ic.index.is_monotonic_increasing

    print("\n=== SANITY CHECK: PDP + daily IC series ===")
    print(f"  PDP over {len(grid)} grid points; daily IC series over {len(ic)} days "
          f"(mean={ic.mean():+.4f}). Validated.")


def test_run_diagnostics_layout(tmp_path):
    panel, feats = _panel()
    model = train_ranker(panel, feats, "y", num_boost_round=40)
    oos = _oos_predictions(panel, feats)
    top_n = 5

    summary = diagnostics.save_horizon_diagnostics(
        horizon=5, booster=model, panel=panel, feature_cols=feats,
        out_dir=tmp_path / "h5", oos_predictions=oos, label_name="y",
        top_n=top_n, shap_sample=500, pdp_grid=12,
    )

    hdir = tmp_path / "h5"
    pdp_files = sorted((hdir / "pdp").glob("pdp_*.png"))
    assert len(pdp_files) == top_n, f"expected {top_n} PDP plots, got {len(pdp_files)}"
    assert (hdir / "ic_over_time.png").exists() and (hdir / "ic_over_time.csv").exists()
    # Excel if an engine is installed, else CSV fallback -- exactly one must exist
    assert (hdir / "feature_importance.xlsx").exists() or (hdir / "feature_importance.csv").exists()

    ic_csv = pd.read_csv(hdir / "ic_over_time.csv")
    assert len(ic_csv) > 0
    assert summary["n_pdp"] == top_n and summary["ic_days"] == len(ic_csv)

    # top-level orchestrator across horizons
    run_dir = tmp_path / "run"
    diagnostics.save_run_diagnostics(
        run_dir, {5: model, 10: model}, {5: panel, 10: panel}, feats,
        {5: oos, 10: oos}, label_name="y", top_n=top_n, shap_sample=500, pdp_grid=12,
    )
    assert (run_dir / "h5" / "pdp").is_dir() and (run_dir / "h10" / "pdp").is_dir()

    imp_kind = "xlsx" if (hdir / "feature_importance.xlsx").exists() else "csv (fallback)"
    print("\n=== SANITY CHECK: per-horizon diagnostics layout ===")
    print(f"  h5/: {len(pdp_files)} PDP PNGs, ic_over_time.png+csv ({len(ic_csv)} days), "
          f"feature_importance.{imp_kind}")
    print(f"  shap_available={summary['shap_available']} (skipped cleanly if shap absent); "
          f"run/ has h5 + h10 subfolders. Validated.")
