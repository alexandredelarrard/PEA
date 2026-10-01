"""Evaluation primitives (src/modelling/utils/metrics.py).

IC_IR annualisation by horizon, warning-free degenerate cases (a strongly regularised member
predicts a constant), and known-truth max drawdown / AUC."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.modelling.utils.metrics import auc, daily_ic, daily_ic_series, max_drawdown, per_day_zscore


def _ic_panel(n_days: int = 80, n_tickers: int = 40, seed: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    frames, preds = [], []
    for d in pd.bdate_range("2020-01-01", periods=n_days):
        y = rng.normal(size=n_tickers)
        frames.append(pd.DataFrame({"date": d, "ticker": [f"T{i:03d}" for i in range(n_tickers)], "y": y}))
        preds.append(y * rng.uniform(0.1, 0.9) + rng.normal(scale=1.0, size=n_tickers))  # day-varying skill
    panel = pd.concat(frames, ignore_index=True)
    return panel, pd.Series(np.concatenate(preds), index=panel.index)


def test_ic_ir_annualization_scales_with_horizon() -> None:
    panel, preds = _ic_panel()
    r1, r5, r20 = (daily_ic(panel, preds, "y", horizon=h) for h in (1, 5, 20))
    assert r1["mean_ic"] == r5["mean_ic"] == r20["mean_ic"]
    assert r1["ic_std"] == r5["ic_std"] == r20["ic_std"]
    assert abs(r5["ic_ir"] / r1["ic_ir"] - 1 / np.sqrt(5)) < 1e-9
    assert abs(r20["ic_ir"] / r1["ic_ir"] - 1 / np.sqrt(20)) < 1e-9
    assert abs(r1["ic_ir"] - r1["mean_ic"] / r1["ic_std"] * np.sqrt(252)) < 1e-9
    series = daily_ic_series(panel.assign(pred=preds.to_numpy()), "y")
    assert len(series) == r1["n_days"] and abs(series.mean() - r1["mean_ic"]) < 1e-12

    print("\n=== SANITY CHECK: IC_IR annualization by horizon ===")
    print(f"  mean_IC={r1['mean_ic']:+.4f} (all horizons); IR h1={r1['ic_ir']:+.2f} h5={r5['ic_ir']:+.2f} h20={r20['ic_ir']:+.2f}")
    print(f"  h20/h1 = {r20['ic_ir'] / r1['ic_ir']:.4f} == 1/sqrt(20); daily_ic_series mean == summary mean. Validated.")


def test_per_day_zscore_warning_safe() -> None:
    vals = np.array([1.0, 2.0, 3.0, 5.0, 5.0, 9.0])
    dates = np.array([1, 1, 1, 2, 2, 3])  # d1: 3 names, d2: constant, d3: single
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        z = per_day_zscore(vals, dates)
    assert np.isfinite(z[:3]).all() and abs(np.nanmean(z[:3])) < 1e-9
    assert np.isnan(z[3]) and np.isnan(z[4]) and np.isnan(z[5])
    print("\n=== SANITY CHECK: per_day_zscore ===")
    print("  multi-name day standardized; constant + single-name days -> NaN, no RuntimeWarning. Validated.")


def test_daily_ic_constant_predictions_no_warning() -> None:
    panel = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3), "y": [0.1, 0.5, 0.9, 0.2, 0.6, 0.8]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = daily_ic(panel, pd.Series(np.full(len(panel), 0.5)), "y", horizon=60)
    assert out["n_days"] == 0 and np.isnan(out["mean_ic"]) and np.isnan(out["ic_ir"])
    print("\n=== SANITY CHECK: daily_ic on constant predictions ===")
    print("  constant member -> 0 valid IC days -> all-NaN result, no warning. Validated.")


def test_max_drawdown_known_truth() -> None:
    curve = pd.Series([0.1, 0.3, 0.2, -0.1, 0.0, 0.4])  # peak 0.3 at pos 1 -> trough -0.1 at pos 3
    depth, length = max_drawdown(curve)
    assert np.isclose(depth, -0.4) and length == 2
    assert max_drawdown(pd.Series([0.1, 0.2, 0.3])) == (0.0, 0)
    first_down = max_drawdown(pd.Series([-0.2, -0.5]))  # falls from the implicit 0 start
    assert np.isclose(first_down[0], -0.5) and first_down[1] == 2
    assert max_drawdown(pd.Series(dtype=float)) == (0.0, 0)
    print("\n=== SANITY CHECK: max_drawdown ===")
    print(
        f"  0.3 -> -0.1 gives depth {depth:+.2f} over {length} steps; monotone/empty -> (0, 0); a curve that starts falling is measured from 0. Validated."
    )


def test_auc_known_truth() -> None:
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert np.isnan(auc(np.array([1, 1]), np.array([0.2, 0.3])))
    print("\n=== SANITY CHECK: rank AUC ===")
    print("  perfect -> 1, reversed -> 0, all ties -> 0.5, one class -> NaN. Validated.")
