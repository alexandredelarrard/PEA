"""
Modelling must be RuntimeWarning-free even under STRONG L1 regularization, which
shrinks the elastic-net coefficients to all-zero -> a CONSTANT prediction. That
constant flows into the per-day z-score and the daily-IC, which previously tripped
numpy's 'Mean of empty slice' / 'Degrees of freedom <= 0' / 'invalid value in
divide' warnings on degenerate (single-name / zero-dispersion) slices.

Each test promotes RuntimeWarning to an ERROR, so any regression fails loudly.
Also pins the monotone-constraint warning: quiet for a constrained feature that is
in inputs.columns but not yet in the cube (stale), loud only for a genuine typo.
"""
from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.modelling.utils_model import model as ml
from src.modelling.step_modelling import StepModelling


def test_per_day_zscore_warning_safe():
    vals = np.array([1., 2., 3., 5., 5., 9.])
    dates = np.array([1, 1, 1, 2, 2, 3])            # d1: 3 names, d2: constant, d3: single
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        z = ml.per_day_zscore(vals, dates)
    assert np.isfinite(z[:3]).all() and abs(np.nanmean(z[:3])) < 1e-9   # d1 standardized
    assert np.isnan(z[3]) and np.isnan(z[4])                            # d2 constant -> NaN
    assert np.isnan(z[5])                                              # d3 single-name -> NaN
    print("\n=== SANITY CHECK: per_day_zscore ===")
    print("  multi-name day standardized (mean~0); constant + single-name days -> NaN, "
          "no DOF/empty-slice warning. Validated.")


def test_daily_ic_constant_predictions_no_warning():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3),
        "y": [0.1, 0.5, 0.9, 0.2, 0.6, 0.8],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = ml.daily_ic(panel, pd.Series(np.full(len(panel), 0.5)), "y", horizon=60)
    assert out["n_days"] == 0 and np.isnan(out["mean_ic"]) and np.isnan(out["ic_ir"])
    print("\n=== SANITY CHECK: daily_ic on constant predictions ===")
    print("  constant member -> 0 valid IC days -> all-NaN result, no empty-slice/DOF "
          "warning. Validated.")


def test_ensemble_predict_constant_member_no_warning():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3 + ["2020-01-03"]),
        "ticker": [f"T{i}" for i in range(7)],
        "f1": [1., 2., 3., 4., 5., 6., 9.],          # day3 = a single-name day
    })

    class _Const:                                     # all-zero-coef linear -> constant
        feature_names = ["f1"]
        def predict(self, X):  # noqa: N802
            return np.zeros(len(X))

    class _Vary:
        feature_names = ["f1"]
        def predict(self, X):  # noqa: N802
            return np.asarray(X["f1"], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        preds, members = ml.ensemble_predict(
            {"elasticnet": _Const(), "lightgbm": _Vary()}, panel, ["f1"])

    assert np.isnan(members["elasticnet"].to_numpy()).all()      # constant member -> all NaN
    p = preds.to_numpy()
    assert np.isfinite(p[:6]).all() and np.isnan(p[6])           # single-name day -> NaN
    print("\n=== SANITY CHECK: ensemble_predict with a constant (strong-L1) member ===")
    print("  constant member contributes NaN; ensemble = the live member; single-name "
          "day -> NaN via manual nan-mean (no 'Mean of empty slice'). Validated.")


def test_monotone_warns_only_on_unlisted_typo(caplog):
    cfg = OmegaConf.create({"inputs": {
        "columns": ["f_a", "f_b", "f_c"],            # f_b: allow-listed but absent from cube
        "monotonic": {"enabled": True,
                      "features": [{"f_a": 1}, {"f_b": -1}, {"f_typo": 1}]}}})
    fake = SimpleNamespace(_config=cfg, _log=logging.getLogger("mono_test"),
                           _lgb_feats=lambda: ["f_a"])            # only f_a is trained
    with caplog.at_level(logging.WARNING, logger="mono_test"):
        cons = StepModelling._monotone_constraints(fake)
    assert cons == [1]                                            # aligned to lgb_feats=[f_a]
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "f_typo" in msgs                                       # genuine typo -> warned
    assert "f_b" not in msgs                                      # stale (in columns) -> quiet
    print("\n=== SANITY CHECK: monotone-skip warning ===")
    print("  warns for f_typo (not in inputs.columns); silent for f_b (allow-listed, just "
          "not in the stale cube). Validated.")


if __name__ == "__main__":
    test_per_day_zscore_warning_safe()
    test_daily_ic_constant_predictions_no_warning()
    test_ensemble_predict_constant_member_no_warning()
