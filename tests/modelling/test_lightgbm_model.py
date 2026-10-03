"""`LightGBMModel` and `train_booster` (src/modelling/transformers/lightgbm_model.py).

Known-truth synthetic cross-sections (tests/modelling/model_fixtures.py): the sklearn-style
surface (fit -> save -> load -> predict, evaluate), deterministic training, IC early stopping,
bagging, monotone constraints, native categoricals, classification, and the projected load."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.data_store.schema import Tables
from src.modelling.transformers import LightGBMModel
from src.modelling.transformers.lightgbm_model import train_booster
from src.modelling.utils.artifacts import save_member
from src.modelling.utils.cv import temporal_valid_split
from src.modelling.utils.metrics import daily_ic
from tests.modelling.model_fixtures import NUMERIC, ctx, make_config, signal_panel


def _fit(cfg_overrides: dict | None = None, panel: pd.DataFrame | None = None) -> tuple[LightGBMModel, pd.DataFrame, pd.DataFrame]:
    panel = signal_panel() if panel is None else panel
    dates = np.sort(panel["date"].unique())
    train, test = panel[panel["date"] < dates[120]], panel[panel["date"] >= dates[125]]
    sub_tr, sub_val = temporal_valid_split(train)
    model = LightGBMModel(ctx(), make_config(**(cfg_overrides or {})), "lgbm", 30).fit(sub_tr, sub_val)
    return model, train, test


def test_fit_save_load_predict_roundtrip_and_evaluate(tmp_path: Path) -> None:
    model, _, test = _fit()
    pred = model.predict(test)
    path = save_member(model, tmp_path / "model_h30_lgbm.pkl")
    loaded = LightGBMModel.load(path)
    assert np.array_equal(loaded.predict(test).to_numpy(), pred.to_numpy()), "pickled booster must predict identically"
    assert loaded.features == model.features and "_context" not in vars(loaded) and "_config" not in vars(loaded)
    assert pred.index.equals(test.index)
    ev = model.evaluate(test)
    assert ev == daily_ic(test, pred, "y", horizon=30) and ev["mean_ic"] > 0.1
    print("\n=== SANITY CHECK: LightGBMModel roundtrip ===")
    print(f"  {model!r}; pickle -> identical predictions; OOS mean IC {ev['mean_ic']:+.3f} (IR {ev['ic_ir']:+.2f}) == utils.daily_ic. Validated.")


def test_features_are_configured_order_present_and_never_meta() -> None:
    panel = signal_panel()
    panel["target_rank_h60"] = 0.5  # a target column can never become a feature
    cfg_cols = ["f3", "target_rank_h60", "missing_col", "f0"]
    model, _, _ = _fit({"lgbm": {"columns": cfg_cols}}, panel=panel)
    assert model.features == ["f3", "f0", "sector"], model.features
    assert model.categoricals == ["sector"]
    print("\n=== SANITY CHECK: feature resolution ===")
    print(f"  configured {cfg_cols} -> trained {model.features} (configured order, absent dropped, target excluded, categorical last). Validated.")


def test_training_is_reproducible_and_early_stopping_stable() -> None:
    a, _, test = _fit()
    b, _, _ = _fit()
    assert a.model.model_to_string() == b.model.model_to_string()
    assert a.model.best_iteration == b.model.best_iteration > 1
    assert np.array_equal(a.predict(test).to_numpy(), b.predict(test).to_numpy())
    print("\n=== SANITY CHECK: deterministic training ===")
    print(f"  two fits -> identical model dumps, best_iteration {a.model.best_iteration} both times. Validated.")


def test_ic_early_stopping_does_not_collapse() -> None:
    a, _, _ = _fit({"lgbm": {"eval_metric": "ic", "num_boost_round": 300, "early_stopping_rounds": 40}})
    b, _, _ = _fit({"lgbm": {"eval_metric": "ic", "num_boost_round": 300, "early_stopping_rounds": 40}})
    assert a.model.best_iteration > 1 and a.model.best_iteration == b.model.best_iteration
    print("\n=== SANITY CHECK: IC early stopping ===")
    print(f"  best_iteration={a.model.best_iteration} (>1, reproducible): the daily-IC metric does not stop after one RMSE-flat round. Validated.")


def test_bagging_freq_activates_subsample() -> None:
    panel = signal_panel(n_days=80, n_tickers=50, sparse=False)
    base = {"colsample_bytree": 1.0, "subsample": 0.6}

    def fit(seed: int, freq: int) -> np.ndarray:
        b = train_booster(panel, NUMERIC, "y", num_boost_round=40, params={**base, "bagging_freq": freq, "seed": seed})
        return np.asarray(b.predict(panel[NUMERIC]))

    assert np.array_equal(fit(1, 0), fit(2, 0)), "bagging_freq=0 -> subsample is a no-op"
    assert not np.array_equal(fit(1, 1), fit(2, 1)), "bagging_freq=1 -> the seed must matter"
    assert np.array_equal(fit(1, 1), fit(1, 1)), "same seed -> identical"
    print("\n=== SANITY CHECK: bagging ===")
    print("  freq=0: seed irrelevant (no bagging); freq=1: seed changes the fit; fixed seed reproducible. Validated.")


def test_monotone_constraints_reach_the_booster() -> None:
    model, _, _ = _fit()
    dumped = model.model.dump_model()["monotone_constraints"]
    expected = [1 if f == "f0" else -1 if f == "f2" else 0 for f in model.features]
    assert list(dumped) == expected
    print("\n=== SANITY CHECK: monotone constraints ===")
    print(f"  lgbm.monotonic {{f0: +1, f2: -1}} -> booster constraints {list(dumped)} aligned to {model.features}. Validated.")


def test_monotone_warns_only_on_unlisted_typo(caplog: pytest.LogCaptureFixture) -> None:
    overrides = {"lgbm": {"columns": ["f0", "f_stale"], "monotonic": {"enabled": True, "features": [{"f0": 1}, {"f_stale": -1}, {"f_typo": 1}]}}}
    with caplog.at_level(logging.WARNING, logger="src.modelling.transformers.lightgbm_model"):
        model, _, _ = _fit(overrides)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "f_typo" in msgs and "f_stale" not in msgs
    assert list(model.model.dump_model()["monotone_constraints"]) == [1, 0]
    print("\n=== SANITY CHECK: monotone warning ===")
    print("  f_typo (not in lgbm.columns) warned; f_stale (listed, absent from the panel) silent. Validated.")


def test_categorical_is_native_and_drives_gain() -> None:
    model, _, _ = _fit()
    gains = model.importance()
    assert "sector" in model.features and gains["sector"] > 0
    print("\n=== SANITY CHECK: native categorical ===")
    print(f"  'sector' is a native categorical split with gain {gains['sector']:.0f} (it shifts the label). Validated.")


def test_classification_needs_a_binary_label() -> None:
    panel = signal_panel()
    panel["y"] = (panel["y"] > 0.5).astype(int)
    model, _, test = _fit({"lgbm": {"task": "classification"}}, panel=panel)
    proba = model.predict(test)
    ev = model.evaluate(test)
    assert proba.between(0, 1).all() and ev["auc"] > 0.6
    with pytest.raises(ValueError, match="needs a \\{0, 1\\} label"):
        _fit({"lgbm": {"task": "classification"}})  # the rank label in [0, 1] is not binary
    with pytest.raises(ValueError, match="task must be"):
        LightGBMModel(ctx(), make_config(lgbm={"task": "ranking"}), "lgbm", 30)
    print("\n=== SANITY CHECK: classification ===")
    print(f"  binary label -> probabilities in [0,1], AUC {ev['auc']:.3f}; rank label -> ValueError; unknown task -> ValueError. Validated.")


def test_load_data_projects_configured_columns(sqlite_store: Any) -> None:
    panel = signal_panel(n_days=5, n_tickers=4)
    panel["unused"] = 1.0
    sqlite_store.replace(Tables.cube, panel)
    model = LightGBMModel(ctx(sqlite_store), make_config(lgbm={"columns": ["f1", "f0", "not_in_table"]}), "lgbm", 30)
    frame = model.load_data(Tables.cube)
    assert frame is not None
    assert list(frame.columns) == ["date", "ticker", "f1", "f0", "sector"]
    assert str(frame["f1"].dtype) == "float32"
    full = model.load_data(Tables.cube, columns=["date", "ticker", "f1"], downcast=False)
    assert full is not None and str(cast(pd.DataFrame, full)["f1"].dtype) == "float64"
    print("\n=== SANITY CHECK: load_data ===")
    print(
        f"  default projection {list(frame.columns)} (configured order, absent column skipped, 'unused' never read); float32 unless downcast=False. Validated."
    )
