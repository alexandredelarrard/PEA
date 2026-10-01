"""TEMPORARY oracle (removed with src/modelling/long_short in Phase 6 of the refactor): every
new model transformer reproduces the legacy `StepModelling` member fit bit for bit.

The legacy step is built exactly as its `_setup` builds it (configured ∩ available columns per
family and horizon, categoricals, sorted union) and fits through `_fit_one`; the transformer fits
through `fit`. Same float32 panel, same 90/10 temporal split. Predictions on a later test window
and feature importances must be IDENTICAL (np.array_equal, not allclose)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig

from src.data_aggregate.utils.assemble.cube import is_meta_column
from src.modelling.long_short.step_train import StepModelling
from src.modelling.long_short.utils import baselines, diagnostics
from src.modelling.long_short.utils import model as ml
from src.modelling.transformers import model_class
from src.modelling.transformers.monitor import Monitor
from src.modelling.utils.cv import temporal_valid_split
from src.modelling.utils.ensemble import ensemble_predict
from src.modelling.utils.features import downcast_f32
from tests.modelling.model_fixtures import ctx, make_config, signal_panel

H = 30


def _legacy_step(cfg: DictConfig, panel: pd.DataFrame, horizons: list[int]) -> StepModelling:
    s: Any = StepModelling.__new__(StepModelling)
    s._config = cfg
    s._log = logging.getLogger("legacy-parity")
    s.label_column = "y"
    s.model_types = list(cfg.model.ensemble)
    avail = {c for c in panel.columns if not is_meta_column(c) and c != "y"}
    s.linear_cols = [c for c in s._linear_columns() if c in avail]
    s.lgbm_cols = [c for c in s._lgbm_columns() if c in avail]
    s.rf_cols = [c for c in s._rf_columns() if c in avail]
    s.categorical_cols = [c for c in s._lgbm_categoricals() if c in avail]
    s.linear_cols_by_h = {h: [c for c in s._linear_columns(h) if c in avail] for h in horizons}
    s.lgbm_cols_by_h = {h: [c for c in s._lgbm_columns(h) if c in avail] for h in horizons}
    s.rf_cols_by_h = {h: [c for c in s._rf_columns(h) if c in avail] for h in horizons}
    allcols = set(s.linear_cols) | set(s.lgbm_cols) | set(s.rf_cols)
    for h in horizons:
        allcols |= set(s.linear_cols_by_h[h]) | set(s.lgbm_cols_by_h[h]) | set(s.rf_cols_by_h[h])
    s.feature_cols = sorted(allcols)
    return s


def _legacy_scores(model: Any, panel: pd.DataFrame) -> np.ndarray:
    return ml.predict(model, panel, list(model.feature_names)).to_numpy()


def _legacy_importance(model: Any) -> pd.Series:
    if hasattr(model, "feature_importance"):
        return pd.Series(ml.feature_importance(model, list(model.feature_names)), dtype=float)
    return pd.Series(baselines.linear_importance(model), dtype=float)


CASES = [
    ("elasticnet", {}),
    ("elasticnet", {"model": {"weight_decay": {"enabled": True}}}),
    ("ridge", {}),
    ("ridge", {"model": {"weight_decay": {"enabled": True}}}),
    ("lgbm", {}),
    ("lgbm", {"lgbm": {"eval_metric": "ic"}, "model": {"weight_decay": {"enabled": True}}}),
    ("random_forest", {}),
    ("random_forest", {"random_forest": {"columns_by_horizon": {30: ["f0", "f3", "f5"]}}}),
    ("random_forest", {"random_forest": {"columns": ["f1", "f4"]}, "model": {"weight_decay": {"enabled": True}}}),
    ("elasticnet", {"linear": {"columns_by_horizon": {30: ["f1", "f5"]}}}),
]


@pytest.mark.parametrize(("family", "overrides"), CASES, ids=[f"{f}-{i}" for i, (f, _) in enumerate(CASES)])
def test_transformer_matches_legacy_member_fit(family: str, overrides: dict) -> None:
    cfg = make_config(**overrides)
    panel = downcast_f32(signal_panel(n_days=140))
    dates = np.sort(panel["date"].unique())
    train_all, test = panel[panel["date"] <= dates[99]], panel[panel["date"] >= dates[110]]
    sub_tr, sub_val = temporal_valid_split(train_all)

    legacy = _legacy_step(cfg, panel, [H])
    old = legacy._fit_one(family, sub_tr, sub_val, H)
    new = model_class(family)(ctx(), cfg, family, H).fit(sub_tr, sub_val)

    assert list(new.features) == list(cast(Any, old).feature_names), "feature list or ORDER differs"
    old_pred, new_pred = _legacy_scores(old, test), new.predict(test).to_numpy()
    assert np.array_equal(old_pred, new_pred), f"max abs diff {np.max(np.abs(old_pred - new_pred)):.3g}"
    old_imp, new_imp = _legacy_importance(old), new.importance()
    assert list(old_imp.index) == list(new_imp.index) and np.array_equal(old_imp.to_numpy(), new_imp.to_numpy())

    print(f"\n=== SANITY CHECK: legacy parity {family} {overrides or '(defaults)'} ===")
    print(f"  {len(new.features)} features {new.features}; {len(test)} test rows; predictions and importances bit-identical.")


def test_monitor_matches_legacy_diagnostics(tmp_path: Path) -> None:
    """One booster (flat layout): every legacy KPI key identical, raw SHAP matrix identical."""
    diag = {"enabled": True, "top_n_features": 3, "shap_sample": 300, "pdp_grid": 8}
    cfg = make_config(model={"ensemble": ["elasticnet", "lgbm"], "diagnostics": diag})
    panel = signal_panel()
    dates = np.sort(panel["date"].unique())
    train, test = panel[panel["date"] < dates[110]], panel[panel["date"] >= dates[120]]
    sub_tr, sub_val = temporal_valid_split(train)
    members = {fam: model_class(fam)(ctx(), cfg, fam, 30).fit(sub_tr, sub_val) for fam in ("elasticnet", "lgbm")}
    blended, _ = ensemble_predict(members, test)
    oos = pd.DataFrame({"date": test["date"].to_numpy(), "ticker": test["ticker"].to_numpy(), "pred": blended.to_numpy(), "y": test["y"].to_numpy()})
    kpis = {
        "cv_mean_ic": 0.05,
        "cv_ic_ir": 1.2,
        "members": {"elasticnet": {"cv_mean_ic": 0.04, "cv_ic_ir": 1.0}, "lgbm": {"cv_mean_ic": 0.05, "cv_ic_ir": 1.1}},
    }

    monitor = Monitor(SimpleNamespace(save=True, paths={"OUTPUT_DIR": tmp_path / "new"}), cfg)  # type: ignore[arg-type]
    monitor.start_run()
    new = monitor.horizon_report(30, panel, oos, members, kpis, "y")
    booster = members["lgbm"].model
    old = diagnostics.save_horizon_diagnostics(
        horizon=30,
        booster=None,
        panel=panel,
        feature_cols=members["lgbm"].features,
        out_dir=tmp_path / "old" / "h30",
        oos_predictions=oos,
        boosters={"lgbm": booster},
        feature_cols_by_member={"lgbm": list(booster.feature_name())},
        kpis=kpis,
        label_name="y",
        top_n=3,
        shap_sample=300,
        pdp_grid=8,
        logger=logging.getLogger("legacy"),
    )
    assert new is not None
    for key, value in old.items():
        assert key in new, key
        same = new[key] == value or (isinstance(value, float) and np.isnan(value) and np.isnan(new[key]))
        assert same, f"{key}: legacy {value!r} != new {new[key]!r}"
    old_shap = pd.read_parquet(tmp_path / "old" / "h30" / "shap_values.parquet")
    new_shap = pd.read_parquet(monitor.run_dir / "h30" / "shap_values.parquet")
    pd.testing.assert_frame_equal(old_shap, new_shap)
    print("\n=== SANITY CHECK: Monitor == legacy diagnostics ===")
    print(
        f"  {len(old)} legacy KPI keys identical; raw SHAP matrix {new_shap.shape} identical; added keys: {sorted(set(new) - set(old))}. Validated."
    )
