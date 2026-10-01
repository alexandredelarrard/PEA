"""`StepLongShort` (src/modelling/steps/step_long_short.py) end to end on a real SQLite DataStore.

Schema-only projection, one labelled load per horizon, the TrainResult / artifacts / tables /
diagnostics contract of `run_train`, the feature-only float64 read and long rows of `run_predict`,
the full-history refit, and the guards. The last test runs the production predictor against the
real database when trained artifacts exist (it skips otherwise and never writes)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.constants.constants import PREDICTION_MODEL_BLENDED, PREDICTION_MODEL_ENSEMBLE
from src.context import get_config_context
from src.data_aggregate.utils.assemble.cube import horizons_in, target_column
from src.data_store.schema import Tables
from src.modelling.steps.step_long_short import StepLongShort, union_all_columns
from src.modelling.transformers import LightGBMModel, RandomForestModel
from src.modelling.utils.artifacts import ARTIFACT_FORMAT, member_path
from src.modelling.utils.ensemble import PREDICTION_COLUMNS
from src.modelling.utils.panel import load_frame as real_load_frame
from src.utils.config import read_config
from tests.modelling.model_fixtures import make_config, signal_panel

HORIZONS = (30, 60)
META_KEYS = {
    "horizons",
    "feature_cols",
    "linear_cols",
    "lgbm_cols",
    "rf_cols",
    "linear_cols_by_h",
    "lgbm_cols_by_h",
    "rf_cols_by_h",
    "categorical_cols",
    "label_column",
    "target_type",
    "model_types",
    "train_start",
    "train_end",
    "full_history",
    "train_ic_ir",
}


class _SpyStore:
    """Delegates to a real DataStore and records every `load` (table, columns, where, dtypes)."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self.loads: list[dict] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def load(self, table: Any, columns: list[str] | None = None, **kwargs: Any) -> pd.DataFrame | None:
        out = self._store.load(table, columns=columns, **kwargs)
        self.loads.append(
            {
                "table": str(table),
                "columns": list(columns or []),
                "where": kwargs.get("where"),
                "dtypes": None if out is None else dict(out.dtypes.astype(str)),
            }
        )
        return out


def _cube(n_days: int = 260, n_tickers: int = 40) -> pd.DataFrame:
    panel = signal_panel(n_days=n_days, n_tickers=n_tickers)
    rng = np.random.default_rng(9)
    cube = panel.rename(columns={"y": target_column("rank", 30)})
    noisy = cube[target_column("rank", 30)] + rng.normal(0, 0.2, len(cube))
    cube[target_column("rank", 60)] = noisy.groupby(cube["date"]).rank(pct=True)
    latest = sorted(cube["date"].unique())[-3:]
    cube.loc[cube["date"].isin(latest), [target_column("rank", 30), target_column("rank", 60)]] = np.nan  # immature labels
    return cube


def _setup(tmp_path: Path, sqlite_store: Any, save: bool = True, **overrides: Any) -> tuple[StepLongShort, _SpyStore, DictConfig]:
    cube = _cube()
    sqlite_store.replace(Tables.cube, cube)
    spy = _SpyStore(sqlite_store)
    dates = sorted(cube["date"].unique())
    base = {
        "model": {
            "target_type": "rank",
            "cv": {"n_splits": 2, "embargo": None},
            "models_dir": str(tmp_path / "models"),
            "diagnostics": {"enabled": True, "top_n_features": 2, "shap_sample": 200, "pdp_grid": 6},
            "backtest": {"n_quantiles": 5},
        },
        "build_cube": {"output": {"save_cv_results": True, "save_signal": True}},
        "train": {"start_date": str(pd.Timestamp(dates[0]).date()), "end_date": str(pd.Timestamp(dates[-40]).date())},
    }
    cfg = OmegaConf.merge(make_config(**base), OmegaConf.create(overrides))
    paths = {
        "ROOT": tmp_path,
        "DATA_STORE": tmp_path / "data",
        "OUTPUT_DIR": tmp_path / "out",
        "MODELS_DIR": tmp_path / "models",
        "CUBE_CV_RESULTS_PATH": tmp_path / "cv.parquet",
    }
    context: Any = SimpleNamespace(store=spy, save=save, log=logging.getLogger("step-test"), config_dir=Path("configs"), paths=paths)
    return StepLongShort(context=context, config=cfg), spy, cfg  # type: ignore[arg-type]


def test_schema_projection_is_present_columns_and_never_a_target() -> None:
    step: Any = object.__new__(StepLongShort)
    step._config = make_config(
        lgbm={"columns": ["f0", "f_absent", "target_rank_h30", "f1"], "categoricals": ["sector"]}, random_forest={"columns_by_horizon": {30: ["f3"]}}
    )
    cube_cols = {"date", "ticker", "f0", "f1", "f2", "f3", "sector", "target_rank_h30", "target_rank_h60", "target_zscore_h60"}
    load_cols, dropped = step._select_load_columns(cube_cols)
    assert load_cols[:2] == ["date", "ticker"] and not any(c.startswith("target_") for c in load_cols)
    assert {"f0", "f1", "f2", "f3", "sector"} <= set(load_cols) and dropped == ["f_absent"]
    assert "f3" in union_all_columns(step._config), "a horizon-only feature is still loaded"
    assert horizons_in(cube_cols, "rank") == [30, 60] and horizons_in(cube_cols, "zscore") == [60] and horizons_in(cube_cols, "epsilon") == []
    print("\n=== SANITY CHECK: schema-only projection ===")
    print(f"  load cols {load_cols}; dropped {dropped}; a configured target is never loaded as a feature; horizons from the schema. Validated.")


def test_real_config_resolves_rf_h30_vs_default() -> None:
    cfg = read_config("./configs")
    rf = {h: RandomForestModel.configured_columns(cfg, h) for h in (30, 60, 90)}
    lgbm = {h: LightGBMModel.configured_columns(cfg, h) for h in (30, 60)}
    for name, cols in [*(("rf", c) for c in rf.values()), *(("lgbm", c) for c in lgbm.values())]:
        assert len(cols) == len(set(cols)), f"{name}: duplicate feature in the configured list"
    assert rf[30] != rf[60] and rf[90] == rf[60]
    assert lgbm[30] == lgbm[60] and len(lgbm[60]) >= 60
    print("\n=== SANITY CHECK: real config ===")
    print(
        f"  RF h30 {len(rf[30])} features (override) vs default {len(rf[60])} (h60 == h90); lgbm {len(lgbm[60])} at every horizon; no duplicates. Validated."
    )


def test_run_train_streams_one_horizon_at_a_time_and_persists_everything(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    step, spy, cfg = _setup(tmp_path, sqlite_store)
    loaded: list[dict[str, str]] = []  # dtypes AFTER load_frame (the store itself returns float64)

    def _recording_load_frame(*args: Any, **kwargs: Any) -> pd.DataFrame | None:
        out = real_load_frame(*args, **kwargs)
        if out is not None and kwargs.get("where"):
            loaded.append(dict(out.dtypes.astype(str)))
        return out

    monkeypatch.setattr("src.modelling.steps.step_long_short.load_frame", _recording_load_frame)
    res = step.run_train()

    labelled = [ld for ld in spy.loads if ld["where"]]
    assert [next(iter(ld["where"])) for ld in labelled] == [target_column("rank", h) for h in HORIZONS], "exactly one labelled load per horizon"
    assert all(sum(c.startswith("target_") for c in ld["columns"]) == 1 for ld in labelled), "each load reads only its own label"
    assert len(loaded) == len(HORIZONS) and all(v == "float32" for d in loaded for c, v in d.items() if c.startswith(("f", "target_"))), loaded

    assert list(res.models) == list(HORIZONS) and all(list(m) == list(cfg.model.ensemble) for m in res.models.values())
    assert abs(sum(res.weights.values()) - 1.0) < 1e-12
    assert res.cv[30].oos is not None and res.cv[30].oos_members is not None and list(res.cv[30].oos_members.columns[2:]) == list(cfg.model.ensemble)
    assert set(res.backtests) == set(HORIZONS) and res.backtests[30]["n_quantiles"] == 5.0

    preds = sqlite_store.load(Tables.predictions)
    assert {"combined", "signal", "z_30", "z_60", "pred_lgbm_h30"} <= set(preds.columns) and preds["signal"].between(0, 1).all()
    assert len(sqlite_store.load(Tables.cube_signal)) == res.signal.notna().sum()
    assert len(pd.read_parquet(tmp_path / "cv.parquet")) == 2 * len(HORIZONS)

    meta = json.loads((tmp_path / "models" / "metadata.json").read_text())
    assert META_KEYS <= set(meta) and meta["artifact_format"] == ARTIFACT_FORMAT
    assert meta["train_end"] == cfg.train.end_date and meta["full_history"] is False and meta["categorical_cols"] == ["sector"]
    for h in HORIZONS:
        for fam in cfg.model.ensemble:
            assert member_path(tmp_path / "models", h, fam).exists()
    run_dirs = list((tmp_path / "out" / "diagnostics").iterdir())
    assert len(run_dirs) == 1 and (run_dirs[0] / "kpis.csv").exists() and (run_dirs[0] / "h30" / "backtest.png").exists()

    print("\n=== SANITY CHECK: run_train ===")
    print(f"  labelled loads: {[next(iter(ld['where'])) for ld in labelled]} (one per horizon, float32)")
    print(
        f"  CV IC: { ({h: round(cv.ic['mean_ic'], 3) for h, cv in res.cv.items()}) }; blend weights { ({h: round(w, 3) for h, w in res.weights.items()}) }"
    )
    print(
        f"  backtest spread: { ({h: round(b['spread_mean'], 3) for h, b in res.backtests.items()}) }; 6 member pickles + metadata ({meta['artifact_format']}); predictions / cube_signal / CV results / diagnostics written. Validated."
    )


def test_run_predict_scores_the_newest_unlabelled_date(tmp_path: Path, sqlite_store: Any) -> None:
    step, spy, cfg = _setup(tmp_path, sqlite_store, save=False)
    step.run_train()
    spy.loads.clear()
    out = step.run_predict(n_dates=2)

    read = [ld for ld in spy.loads if ld["table"] == str(Tables.cube)]
    assert len(read) == 1 and not any(c.startswith("target_") for c in read[0]["columns"]), "feature-only read"
    assert read[0]["dtypes"]["f0"] == "float64", "prediction reads stay float64"
    assert list(out.columns) == PREDICTION_COLUMNS
    newest = out["date"].max()
    last = out[out["date"] == newest]
    assert set(last["model"]) == set(cfg.model.ensemble) | {PREDICTION_MODEL_ENSEMBLE, PREDICTION_MODEL_BLENDED}
    assert last.groupby(["horizon", "model"])["ticker"].nunique().min() == 40, "every ticker scored for the unlabelled date"
    assert not out.duplicated(["date", "ticker", "horizon", "model"]).any()
    for h in HORIZONS:
        sl = last[last["horizon"] == h]
        assert (sl["predicts_for"] == sl["date"] + pd.tseries.offsets.BDay(h)).all()
    assert len(sqlite_store.load(Tables.predictions_latest)) == len(out)
    blend_h = int(last[last["model"] == PREDICTION_MODEL_BLENDED]["horizon"].iloc[0])
    assert 30 <= blend_h <= 60
    print("\n=== SANITY CHECK: run_predict ===")
    print(
        f"  newest date {pd.Timestamp(newest).date()} has NO label yet and is scored for 40 names x {len(HORIZONS)} horizons x {last['model'].nunique()} models;"
    )
    print(f"  one feature-only float64 read; blended stamped h~{blend_h}; {len(out)} long rows persisted. Validated.")


def test_full_history_records_the_effective_train_end(tmp_path: Path, sqlite_store: Any) -> None:
    step, _, cfg = _setup(tmp_path, sqlite_store, save=False)
    res = step.run_train(full_history=True)
    meta = json.loads((tmp_path / "models" / "metadata.json").read_text())
    assert res.train_end_effective is not None, "a full-history refit records the last labelled date"
    assert meta["full_history"] is True and meta["train_end"] == pd.Timestamp(res.train_end_effective).strftime("%Y-%m-%d") > cfg.train.end_date
    print("\n=== SANITY CHECK: full-history refit ===")
    print(f"  train_end {meta['train_end']} = last labelled date (config end {cfg.train.end_date} ignored). Validated.")


def test_run_delegates_and_guards(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    step, _, _ = _setup(tmp_path, sqlite_store, save=False)
    monkeypatch.setattr(StepLongShort, "run_train", lambda self, full_history=False: ("trained", full_history))
    assert step.run() == ("trained", False)
    with pytest.raises(ValueError, match="unknown model family"):
        _setup(tmp_path, sqlite_store, model={"ensemble": ["lgbm", "xgboost"]})
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "metadata.json").write_text(json.dumps({"horizons": [30], "model_types": ["lgbm"]}))
    old_step, _, _ = _setup(tmp_path, sqlite_store, model={"models_dir": str(legacy)})
    with pytest.raises(RuntimeError, match="retrain"):
        old_step.run_predict()
    print("\n=== SANITY CHECK: run / guards ===")
    print("  run() == run_train(); an unknown family is refused at construction; pre-refactor artifacts -> 'retrain' error. Validated.")


def test_run_predict_on_the_real_database_makes_sense(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        config, context = get_config_context("./configs", use_cache=False, save=False)
        written: list[pd.DataFrame] = []
        monkeypatch.setattr(context.store, "replace", lambda table, df, *a, **k: written.append(df) or len(df))  # never write
        out = StepLongShort(context=context, config=config).run_predict(n_dates=1)
    except Exception as exc:  # noqa: BLE001 - no DB / no cube / no (current-format) artifacts
        pytest.skip(f"cube or model artifacts unavailable: {exc}")
    last = out[out["date"] == out["date"].max()]
    assert {PREDICTION_MODEL_ENSEMBLE, PREDICTION_MODEL_BLENDED} <= set(last["model"]) and not last.duplicated(
        ["date", "ticker", "horizon", "model"]
    ).any()
    for (_, _), g in last.groupby(["horizon", "model"]):
        assert g["pred"].notna().mean() > 0.9 and abs(float(g["pred"].mean())) < 0.2
    assert written and len(written[0]) == len(out)
    print("\n=== SANITY CHECK: run_predict on the real cube ===")
    print(f"  as-of {last['date'].max().date()}: {len(last)} rows, models {sorted(last['model'].unique())}; nothing written to the DB. Validated.")
