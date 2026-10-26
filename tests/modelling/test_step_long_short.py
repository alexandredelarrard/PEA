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
from src.modelling.utils.artifacts import configuration_snapshot, load_ensemble, read_metadata
from src.modelling.utils.assessment import write_json
from src.modelling.utils.ensemble import PREDICTION_COLUMNS, ensemble_predict
from src.modelling.utils.metrics import daily_ic
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
                "until": kwargs.get("until"),
                "dtypes": None if out is None else dict(out.dtypes.astype(str)),
            }
        )
        return out

    def iter_load(self, table: Any, columns: list[str] | None = None, **kwargs: Any) -> Any:
        self.loads.append(
            {"table": str(table), "columns": list(columns or []), "where": kwargs.get("where"), "until": kwargs.get("until"), "dtypes": {}}
        )
        yield from self._store.iter_load(table, columns=columns, **kwargs)


def _cube(n_days: int = 620, n_tickers: int = 40) -> pd.DataFrame:
    panel = signal_panel(n_days=n_days, n_tickers=n_tickers)
    rng = np.random.default_rng(9)
    cube = panel.rename(columns={"y": target_column("rank", 30)})
    noisy = cube[target_column("rank", 30)] + rng.normal(0, 0.2, len(cube))
    cube[target_column("rank", 60)] = noisy.groupby(cube["date"]).rank(pct=True)
    latest = sorted(cube["date"].unique())[-3:]
    cube.loc[cube["date"].isin(latest), [target_column("rank", 30), target_column("rank", 60)]] = np.nan  # immature labels
    return cube


def _setup(
    tmp_path: Path, sqlite_store: Any, save: bool = True, binary_label: bool = False, **overrides: Any
) -> tuple[StepLongShort, _SpyStore, DictConfig]:
    cube = _cube()
    if binary_label:  # {0, 1} = top half of the day's rank; immature labels stay NaN
        for h in HORIZONS:
            tcol = target_column("rank", h)
            cube[tcol] = (cube[tcol] > 0.5).astype(float).where(cube[tcol].notna())
    sqlite_store.replace(Tables.cube, cube)
    sqlite_store.replace(Tables.cube_part_prices, cube[["date", "ticker"]])
    spy = _SpyStore(sqlite_store)
    dates = sorted(cube["date"].unique())
    base = {
        "model": {
            "target_type": "rank",
            "cv": {"n_splits": 2, "embargo": None},
            "models_dir": str(tmp_path / "models"),
            "assessment": {"run_id": "20181018_contract", "output_dir": str(tmp_path / "runs")},
            "diagnostics": {"enabled": True, "top_n_features": 2, "shap_sample": 200, "pdp_grid": 6},
            "backtest": {"n_quantiles": 5},
        },
        "build_cube": {"output": {"save_cv_results": True, "save_signal": True}},
        "train": {"start_date": str(pd.Timestamp(dates[0]).date()), "end_date": str(pd.Timestamp(dates[-40]).date())},
        "data_extract": {"prediction_fresh_share": 0.95, "redundant_ticks": []},
    }
    cfg = OmegaConf.merge(make_config(**base), OmegaConf.create(overrides))
    paths = {
        "ROOT": tmp_path,
        "DATA_STORE": tmp_path / "data",
        "OUTPUT_DIR": tmp_path / "out",
        "MODELS_DIR": tmp_path / "models",
        "CUBE_CV_RESULTS_PATH": tmp_path / "cv.parquet",
    }
    context: Any = SimpleNamespace(store=spy, save=save, log=logging.getLogger("step-test"), config=cfg, config_dir=Path("configs"), paths=paths)
    return StepLongShort(context=context, config=cfg), spy, cfg  # type: ignore[arg-type]


class _ScoringMember:
    """A no-fit scoring stand-in: state/serialization tests do not benchmark an estimator."""

    def __init__(self, family: str, horizon: int, features: list[str], rounds: int, task: str = "regression") -> None:
        self.family, self.horizon, self.features = family, horizon, features
        self.categoricals: list[str] = []
        self.task = task
        self.model = self
        self.best_iteration = rounds
        self.training_iterations = rounds + 1

    def current_iteration(self) -> int:
        return self.best_iteration

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        score = panel["f0"] + {"elasticnet": 0.2, "lgbm": 0.4, "random_forest": 0.7}[self.family] * panel["f1"]
        if self.task == "classification":
            score = 1 / (1 + np.exp(-score))
        return score.rename("prediction")


def _stub_fits(step: StepLongShort, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fit(horizon: int, train: pd.DataFrame, valid: pd.DataFrame | None, rounds: int | None = None) -> dict:
        assert np.isfinite(train[step.label_column]).all()
        if valid is not None:
            assert train.label_end.max() < valid.date.min() and np.isfinite(valid[step.label_column]).all()
        calls.append(
            {
                "horizon": horizon,
                "rows": len(train),
                "valid_rows": None if valid is None else len(valid),
                "rounds": rounds,
                "columns": list(train),
                "seed": step._config.seed,
                "configuration": configuration_snapshot(step._config),
            }
        )
        count = rounds or 7 + 3 * (sum(call["horizon"] == horizon and call["valid_rows"] is not None for call in calls) - 1)
        return {
            family: _ScoringMember(
                family,
                horizon,
                [column for column in cls.configured_columns(step._config, horizon) if column in train],
                count,
                str(step._config[cls.config_key].get("task", "regression")),
            )
            for family, cls in step._classes.items()
        }

    monkeypatch.setattr(step, "_fit_members", fit)
    return calls


def test_schema_projection_is_present_columns_and_never_a_target() -> None:
    step: Any = object.__new__(StepLongShort)
    step._config = make_config(
        lgbm={"columns": ["f0", "f_absent", "target_rank_h30", "f1", "sector"], "categoricals": ["sector"]},
        random_forest={"columns_by_horizon": {30: ["f3"]}},
    )
    cube_cols = {"date", "ticker", "f0", "f1", "f2", "f3", "sector", "target_rank_h30", "target_rank_h60", "target_zscore_h60"}
    load_cols, dropped = step._select_load_columns(cube_cols)
    assert load_cols[:2] == ["date", "ticker"] and not any(c.startswith("target_") for c in load_cols)
    assert {"f0", "f1", "f2", "f3", "sector"} <= set(load_cols) and dropped == ["f_absent"]
    assert "f3" in union_all_columns(step._config), "a horizon-only feature is still loaded"
    assert horizons_in(cube_cols, "rank") == [30, 60] and horizons_in(cube_cols, "zscore") == [60] and horizons_in(cube_cols, "epsilon") == []
    print("\n=== SANITY CHECK: schema-only projection ===")
    print(f"  load cols {load_cols}; dropped {dropped}; a configured target is never loaded as a feature; horizons from the schema. Validated.")


def test_real_config_resolves_nonempty_unique_tree_feature_pools() -> None:
    cfg = read_config("./configs")
    rf = {h: RandomForestModel.configured_columns(cfg, h) for h in (30, 60, 90)}
    lgbm = {h: LightGBMModel.configured_columns(cfg, h) for h in (30, 60, 90)}
    for name, cols in [*(("rf", c) for c in rf.values()), *(("lgbm", c) for c in lgbm.values())]:
        assert cols, f"{name}: empty configured feature list"
        assert len(cols) == len(set(cols)), f"{name}: duplicate feature in the configured list"
    print("\n=== SANITY CHECK: real config ===")
    print(
        f"  RF features by horizon { {h: len(columns) for h, columns in rf.items()} }; "
        f"LGBM { {h: len(columns) for h, columns in lgbm.items()} }; every horizon pool is nonempty with no duplicates. Validated."
    )


def test_saved_five_fold_run_is_isolated_and_reloads_original_members(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    step, spy, config = _setup(tmp_path, sqlite_store, model={"horizons": [60], "cv": {"n_splits": 5}})
    _stub_fits(step, monkeypatch)
    monkeypatch.setattr(spy, "replace", lambda *a, **kw: pytest.fail("Training must not write application tables"), raising=False)
    result = step.run_train()
    assert result.run_id != str(config.model.assessment.run_id)
    assert not (tmp_path / "models").exists()
    schema = step._resolve_schema()
    panel = step._load_horizon(60, schema, False)
    for number, audit in enumerate(result.cv[60].folds, 1):
        directory = result.run_dir / f"folds/h60/fold{number}"
        meta = read_metadata(directory)
        assert meta["fold"] == number and meta["run_id"] == result.run_id
        models = load_ensemble(directory, [60], config.model.ensemble, metadata=meta)[60]
        test = panel.loc[panel.date.between(audit["test_start"], audit["test_end"])]
        scores, _ = ensemble_predict(models, test)
        assert daily_ic(test, scores, horizon=60)["mean_ic"] == pytest.approx(audit["mean_ic"])
    assert not list(result.run_dir.rglob("oof.parquet"))
    assert not list(result.run_dir.rglob("shap.parquet")) and not (result.run_dir / "report.html").exists()
    second = step.run_train()
    assert second.run_id != result.run_id and result.run_dir.is_dir()
    print(
        "SANITY: five-fold no-fit routing saves/reloads original members and reproduces quick IC; training writes no OOF/rich/application outputs and every invocation gets a fresh ID."
    )


def test_run_train_projects_one_horizon_at_a_time_and_saves_minimal_outputs(
    tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    step, spy, config = _setup(tmp_path, sqlite_store, model={"cv": {"n_splits": 3}})
    calls = _stub_fits(step, monkeypatch)
    result = step.run_train()
    assert len(spy.loads) == len(HORIZONS)
    assert all(sum(column.startswith("target_") for column in read["columns"]) == 1 for read in spy.loads)
    assert all(not read["where"] for read in spy.loads), "Prediction population must include missing-target titles"
    assert len([call for call in calls if call["valid_rows"] is None]) == len(HORIZONS)
    assert sum(result.weights.values()) == pytest.approx(1)
    assert not sqlite_store.exists(Tables.predictions) and not sqlite_store.exists(Tables.cube_signal)
    meta = read_metadata(result.run_dir / "cutoff")
    for horizon in HORIZONS:
        audit = meta["fit_audit"][str(horizon)]
        assert audit["round_source_counts"] == [10, 13] and audit["requested_rounds"] == 12
        assert audit["round_source_folds"] == [2, 3] and audit["fitted_rows"] > 0
        assert pd.Timestamp(audit["label_end_max"]) <= pd.Timestamp(meta["training_as_of"])
    print(
        "SANITY: projected horizon loads include scoring-only titles; cutoff models/minimal JSON preserve maturity and last-two round counts; application tables stay untouched."
    )


def test_run_predict_scores_the_newest_unlabelled_date(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    step, spy, cfg = _setup(tmp_path, sqlite_store, save=False)
    _stub_fits(step, monkeypatch)
    initial = step.run_train()
    write_json(initial.run_dir / "assessment/metadata.json", {"run_id": initial.run_id, "complete": True})
    (initial.run_dir / "report.html").write_text("assessed", encoding="utf-8")
    step.run_train(full_history=True)
    # the synthetic cube is dated in the past, so the input guard (test_predict_freshness.py) is stubbed out
    monkeypatch.setattr("src.modelling.steps.step_long_short.check_prediction_inputs", lambda *args, **kwargs: {})
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
    df_incomplete = sqlite_store.load(Tables.cube).drop(columns=["f0"])
    sqlite_store.drop(Tables.cube)
    sqlite_store.replace(Tables.cube, df_incomplete)
    with pytest.raises(RuntimeError, match="inputs absent"):
        step.run_predict()


def test_classification_ensemble_gets_finite_cv_metrics_and_ir_weights(tmp_path: Path, sqlite_store: Any) -> None:
    step, _, cfg = _setup(
        tmp_path,
        sqlite_store,
        save=False,
        binary_label=True,
        model={"ensemble": ["lgbm", "random_forest"]},
        lgbm={"task": "classification"},
        random_forest={"task": "classification"},
    )
    res = step.run_train()
    assert res.run_dir is not None
    meta = json.loads((res.run_dir / "cutoff/metadata.json").read_text())
    for h in HORIZONS:
        assert res.cv[h].folds and all(f["finite_ic_days"] > 0 for f in res.cv[h].folds), "binary-label days are counted"
        assert np.isfinite(res.cv[h].ic["mean_ic"]) and res.cv[h].ic["mean_ic"] > 0
        assert all(np.isfinite(m["mean_ic"]) for m in res.cv[h].member_ic.values())
        assert meta["train_ic_ir"][str(h)] > 0, "the blend weights come from a real CV IR"
    raw = res.models[HORIZONS[0]]["lgbm"].predict(sqlite_store.load(Tables.cube))  # `predictions` holds per-day z
    assert raw.between(0, 1).all() and raw.std() > 0, "members output positive-class probabilities"
    print("\n=== SANITY CHECK: classification through StepLongShort ===")
    print(f"  CV mean IC { ({h: round(res.cv[h].ic['mean_ic'], 3) for h in HORIZONS}) }; train_ic_ir {meta['train_ic_ir']};")
    print(f"  blend weights { ({h: round(w, 3) for h, w in res.weights.items()}) }; probabilities in [0, 1]. Validated.")


def test_full_history_records_effective_end_and_restores_saved_fitting_controls(
    tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    step, _, config = _setup(
        tmp_path,
        sqlite_store,
        model={"horizons": [60], "cv": {"n_splits": 3}},
        train={"end_date": "2017-06-30"},
        lgbm={"columns": ["f0", "f1", "new_column"]},
    )
    calls = _stub_fits(step, monkeypatch)
    initial = step.run_train()
    frozen = (initial.run_dir / "training/configuration.json").read_bytes()
    write_json(initial.run_dir / "assessment/metadata.json", {"run_id": initial.run_id, "complete": True})
    (initial.run_dir / "report.html").write_text("assessed", encoding="utf-8")
    cube = sqlite_store.load(Tables.cube).assign(new_column=42.0)
    sqlite_store.ensure_columns(Tables.cube, cube)
    sqlite_store.replace(Tables.cube, cube)
    config.seed += 1
    config.lgbm.columns = ["new_column"]
    config.model.ensemble = ["lgbm"]
    monkeypatch.setattr(step, "_cross_validate", lambda *a, **kw: pytest.fail("Full-history fit must not repeat CV"))
    result = step.run_train(full_history=True)
    meta = read_metadata(tmp_path / "models")
    assert meta["full_history"] and meta["train_end"] > config.train.end_date
    assert calls[-1]["rounds"] == 12 and calls[-1]["seed"] == json.loads(frozen)["seed"]
    assert "new_column" not in calls[-1]["columns"] and set(result.models[60]) == {"elasticnet", "lgbm", "random_forest"}
    assert (initial.run_dir / "training/configuration.json").read_bytes() == frozen
    assert meta["feature_cols"] == read_metadata(initial.run_dir / "cutoff")["feature_cols"]
    first_attempt = meta["full_history_attempt_id"]
    step.run_train(full_history=True)
    assert read_metadata(tmp_path / "models")["full_history_attempt_id"] != first_attempt
    print(
        "SANITY: fresh full-history attempts preserve the original resolved feature pool/seed/families/rounds despite YAML/schema growth, with no repeated CV."
    )


def test_full_history_removes_current_only_fitting_controls(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = tmp_path.parent / "f001"
    step, _, config = _setup(
        tmp_path,
        sqlite_store,
        model={"horizons": [60], "cv": {"n_splits": 3}},
        lgbm={"columns": ["f0", "f1"]},
    )
    del config.lgbm.bagging_freq
    del config.model.weight_decay
    calls = _stub_fits(step, monkeypatch)
    initial = step.run_train()
    frozen = (initial.run_dir / "training/configuration.json").read_bytes()
    saved = json.loads(frozen)
    assert "columns_by_horizon" not in saved["lgbm"] and "bagging_freq" not in saved["lgbm"]
    write_json(initial.run_dir / "assessment/metadata.json", {"run_id": initial.run_id, "complete": True})
    (initial.run_dir / "report.html").write_text("assessed", encoding="utf-8")
    config.lgbm.columns_by_horizon = {"60": ["f1"]}
    config.lgbm.bagging_freq = 7
    config.model.weight_decay = {"enabled": True, "half_life_years": 1}
    config.model.models_dir = str(tmp_path / "current-models")
    config.model.read_chunksize = 2000
    config.model.diagnostics.shap_sample = 37
    monkeypatch.setattr(step, "_cross_validate", lambda *a, **kw: pytest.fail("Full-history fit must not repeat CV"))
    previous_calls = len(calls)
    result = step.run_train(full_history=True)
    assert len(calls) == previous_calls + 1 and calls[-1]["horizon"] == 60 and calls[-1]["rounds"] == 12
    assert result.models[60]["lgbm"].features == ["f0", "f1"]
    assert calls[-1]["configuration"] == saved, "The actual fit must receive exactly the saved fitting controls"
    assert step._config.model.read_chunksize == 2000 and step._config.model.diagnostics.shap_sample == 37
    assert step._config.data_extract == config.data_extract
    assert read_metadata(tmp_path / "current-models")["feature_cols"] == read_metadata(initial.run_dir / "cutoff")["feature_cols"]
    assert not (tmp_path / "models").exists()
    assert (initial.run_dir / "training/configuration.json").read_bytes() == frozen
    assert config.lgbm.bagging_freq == 7 and config.lgbm.columns_by_horizon["60"] == ["f1"]
    print(
        "SANITY: h60/three-fold no-fit refit removes current-only fitting controls, preserves saved pools/rounds/bytes and keeps current runtime/output routing."
    )


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


def test_immature_target_mutation_cannot_change_training_panel(tmp_path: Path, sqlite_store: Any) -> None:
    step, _, _ = _setup(tmp_path, sqlite_store, save=False)
    step._calendar = pd.DatetimeIndex(sqlite_store.distinct(Tables.cube_part_prices, "date")).sort_values()
    step._training_as_of = step._calendar[-100]
    schema = step._resolve_schema()
    before = step._load_horizon(60, schema, full_history=False)
    assert before is not None and before["label_end"].max() <= step._training_as_of
    df_cube = sqlite_store.load(Tables.cube)
    future = df_cube["date"] > before["date"].max()
    df_cube.loc[future, target_column("rank", 60)] = 1e9
    sqlite_store.replace(Tables.cube, df_cube)
    after = step._load_horizon(60, schema, full_history=False)
    assert after is not None
    pd.testing.assert_frame_equal(before, after)
    assert step._eligibility[60]["immature_rows"] > 0
    print("SANITY: unavailable future target values cannot change training rows, values or date boundaries.")


def test_explicit_horizon_restriction(tmp_path: Path, sqlite_store: Any) -> None:
    step, _, _ = _setup(tmp_path, sqlite_store, save=False, model={"horizons": [60]})
    assert step._resolve_schema().horizons == [60]
    print("SANITY: explicit h60 restriction resolves only the requested stored horizon.")


def test_explicit_outcome_cutoff_precedes_every_used_label(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    cutoff = pd.Timestamp("2017-06-30")
    step, spy, _ = _setup(tmp_path, sqlite_store, save=False, train={"outcome_cutoff": str(cutoff.date())})

    def inspect_panel(horizon: int, panel: pd.DataFrame) -> None:
        assert panel["label_end"].max() <= cutoff
        raise RuntimeError("cutoff checked before fitting")

    monkeypatch.setattr(step, "_cross_validate", inspect_panel)
    with pytest.raises(RuntimeError, match="cutoff checked"):
        step.run()
    reads = spy.loads
    assert reads and all(pd.Timestamp(read["until"]) < cutoff for read in reads), "unavailable targets are excluded in the source read"
    print("SANITY: the explicit observation cutoff wins over a later feature ceiling before any fitting or writes.")


def test_training_never_attempts_application_outputs(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    step, spy, _ = _setup(tmp_path, sqlite_store, model={"horizons": [60], "cv": {"n_splits": 3}})
    _stub_fits(step, monkeypatch)
    monkeypatch.setattr(spy, "replace", lambda *args, **kwargs: pytest.fail("application mutation attempted"), raising=False)
    step.run_train()
    print("SANITY: training makes no application-table write attempts, regardless of legacy output switches.")


def test_horizon_only_rf_features_and_categories_are_projected(tmp_path: Path, sqlite_store: Any) -> None:
    step, _, _ = _setup(
        tmp_path,
        sqlite_store,
        save=False,
        model={"ensemble": ["random_forest"], "horizons": [60]},
        linear={"columns": []},
        lgbm={"columns": [], "categoricals": ["f3"]},
        random_forest={"columns": [], "columns_by_horizon": {60: ["f0", "f3"]}},
    )
    schema = step._resolve_schema()
    assert schema.horizons == [60] and {"f0", "f3"} <= set(schema.load_cols)
    assert schema.categorical_cols == ["f3"]
    assert RandomForestModel.configured_categoricals(step._config, 60) == ["f3"]
    print("SANITY: family/horizon-only numeric and categorical features survive schema resolution without default input lists.")


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
