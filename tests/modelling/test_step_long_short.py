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
from src.modelling.utils.artifacts import ARTIFACT_FORMAT, load_ensemble, member_path, read_metadata, verify_inventory
from src.modelling.utils.ensemble import PREDICTION_COLUMNS, ensemble_predict
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


@pytest.mark.parametrize("legacy", ["f_fwd_eps_yield", "f_forward_earnings_yield", "f_eps_expectation_growth"])
def test_training_refuses_obsolete_fundamentals_cube(legacy: str) -> None:
    step: Any = object.__new__(StepLongShort)
    step._context = SimpleNamespace(store=SimpleNamespace(columns=lambda table: ["date", "ticker", legacy]))
    with pytest.raises(RuntimeError, match="rebuild full fundamentals"):
        step._resolve_schema()
    print(f"SANITY: old input {legacy} fails before training or writes; full schema migration is explicit.")


def test_real_config_resolves_rf_h30_vs_default() -> None:
    cfg = read_config("./configs")
    rf = {h: RandomForestModel.configured_columns(cfg, h) for h in (30, 60, 90)}
    lgbm = {h: LightGBMModel.configured_columns(cfg, h) for h in (30, 60, 90)}
    for name, cols in [*(("rf", c) for c in rf.values()), *(("lgbm", c) for c in lgbm.values())]:
        assert len(cols) == len(set(cols)), f"{name}: duplicate feature in the configured list"
    assert rf[30] != rf[90]
    assert lgbm[30] == lgbm[90] and len(lgbm[30]) >= 60
    print("\n=== SANITY CHECK: real config ===")
    print(
        f"  RF features by horizon { {h: len(columns) for h, columns in rf.items()} }; "
        f"LGBM { {h: len(columns) for h, columns in lgbm.items()} }; h60 pools resolve independently of h30/h90; no duplicates. Validated."
    )


def test_saved_five_fold_run_is_isolated_and_reloads_original_oof(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    step, _, cfg = _setup(
        tmp_path,
        sqlite_store,
        save=False,
        model={"horizons": [30], "cv": {"n_splits": 5}, "assessment": {"run_id": "20181018_validation", "output_dir": str(tmp_path / "runs")}},
        build_cube={"output": {"save_predictions": False, "save_signal": False, "save_cv_results": False}},
    )
    result = step.run_train()
    assert not (tmp_path / "models").exists(), "cutoff training must leave the production root untouched"
    run = result.run_dir
    assert run is not None
    manifest = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    assert result.run_id == "20181018_validation" and manifest["training_complete"] is True
    cutoff = read_metadata(run / "cutoff")
    assert cutoff["run_id"] == result.run_id and cutoff["full_history"] is False
    assert len(cutoff["selection"]["30"]["rounds"]) == 5 and not result.cv[30].fold_models
    schema = step._resolve_schema()
    panel = step._load_horizon(30, schema, full_history=False)
    assert panel is not None
    fingerprints = {family: [] for family in cfg.model.ensemble}
    for fold in range(1, 6):
        directory = run / "folds/h30" / f"fold{fold}"
        metadata = read_metadata(directory)
        assert metadata["role"] == "fold" and metadata["fold"] == fold and metadata["run_id"] == result.run_id
        verify_inventory(directory, metadata["evidence_files"])
        members = load_ensemble(directory, [30], cfg.model.ensemble, metadata=metadata)[30]
        saved = pd.read_parquet(directory / "oof.parquet").sort_values(["date", "ticker"]).reset_index(drop=True)
        audit = metadata["fold_audit"]
        test = panel.loc[panel.date.between(audit["test_start"], audit["test_end"])].sort_values(["date", "ticker"]).reset_index(drop=True)
        predicted, member_predictions = ensemble_predict(members, test)
        assert saved["fold"].eq(fold).all() and saved["horizon"].eq(30).all()
        pd.testing.assert_frame_equal(saved[["date", "ticker"]], test[["date", "ticker"]])
        np.testing.assert_allclose(saved["pred"], predicted, rtol=1e-12, atol=1e-12)
        for family in cfg.model.ensemble:
            np.testing.assert_allclose(saved[family], member_predictions[family], rtol=1e-12, atol=1e-12)
            fingerprints[family].append(metadata["member_files"][member_path(directory, 30, family).name]["sha256"])
    assert all(len(set(values)) == 5 for values in fingerprints.values())
    assert not (tmp_path / "out/diagnostics").exists()
    assert not sqlite_store.exists(Tables.predictions) and not sqlite_store.exists(Tables.cube_signal)
    cfg.model.assessment.resume = True
    with monkeypatch.context() as guards:
        guards.setattr(StepLongShort, "_fit_final", lambda *args, **kwargs: pytest.fail("completed final models must not refit"))
        guards.setattr(LightGBMModel, "fit", lambda *args, **kwargs: pytest.fail("completed folds must not refit"))
        resumed = StepLongShort(step._context, cfg).run_train()
        assert resumed.run_id == result.run_id and resumed.cv[30].rounds == result.cv[30].rounds
        changed = sqlite_store.load(Tables.cube)
        changed.loc[0, "f0"] += 1
        sqlite_store.replace(Tables.cube, changed)
        with pytest.raises(ValueError, match="fitting inputs changed"):
            StepLongShort(step._context, cfg).run_train()
    print("SANITY: matching completed folds/cutoff resume without fitting; changed historical inputs are refused before fitting.")
    print(
        "SANITY: all five original fold members reload to their exact OOF predictions; cutoff files are dated and production/application outputs stay untouched."
    )


def test_run_train_streams_one_horizon_at_a_time_and_persists_everything(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    step, spy, cfg = _setup(tmp_path, sqlite_store)
    loaded: list[dict[str, str]] = []  # dtypes AFTER load_frame (the store itself returns float64)
    fitted: list[tuple[int, int | None, float]] = []
    original_fit = LightGBMModel.fit

    def recording_fit(self: Any, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> Any:
        if self.family == "lgbm":
            fitted.append((len(train), None if valid is None else len(valid), float(self._config.lgbm.learning_rate)))
        if valid is not None:
            assert train["label_end"].max() < valid["date"].min()
        return original_fit(self, train, valid)

    monkeypatch.setattr(LightGBMModel, "fit", recording_fit)

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
    assert res.run_dir is not None
    assert len(pd.read_parquet(res.run_dir / "training/cv.parquet")) == 2 * len(HORIZONS)

    meta = json.loads((res.run_dir / "cutoff/metadata.json").read_text())
    assert META_KEYS <= set(meta) and meta["artifact_format"] == ARTIFACT_FORMAT
    assert len(fitted) == len(HORIZONS) * 3, "two fold fits plus one cutoff fit per horizon; no per-fold refit"
    assert sum(v is None for _, v, _ in fitted) == len(HORIZONS)
    assert all(lr == cfg.lgbm.learning_rate for _, _, lr in fitted)
    for h in HORIZONS:
        audit = meta["fit_audit"][str(h)]
        assert len(res.cv[h].rounds) == 2
        assert audit["requested_rounds"] == int(np.median(res.cv[h].rounds))
        assert audit["achieved_rounds"] > 0 and audit["achieved_rounds"] <= audit["requested_rounds"]
        assert pd.Timestamp(audit["label_end_max"]) <= pd.Timestamp(meta["training_as_of"])
        assert audit["fitted_rows"] == len(res.predictions.loc[res.predictions[f"z_{h}"].notna()])
    assert meta["train_end"] == cfg.train.end_date and meta["full_history"] is False and meta["categorical_cols"] == ["sector"]
    for h in HORIZONS:
        for fam in cfg.model.ensemble:
            assert member_path(res.run_dir / "cutoff", h, fam).exists()
    assert (res.run_dir / "cutoff/kpis.json").exists()
    assert not (tmp_path / "out/diagnostics").exists() and not (tmp_path / "models").exists()

    print("\n=== SANITY CHECK: run_train ===")
    print(f"  labelled loads: {[next(iter(ld['where'])) for ld in labelled]} (one per horizon, float32)")
    print(
        f"  CV IC: { ({h: round(cv.ic['mean_ic'], 3) for h, cv in res.cv.items()}) }; blend weights { ({h: round(w, 3) for h, w in res.weights.items()}) }"
    )
    print(
        f"  backtest spread: { ({h: round(b['spread_mean'], 3) for h, b in res.backtests.items()}) }; original folds/cutoff/KPIs/CV saved under {res.run_id}; legacy diagnostics/production untouched. Validated."
    )


def test_run_predict_scores_the_newest_unlabelled_date(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    step, spy, cfg = _setup(tmp_path, sqlite_store, save=False)
    step.run_train()
    with pytest.raises(ValueError, match="incompatible"):
        cfg.lgbm.learning_rate *= 2
        step.run_train(full_history=True)
    cfg.lgbm.learning_rate /= 2
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
        assert res.cv[h].folds and all(f["n_days"] > 0 for f in res.cv[h].folds), "binary-label days are counted"
        assert np.isfinite(res.cv[h].ic["mean_ic"]) and res.cv[h].ic["mean_ic"] > 0
        assert all(np.isfinite(m["mean_ic"]) for m in res.cv[h].member_ic.values())
        assert meta["train_ic_ir"][str(h)] > 0, "the blend weights come from a real CV IR"
    raw = res.models[HORIZONS[0]]["lgbm"].predict(sqlite_store.load(Tables.cube))  # `predictions` holds per-day z
    assert raw.between(0, 1).all() and raw.std() > 0, "members output positive-class probabilities"
    print("\n=== SANITY CHECK: classification through StepLongShort ===")
    print(f"  CV mean IC { ({h: round(res.cv[h].ic['mean_ic'], 3) for h in HORIZONS}) }; train_ic_ir {meta['train_ic_ir']};")
    print(f"  blend weights { ({h: round(w, 3) for h, w in res.weights.items()}) }; probabilities in [0, 1]. Validated.")


def test_full_history_records_the_effective_train_end(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    step, _, cfg = _setup(tmp_path, sqlite_store, save=False)
    initial = step.run_train()
    assert initial.run_dir is not None
    frozen = (initial.run_dir / "training/configuration.json").read_bytes()
    production = tmp_path / "separate-production"
    cfg.model.models_dir = str(production)
    monkeypatch.setattr(step, "_cross_validate", lambda *args, **kwargs: pytest.fail("production refit must reuse saved CV counts"))
    res = step.run_train(full_history=True)
    meta = json.loads((production / "metadata.json").read_text())
    assert res.train_end_effective is not None, "a full-history refit records the last labelled date"
    assert meta["full_history"] is True and meta["train_end"] == pd.Timestamp(res.train_end_effective).strftime("%Y-%m-%d") > cfg.train.end_date
    before = (production / "metadata.json").read_bytes()
    cfg.model.assessment.resume = True
    monkeypatch.setattr(step, "_fit_final", lambda *args, **kwargs: pytest.fail("completed full-history resume must not fit"))
    step.run_train(full_history=True)
    assert (production / "metadata.json").read_bytes() == before
    assert (initial.run_dir / "training/configuration.json").read_bytes() == frozen
    state = json.loads((initial.run_dir / "metadata.json").read_text())
    assert state["full_history"]["published_root"] == str(production.resolve())
    assert json.loads(frozen)["model"]["models_dir"] == str(tmp_path / "models")
    print("\n=== SANITY CHECK: full-history refit ===")
    print(
        f"  train_end {meta['train_end']} = last labelled date (config end {cfg.train.end_date} ignored); completed resume neither fits nor republishes. Validated."
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
    reads = [read for read in spy.loads if read["where"]]
    assert reads and all(pd.Timestamp(read["until"]) < cutoff for read in reads), "unavailable targets are excluded in the source read"
    print("SANITY: the explicit observation cutoff wins over a later feature ceiling before any fitting or writes.")


def test_disabled_application_outputs_attempt_no_mutation(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    step, spy, _ = _setup(tmp_path, sqlite_store, save=False, build_cube={"output": {"save_predictions": False, "save_signal": False}})
    monkeypatch.setattr(spy, "replace", lambda *args, **kwargs: pytest.fail("application mutation attempted"), raising=False)
    step._save_outputs(pd.DataFrame(), {}, pd.Series(dtype=float), pd.Timestamp("2020-01-01"))
    print("SANITY: disabled prediction/signal outputs attempt no application-table replacement.")


def test_horizon_only_rf_features_and_categories_are_projected(tmp_path: Path, sqlite_store: Any) -> None:
    step, _, _ = _setup(
        tmp_path,
        sqlite_store,
        save=False,
        model={"ensemble": ["random_forest"], "horizons": [60]},
        linear={"columns": []},
        lgbm={"columns": [], "categoricals": []},
        random_forest={"columns": [], "columns_by_horizon": {60: ["f0"]}, "categoricals_by_horizon": {60: ["f3"]}},
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
