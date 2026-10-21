"""Train purged folds with external-test early stopping, then refit from saved counts.

Production refits consume all matured labels without repeating CV. Latest prediction uses
a separate feature-only load so unlabelled current rows remain scoreable.
"""

from __future__ import annotations

import gc
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import PREDICTION_MODEL_BLENDED, PREDICTION_MODEL_ENSEMBLE
from src.context import Context
from src.data_aggregate.utils.assemble.cube import horizons_in, is_meta_column, target_column
from src.data_store.schema import Tables
from src.modelling.transformers import BaseModel, LightGBMModel, LinearRegression, RandomForestModel, model_class
from src.modelling.transformers.base import half_life_years
from src.modelling.utils.artifacts import (
    METADATA_FILE,
    configuration_snapshot,
    load_ensemble,
    member_inventory,
    member_path,
    models_dir,
    publish_generation,
    read_metadata,
    save_member,
    write_metadata,
)
from src.modelling.utils.assessment import write_json
from src.modelling.utils.cv import label_end_dates, purged_wf_splits
from src.modelling.utils.ensemble import blend_horizons, ensemble_predict, ir_horizon_weights, prediction_rows
from src.modelling.utils.metrics import daily_ic, per_day_zscore
from src.modelling.utils.panel import load_frame
from src.utils.freshness import check_prediction_inputs
from src.utils.modelling_runs import contained_path, new_run_id, read_fitting_configuration, run_directory, select_run
from src.utils.step import Step
from src.utils.universe import load_universe_tickers

# metadata.json name of each family block's resolved columns (kept for the strategies / app)
_FAMILY_BLOCKS: dict[str, type[BaseModel]] = {"linear": LinearRegression, "lgbm": LightGBMModel, "rf": RandomForestModel}
_CONFIG_KEYS = ("linear", "lgbm", "random_forest")


@dataclass
class Schema:
    """Everything resolved from the cube's column names, before any data is read."""

    horizons: list[int]
    load_cols: list[str]  # date, ticker + every configured feature / categorical the cube has
    feature_cols: list[str]  # sorted union of all families' resolved columns
    categorical_cols: list[str]
    panel_cols: list[str]
    family_cols: dict[str, list[str]]
    family_cols_by_h: dict[str, dict[int, list[str]]]


@dataclass
class HorizonCV:
    folds: list[dict] = field(default_factory=list)
    member_folds: dict[str, list[dict]] = field(default_factory=dict)
    ic: dict[str, float] = field(default_factory=dict)
    member_ic: dict[str, dict[str, float]] = field(default_factory=dict)
    rounds: list[int] = field(default_factory=list)
    audit: dict = field(default_factory=dict)


@dataclass
class TrainResult:
    models: dict[int, dict[str, BaseModel]]
    cv: dict[int, HorizonCV]
    weights: dict[int, float]
    train_end_effective: pd.Timestamp | None
    summaries: dict[int, dict]
    run_id: str
    run_dir: Path


def union_all_columns(config: DictConfig) -> list[str]:
    """Every column any family might use at ANY horizon: each block's `columns` plus all of its
    `columns_by_horizon` lists, so a horizon-only feature is still loaded from the cube."""
    out: list[str] = []
    for key in _CONFIG_KEYS:
        block = config.get(key)
        if not block:
            continue
        out += list(block.get("columns") or [])
        for cols in (block.get("columns_by_horizon") or {}).values():
            out += list(cols) if cols else []
    return list(dict.fromkeys(out))


class StepLongShort(Step):
    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self.label_column = str(config.model.label_column)
        self.target_type = str(config.model.get("target_type", "rank"))
        ensemble = config.model.get("ensemble")
        if not ensemble:
            raise ValueError("model.ensemble must list the families to train, e.g. [elasticnet, lgbm, random_forest]")
        self.families = [str(f) for f in ensemble]
        self._classes = {fam: model_class(fam) for fam in self.families}  # validates every name
        self._eligibility: dict[int, dict[str, Any]] = {}
        self._run_path: Path | None = None
        self._run_id: str | None = None
        self._run_state: dict[str, Any] = {}
        self._model_directory: Path | None = None
        self._final_state: dict[str, Any] = {}

    def run(self) -> TrainResult:
        return self.run_train()

    # ================================================================== #
    # Training                                                           #
    # ================================================================== #
    def _prepare_training(self, full_history: bool = False) -> tuple[Schema, dict | None]:
        """Resolve inputs and the observation cutoff without fitting or writing artifacts."""
        prior = read_metadata(self._run_path / "cutoff") if full_history and self._run_path is not None else None
        if prior is None:
            schema = self._resolve_schema()
        else:
            features, categories = list(prior["feature_cols"]), list(prior["categorical_cols"])
            columns = list(dict.fromkeys(features + categories))
            schema = Schema(
                list(prior["horizons"]),
                ["date", "ticker", *columns],
                features,
                categories,
                columns,
                {key: list(prior[f"{key}_cols"]) for key in _FAMILY_BLOCKS},
                {key: {int(h): list(cols) for h, cols in prior[f"{key}_cols_by_h"].items()} for key in _FAMILY_BLOCKS},
            )
        store = self._context.store
        calendar = pd.DatetimeIndex(store.distinct(Tables.cube_part_prices, "date")).normalize().sort_values().drop_duplicates()
        cube_dates = store.distinct(Tables.cube, "date", order="desc", limit=1)
        if calendar.empty or not cube_dates:
            raise ValueError("Training requires the complete price-part calendar and a populated cube")

        # what is the latest both in cube and price ?
        latest = min(calendar[-1], pd.Timestamp(cube_dates[0]))
        self._calendar = calendar[calendar <= latest]
        self._training_as_of = latest
        if not full_history and self._config.train.end_date:
            # get the dates for training
            cutoff = pd.Timestamp(self._config.train.end_date)
            sessions = calendar[calendar <= cutoff]
            if sessions.empty:
                raise ValueError("Training cutoff precedes the price calendar")

            # get the first date not overlapping with training, in validate
            endpoint = label_end_dates(calendar, max(schema.horizons)).get(sessions[-1])
            if pd.notna(endpoint):
                self._training_as_of = min(latest, endpoint)

        explicit_cutoff = self._config.train.get("outcome_cutoff")
        if explicit_cutoff is not None and not full_history:
            self._training_as_of = min(latest, pd.Timestamp(explicit_cutoff))

        if full_history:
            if self._run_path is None:
                raise ValueError("Full-history training requires an explicit saved run ID")
            assert prior is not None
            if prior.get("full_history") is not False or prior.get("run_id") != self._run_id:
                raise ValueError("Full-history training requires this run's completed cutoff selection")
        if prior is not None and not isinstance(prior.get("selection"), dict):
            raise ValueError("Saved model has no fold selection; run modelling train before full-train")
        if prior is not None and (pd.Timestamp(prior["selection_as_of"]) > self._training_as_of):
            raise ValueError("Saved fold selection is incompatible or not yet observable; run modelling train first")
        self._selection_as_of = prior["selection_as_of"] if prior else str(self._training_as_of.date())
        return schema, prior

    def run_train(self, full_history: bool = False, *, run_id: str | None = None) -> TrainResult:
        """Fit a fresh dated run, or a fresh full-history attempt from saved fitting controls."""
        self._eligibility = {}
        if full_history:
            self._run_path, self._run_state = select_run(self._context, self._config, run_id or self._run_id, assessed=True)
            self._run_id = self._run_state["run_id"]
            saved = read_fitting_configuration(self._run_path)
            restored = self._config.copy()
            for key in configuration_snapshot(restored)["model"]:
                del restored.model[key]
            for key, value in saved.pop("model").items():
                restored.model[key] = value
            for key, value in saved.items():
                restored[key] = value
            self._config = restored
            self.label_column = str(self._config.model.label_column)
            self.target_type = str(self._config.model.get("target_type", "rank"))
            self.families = list(self._run_state["model_types"])
            self._classes = {family: model_class(family) for family in self.families}
            required = set(self._run_state["feature_cols"] + self._run_state["categorical_cols"])
            if missing := required - set(self._context.store.columns(Tables.cube)):
                raise ValueError(f"Saved fitting inputs absent from cube: {sorted(missing)}")
        else:
            self._run_id = new_run_id()
            self._run_path = run_directory(self._context, self._config, self._run_id)
        schema, prior = self._prepare_training(full_history)
        self._open_run(schema, full_history)
        assert self._model_directory is not None and self._run_id is not None and self._run_path is not None
        self._log.info("Training run %s -> %s", self._run_id, self._run_path)
        half_life = half_life_years(self._config)
        if half_life is not None:
            self._log.info("Time-decay sample weights enabled (half_life=%.1f years)", half_life)
        models, cvs, train_ends = {}, {}, []
        for horizon in schema.horizons:
            panel = self._load_horizon(horizon, schema, full_history)
            if panel is None or panel.empty:
                raise ValueError(f"h{horizon}: no mature observations in the training window")
            fitting = panel.loc[np.isfinite(panel[self.label_column])]
            if fitting.empty:
                raise ValueError(f"h{horizon}: no finite mature fitting targets")
            train_ends.append(fitting.date.max())
            self._log.info("h%s: %s fitting rows, %s scoring rows, %s days", horizon, len(fitting), len(panel), panel.date.nunique())
            if prior is None:
                cv = self._cross_validate(horizon, panel)
            else:
                selected = prior["selection"][str(horizon)]
                cv = HorizonCV(
                    folds=selected["folds"],
                    rounds=selected["rounds"],
                    member_ic=selected["member_ic"],
                    ic={key: np.nan if value is None else value for key, value in selected["ic"].items()},
                )
            if "lgbm" in self.families and len(cv.rounds) != int(self._config.model.cv.n_splits):
                raise ValueError(f"h{horizon}: incomplete fold iteration counts")
            models[horizon] = self._final_horizon(horizon, fitting, cv, full_history)
            cvs[horizon] = cv
            del fitting, panel
            gc.collect()
        if not full_history and self._config.get("portfolio", {}).get("start"):
            last_label = max(pd.Timestamp(cv.audit["label_end_max"]) for cv in cvs.values())
            if pd.Timestamp(self._config.portfolio.start) <= last_label:
                raise ValueError(f"portfolio.start must be strictly after training label end {last_label.date()}")
        train_end_effective = max(train_ends)
        weights = ir_horizon_weights({h: cv.ic["ic_ir"] for h, cv in cvs.items()})
        summaries = {h: self._horizon_kpis(cv, full_history) for h, cv in cvs.items()}
        self._save_models(models, schema, cvs, full_history, train_end_effective)
        write_json(self._model_directory / "kpis.json", summaries)
        completed = datetime.now(UTC).isoformat()
        if full_history:
            production = models_dir(self._context, self._config).resolve()
            published = publish_generation(production, self._model_directory, self._run_id, self._attempt_id)
            self._final_state.update(
                {
                    "complete": True,
                    "completed_at_utc": completed,
                    "published_root": str(production),
                    "published_directory": published["member_directory"],
                }
            )
        else:
            self._run_state.update({"training_complete": True, "training_completed_at_utc": completed})
        write_json(self._run_path / METADATA_FILE, self._run_state)
        self._log.info("Training complete: run_id=%s completed_at_utc=%s blend_weights=%s", self._run_id, completed, weights)
        return TrainResult(models, cvs, weights, train_end_effective, summaries, self._run_id, self._run_path)

    def _open_run(self, schema: Schema, full_history: bool) -> None:
        """Allocate fresh fitting output; existing training is never resumed."""
        assert self._run_path is not None
        if full_history:
            self._attempt_id = uuid4().hex
            self._model_directory = contained_path(self._run_path, f"full_history/{self._attempt_id}")
            self._model_directory.mkdir(parents=True, exist_ok=False)
            self._final_state = {"attempt_id": self._attempt_id, "complete": False, "training_as_of": str(self._training_as_of.date())}
            self._run_state["full_history"] = self._final_state
        else:
            self._run_path.mkdir(parents=True, exist_ok=False)
            write_json(self._run_path / "training/configuration.json", configuration_snapshot(self._config))
            pd.DataFrame({"date": self._calendar}).to_parquet(self._run_path / "training/calendar.parquet", index=False)
            self._run_state = {
                "workflow": "fresh-assessment-v1",
                "run_id": self._run_id,
                "horizons": schema.horizons,
                "model_types": self.families,
                "feature_cols": schema.feature_cols,
                "categorical_cols": schema.categorical_cols,
                "observation_frontier": str(self._calendar[-1].date()),
                "training_as_of": str(self._training_as_of.date()),
                "target_type": self.target_type,
                "label_column": self.label_column,
                "training_complete": False,
                "folds": {},
            }
            self._model_directory = contained_path(self._run_path, "cutoff")
            self._final_state = self._run_state
        write_json(self._run_path / METADATA_FILE, self._run_state)

    def _final_horizon(self, horizon: int, panel: pd.DataFrame, cv: HorizonCV, full_history: bool) -> dict[str, BaseModel]:
        """Fit all mature labelled rows and save fresh cutoff/full-history members."""
        assert self._model_directory is not None
        members = self._fit_final(horizon, panel, cv.rounds)
        cv.audit.update(
            {
                **self._eligibility[horizon],
                "training_as_of": str(self._training_as_of.date()),
                "fitted_rows": len(panel),
                "train_days": int(panel.date.nunique()),
                "feature_min": str(panel.date.min().date()),
                "feature_max": str(panel.date.max().date()),
                "label_end_max": str(panel.label_end.max().date()),
                "round_source_folds": [len(cv.folds) - 1, len(cv.folds)] if cv.rounds else [],
                "round_source_windows": [fold["test_start"] for fold in cv.folds[-2:]] if cv.rounds else [],
                "round_source_counts": cv.rounds[-2:],
                "final_rounds": (sum(cv.rounds[-2:]) + 1) // 2 if cv.rounds else None,
                "requested_rounds": (sum(cv.rounds[-2:]) + 1) // 2 if cv.rounds else None,
                "achieved_rounds": members["lgbm"].model.current_iteration() if "lgbm" in members else None,
            }
        )
        for family, model in members.items():
            save_member(model, member_path(self._model_directory, horizon, family))
        return members

    def _resolve_schema(self) -> Schema:

        cube_cols = set(self._context.store.columns(Tables.cube))
        load_cols, dropped = self._select_load_columns(cube_cols)
        horizons = horizons_in(cube_cols, self.target_type)
        if not horizons:
            raise FileNotFoundError(
                f"The cube carries no 'target_{self.target_type}_h*' column at all. Run StepBuildCube first "
                f"(and confirm build_cube.targets.labels includes '{self.target_type}')."
            )
        restriction = self._config.model.get("horizons")
        if restriction is not None:
            requested = sorted(set(int(horizon) for horizon in restriction))
            if not requested or not set(requested) <= set(horizons):
                raise ValueError("Requested horizons must be non-empty and present in the cube")
            horizons = requested
        avail = {c for c in cube_cols if not is_meta_column(c) and c != self.label_column}
        family_cols = {key: [c for c in cls.configured_columns(self._config, None) if c in avail] for key, cls in _FAMILY_BLOCKS.items()}
        family_cols_by_h = {
            key: {h: [c for c in cls.configured_columns(self._config, h) if c in avail] for h in horizons} for key, cls in _FAMILY_BLOCKS.items()
        }
        categorical_cols = list(
            dict.fromkeys(c for cls in self._classes.values() for h in horizons for c in cls.configured_categoricals(self._config, h) if c in avail)
        )
        allcols: set[str] = set()
        for key, by_h in family_cols_by_h.items():
            if _FAMILY_BLOCKS[key].config_key not in {cls.config_key for cls in self._classes.values()}:
                continue
            for cols in by_h.values():
                allcols |= set(cols)
        feature_cols = sorted(allcols)
        if not feature_cols and not categorical_cols:
            raise ValueError("No configured features present in the cube; check linear_modelling.yml / lgbm_modelling.yml `columns`.")
        active = {"date", "ticker", *feature_cols, *categorical_cols}
        load_cols = [column for column in load_cols if column in active]
        schema = Schema(
            horizons, load_cols, feature_cols, categorical_cols, list(dict.fromkeys(feature_cols + categorical_cols)), family_cols, family_cols_by_h
        )
        self._log.info(
            "Setup: horizons=%s target_type=%s models=%s | features -> linear:%d lgbm:%d (+%d cats) union:%d",
            horizons,
            self.target_type,
            self.families,
            len(family_cols["linear"]),
            len(family_cols["lgbm"]),
            len(categorical_cols),
            len(feature_cols),
        )
        if dropped:
            # a configured column the cube lacks silently SHRINKS the feature set
            self._log.warning(
                "modelling.yml columns absent from the cube -- the feature set is SMALLER than configured (%d dropped): %s", len(dropped), dropped
            )
        return schema

    def _select_load_columns(self, cube_cols: set[str]) -> tuple[list[str], list[str]]:
        """Horizon-independent projection: date, ticker + every configured feature / categorical
        present in the cube (never a meta / target column). Returns (load_cols, dropped)."""
        requested = union_all_columns(self._config)
        feats = [c for c in requested if c in cube_cols and not is_meta_column(c)]
        dropped = [c for c in requested if c not in cube_cols]
        return list(dict.fromkeys(["date", "ticker"] + feats)), dropped

    @staticmethod
    def _load_cols_for(schema: Schema, target_col: str) -> list[str]:
        """The shared projection plus the ONE label column this horizon trains on."""
        return list(dict.fromkeys(schema.load_cols + [target_col]))

    # ---- per-horizon stages ------------------------------------------- #
    def _load_horizon(self, horizon: int, schema: Schema, full_history: bool) -> pd.DataFrame | None:
        """This horizon's labelled rows -> modelling panel (float32) within the train window."""
        store = self._context.store
        tcol = target_column(self.target_type, horizon)
        start_date = self._config.train.start_date
        end_date = None if full_history else self._config.train.end_date
        ends = label_end_dates(self._calendar, horizon)
        if self._config.train.get("outcome_cutoff") is not None and not full_history:
            mature = ends.index[ends.notna() & (ends <= self._training_as_of)]
            if mature.empty:
                return None
            end_date = min(pd.Timestamp(end_date) if end_date else mature[-1], mature[-1])
        raw = load_frame(
            store,
            Tables.cube,
            columns=self._load_cols_for(schema, tcol),
            since=start_date,
            until=end_date,
            stream=True,
            feature_cols=schema.panel_cols,
            categorical_features=schema.categorical_cols,
            chunksize=self._config.model.get("read_chunksize"),
        )
        if raw is None:
            return None
        panel = raw.rename(columns={tcol: self.label_column})
        raw = None
        panel["label_end"] = panel["date"].map(ends)
        unknown = int(panel["label_end"].isna().sum())
        immature = int((panel["label_end"] > self._training_as_of).sum())
        self._eligibility[horizon] = {
            "source_labelled_rows": int(np.isfinite(panel[self.label_column]).sum()),
            "unknown_label_end_rows": unknown,
            "immature_rows": immature,
        }
        self._log.info("h%s maturity: %s unknown endpoints, %s outcomes beyond %s excluded", horizon, unknown, immature, self._training_as_of.date())
        panel = panel.loc[panel["label_end"].notna() & (panel["label_end"] <= self._training_as_of)]
        tr = self._config.get("train", None)
        if tr:
            start_date = tr.start_date
            end_date = None if full_history else tr.end_date
            if start_date:
                panel = panel.loc[panel["date"] >= start_date]
            if end_date:
                panel = panel.loc[panel["date"] <= end_date]
        return panel if not panel.empty else None

    def _fit_members(self, horizon: int, train: pd.DataFrame, valid: pd.DataFrame | None, rounds: int | None = None) -> dict[str, BaseModel]:
        """Fresh members of every family, in `model.ensemble` order."""
        members: dict[str, BaseModel] = {}
        for fam, cls in self._classes.items():
            model = cls(self._context, self._config, fam, horizon)
            if cls is LightGBMModel and rounds is not None:
                assert isinstance(model, LightGBMModel)
                model.fit_fixed_rounds(train, rounds)
            else:
                model.fit(train, valid)
            members[fam] = model
        return members

    def _cross_validate(self, horizon: int, panel: pd.DataFrame) -> HorizonCV:
        """Fit complete purged folds; external tests select RMSE rounds and supply quick IC."""
        cfg = self._config.model.cv
        embargo = cfg.embargo or horizon
        if int(embargo) < horizon:
            raise ValueError("Assessment embargo must be at least the horizon")
        windows = cfg.get("test_windows")
        cv = HorizonCV()
        splits = purged_wf_splits(
            panel.date, int(cfg.n_splits), embargo, test_windows=windows, calendar=self._calendar if windows is not None else None
        )
        for number, (train_days, test_days) in enumerate(splits, 1):
            test = panel.loc[panel.date.isin(test_days)]
            train = panel.loc[panel.date.isin(train_days) & np.isfinite(panel[self.label_column])]
            train = train.loc[train.label_end < test.date.min()]
            valid = test.loc[np.isfinite(test[self.label_column])]
            if train.empty or valid.empty:
                raise ValueError(f"h{horizon} fold{number}: no eligible fitting/evaluation rows")
            fitted = self._fit_members(horizon, train, valid)
            if "lgbm" in fitted:
                booster = fitted["lgbm"].model
                cv.rounds.append(int(booster.best_iteration or booster.current_iteration()))
            prediction, member_scores = ensemble_predict(fitted, test)
            audit = daily_ic(test, prediction, self.label_column, horizon=horizon)
            audit["finite_ic_days"] = audit.pop("n_days")
            audit.update(
                {
                    "fold": number,
                    "train_start": str(train.date.min().date()),
                    "train_end": str(train.date.max().date()),
                    "train_label_end_max": str(train.label_end.max().date()),
                    "train_rows": len(train),
                    "train_days": int(train.date.nunique()),
                    "test_start": str(test.date.min().date()),
                    "test_end": str(test.date.max().date()),
                    "test_label_end_max": str(valid.label_end.max().date()),
                    "test_scoring_rows": len(test),
                    "test_days": int(test.date.nunique()),
                    "test_paired_rows": int((np.isfinite(prediction) & np.isfinite(test[self.label_column])).sum()),
                    "early_stopping_sample": "external_test" if "lgbm" in fitted else None,
                    "selected_rounds": cv.rounds[-1] if cv.rounds else None,
                    "executed_rounds": fitted["lgbm"].model.training_iterations if cv.rounds else None,
                    "round_cap_hit": fitted["lgbm"].model.training_iterations >= int(self._config.lgbm.num_boost_round) if cv.rounds else False,
                }
            )
            if windows is not None:
                audit.update({"configured_test_start": str(windows[number - 1][0]), "configured_test_end": str(windows[number - 1][1])})
            cv.folds.append(audit)
            for name, score in member_scores.items():
                metrics = daily_ic(test, score, self.label_column, horizon=horizon)
                metrics["finite_ic_days"] = metrics.pop("n_days")
                metrics["test_paired_rows"] = int((np.isfinite(score) & np.isfinite(test[self.label_column])).sum())
                cv.member_folds.setdefault(name, []).append(metrics)
                self._log.info("h%s fold%s [%s] mean_IC=%+.4f IC_IR=%+.2f", horizon, number, name, metrics["mean_ic"], metrics["ic_ir"])
            self._log.info("h%s fold%s [ensemble] mean_IC=%+.4f IC_IR=%+.2f", horizon, number, audit["mean_ic"], audit["ic_ir"])
            if self._run_path is not None:
                self._save_fold(horizon, number, fitted, cv)
        if len(cv.folds) != int(cfg.n_splits):
            raise ValueError("Insufficient usable walk-forward folds")
        cv.ic = {str(key): float(value) for key, value in pd.DataFrame(cv.folds)[["mean_ic", "ic_ir"]].mean().items()}
        cv.member_ic = {
            name: {str(key): float(value) for key, value in pd.DataFrame(rows)[["mean_ic", "ic_ir"]].mean().items()}
            for name, rows in cv.member_folds.items()
        }
        self._log.info("h%s equal-fold aggregate IC=%+.4f IC_IR=%+.2f", horizon, cv.ic["mean_ic"], cv.ic["ic_ir"])
        return cv

    def _save_fold(self, horizon: int, fold: int, members: dict[str, BaseModel], cv: HorizonCV) -> None:
        """Persist original members and minimal fitting/test metrics, without prediction panels."""
        assert self._run_path is not None
        directory = contained_path(self._run_path, f"folds/h{horizon}/fold{fold}")
        for family, model in members.items():
            save_member(model, member_path(directory, horizon, family))
        write_metadata(
            directory,
            {
                "run_id": self._run_id,
                "role": "fold",
                "fold": fold,
                "horizons": [horizon],
                "model_types": self.families,
                "target_type": self.target_type,
                "label_column": self.label_column,
                "fold_audit": cv.folds[-1],
                "member_ic": {family: values[-1] for family, values in cv.member_folds.items()},
                "selected_rounds": cv.rounds[-1] if cv.rounds else None,
                "member_files": member_inventory(directory, [horizon], self.families),
            },
        )
        self._run_state["folds"].setdefault(str(horizon), []).append(fold)
        write_json(self._run_path / METADATA_FILE, self._run_state)

    def _fit_final(self, horizon: int, panel: pd.DataFrame, counts: list[int]) -> dict[str, BaseModel]:
        """Fit fresh all-row members using the upward-rounded mean of the last two fold counts."""
        if "lgbm" in self.families and (len(counts) < 2 or any(n < 1 for n in counts)):
            raise ValueError("Final LightGBM fit requires at least two positive fold-selected counts")
        return self._fit_members(horizon, panel, None, (sum(counts[-2:]) + 1) // 2 if counts else None)

    def _horizon_kpis(self, cv: HorizonCV, full_history: bool) -> dict:
        """CV KPIs only the step knows, merged into the horizon's diagnostics summary."""
        train_end = cv.audit.get("feature_max", self._config.train.end_date)
        self._log.info("Train from %s to %s", self._config.train.start_date, train_end)
        return {
            "cv_mean_ic": cv.ic.get("mean_ic"),
            "cv_ic_ir": cv.ic.get("ic_ir"),
            "target_type": self.target_type,
            "label_column": self.label_column,
            "train_start": self._config.train.start_date,
            "train_end": train_end,
            "full_history": bool(full_history),
            **cv.audit,
            "members": {name: {"cv_mean_ic": m.get("mean_ic"), "cv_ic_ir": m.get("ic_ir")} for name, m in cv.member_ic.items()},
        }

    # ---- blend + persistence ------------------------------------------ #

    def _save_models(
        self,
        models: dict[int, dict[str, BaseModel]],
        schema: Schema,
        cvs: dict[int, HorizonCV],
        full_history: bool,
        train_end_effective: pd.Timestamp | None,
    ) -> None:
        """Pickle every member + `metadata.json` (the contract the strategies, app and
        `run_predict` read)."""
        assert self._model_directory is not None
        directory = self._model_directory
        meta = {
            "run_id": self._run_id,
            "role": "full_history" if full_history else "cutoff",
            "horizons": [int(h) for h in models],
            "feature_cols": list(schema.feature_cols),
            "linear_cols": schema.family_cols["linear"],
            "lgbm_cols": schema.family_cols["lgbm"],
            "rf_cols": schema.family_cols["rf"],
            "linear_cols_by_h": {int(h): v for h, v in schema.family_cols_by_h["linear"].items()},
            "lgbm_cols_by_h": {int(h): v for h, v in schema.family_cols_by_h["lgbm"].items()},
            "rf_cols_by_h": {int(h): v for h, v in schema.family_cols_by_h["rf"].items()},
            "categorical_cols": list(schema.categorical_cols),
            "categorical_cols_by_model_h": {int(h): {name: member.categoricals for name, member in members.items()} for h, members in models.items()},
            "label_column": self.label_column,
            "target_type": self.target_type,
            "model_types": list(self.families),
            "train_start": self._config.train.start_date,
            "train_end": (
                pd.Timestamp(train_end_effective).strftime("%Y-%m-%d")
                if full_history and train_end_effective is not None
                else self._config.train.end_date
            ),
            "full_history": bool(full_history),
            "training_as_of": str(self._training_as_of.date()),
            "selection_as_of": self._selection_as_of,
            "selection": {int(h): {"folds": cv.folds, "rounds": cv.rounds, "ic": cv.ic, "member_ic": cv.member_ic} for h, cv in cvs.items()},
            "fit_audit": {int(h): cv.audit for h, cv in cvs.items()},
            "train_ic_ir": {int(h): (float(cvs[h].ic["ic_ir"]) if np.isfinite(cvs[h].ic["ic_ir"]) else 0.0) for h in models},
            "library_versions": {"python": platform.python_version(), "lightgbm": version("lightgbm"), "numpy": version("numpy")},
            "member_files": member_inventory(directory, models, self.families),
            "missing_configured_columns": sorted(set(union_all_columns(self._config)) - set(schema.load_cols)),
        }
        write_metadata(directory, meta)
        self._log.info("Saved %d horizons x %d members + metadata.json to %s", len(models), len(self.families), directory)

    def run_predict(self, n_dates: int = 1) -> pd.DataFrame:
        """Score the latest `n_dates` cube dates with the saved members -> `predictions_latest`
        (predicted_at | date | ticker | horizon | model | predicts_for | pred | rank)."""
        directory = models_dir(self._context, self._config)
        meta = read_metadata(directory)
        models = load_ensemble(directory, [int(h) for h in meta["horizons"]], list(meta.get("model_types") or self.families), metadata=meta)
        feat_cols = list(meta["feature_cols"])
        cat_cols = list(meta.get("categorical_cols", []))
        train_ic = {int(k): float(v) for k, v in meta.get("train_ic_ir", {}).items()}
        predicted_at = pd.Timestamp.now().floor("s")
        store = self._context.store
        min_share = float(self._config.data_extract.prediction_fresh_share)
        shares = check_prediction_inputs(store, load_universe_tickers(self._context), pd.Timestamp(self._today), min_share)
        self._log.info("run_predict: per-key fresh share %s (minimum %.2f)", {name: round(share, 3) for name, share in shares.items()}, min_share)

        dates = sorted(pd.Timestamp(d).normalize() for d in store.distinct(Tables.cube, "date", order="desc", limit=int(n_dates)))
        if not dates:
            raise RuntimeError("cube is empty -> nothing to predict.")
        start = min(dates)
        cube_cols = set(store.columns(Tables.cube))
        missing_inputs = set(feat_cols + cat_cols) - cube_cols
        if missing_inputs:
            raise RuntimeError(f"Model inputs absent from cube: {sorted(missing_inputs)}; rebuild the cube and retrain.")
        want = [c for c in dict.fromkeys(feat_cols + cat_cols) if c in cube_cols]
        # feature-only, float64: the newest rows have no matured label, and the linear member
        # scores in float64
        df_cube = load_frame(store, Tables.cube, columns=list(dict.fromkeys(["date", "ticker"] + want)), since=start, downcast=False)
        if df_cube is None or df_cube.empty:
            raise RuntimeError(f"No cube rows on/after {start.date()}.")
        df_panel = df_cube[["date", "ticker", *dict.fromkeys(feat_cols + cat_cols)]].copy()
        df_keys = df_panel[["date", "ticker"]]

        long_rows: list[pd.DataFrame] = []
        df_ens_wide = None
        for h, members in models.items():
            scores, member_preds = ensemble_predict(members, df_panel)
            per_model = {**{name: p.to_numpy() for name, p in member_preds.items()}, PREDICTION_MODEL_ENSEMBLE: scores.to_numpy()}
            for name, raw in per_model.items():
                long_rows.append(prediction_rows(df_keys, raw, h, name, predicted_at))
            df_z = df_keys.copy()
            df_z[f"z{h}"] = per_day_zscore(scores.to_numpy(), df_keys["date"].to_numpy())
            df_ens_wide = df_z if df_ens_wide is None else df_ens_wide.merge(df_z, on=["date", "ticker"], how="outer")
        if df_ens_wide is None:
            raise RuntimeError("No horizon produced a prediction for the latest cube date(s).")

        hs = [int(h) for h in models if f"z{h}" in df_ens_wide.columns]
        w = ir_horizon_weights({h: train_ic.get(h, 0.0) for h in hs})
        blend = blend_horizons(df_ens_wide[[f"z{h}" for h in hs]].to_numpy(), np.array([w[h] for h in hs]))
        # the blend is stamped with the IR-weighted average horizon: how far ahead it is about
        blend_h = int(round(sum(w[h] * h for h in hs))) if hs else 0
        long_rows.append(prediction_rows(df_ens_wide[["date", "ticker"]], blend, blend_h, PREDICTION_MODEL_BLENDED, predicted_at))

        df_predictions = pd.concat(long_rows, ignore_index=True)
        df_predictions = df_predictions.sort_values(["date", "model", "horizon", "rank"], ascending=[True, True, True, False]).reset_index(drop=True)
        store.replace(Tables.predictions_latest, df_predictions)
        df_last = df_predictions[(df_predictions["date"] == df_predictions["date"].max()) & (df_predictions["model"] == PREDICTION_MODEL_BLENDED)]
        self._log.info(
            "run_predict: %d row(s) -> '%s' | as-of %s | horizons %s x models %s | blend weights %s",
            len(df_predictions),
            Tables.predictions_latest,
            [str(d.date()) for d in dates],
            sorted(df_predictions["horizon"].unique()),
            sorted(df_predictions["model"].unique()),
            {h: round(w[h], 3) for h in hs},
        )
        self._log.info(
            "blended (h~%d) on %s: %d names, predicts_for %s, pred range [%.3f, %.3f]",
            blend_h,
            df_predictions["date"].max().date(),
            len(df_last),
            df_last["predicts_for"].max().date() if not df_last.empty else None,
            float(df_last["pred"].min()),
            float(df_last["pred"].max()),
        )
        return df_predictions
