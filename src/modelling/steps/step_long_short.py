"""Train cutoff ensembles with purged fold-local early stopping, then refit from saved counts.

Production refits consume all matured labels without repeating CV. Latest prediction uses
a separate feature-only load so unlabelled current rows remain scoreable.
"""

from __future__ import annotations

import gc
import platform
from dataclasses import dataclass, field
from importlib.metadata import version

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import PREDICTION_MODEL_BLENDED, PREDICTION_MODEL_ENSEMBLE
from src.context import Context
from src.data_aggregate.utils.assemble.cube import horizons_in, is_meta_column, panel_from_cube, target_column
from src.data_store.schema import Tables
from src.modelling.transformers import BaseModel, LightGBMModel, LinearRegression, RandomForestModel, model_class
from src.modelling.transformers.backtest import Backtest
from src.modelling.transformers.base import half_life_years
from src.modelling.transformers.monitor import Monitor
from src.modelling.utils.artifacts import (
    clear_members,
    load_ensemble,
    member_path,
    models_dir,
    read_metadata,
    save_member,
    selection_signature,
    write_metadata,
)
from src.modelling.utils.cv import label_end_dates, purged_wf_splits, temporal_valid_split
from src.modelling.utils.ensemble import blend_horizons, ensemble_predict, ir_horizon_weights, prediction_rows
from src.modelling.utils.metrics import daily_ic, per_day_zscore
from src.modelling.utils.panel import load_frame
from src.utils.freshness import check_prediction_inputs
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
    folds: list[dict] = field(default_factory=list)  # ensemble daily_ic per fold
    member_folds: dict[str, list[dict]] = field(default_factory=dict)
    oos: pd.DataFrame | None = None  # date, ticker, pred (ensemble), label -- every fold's test window
    oos_members: pd.DataFrame | None = None  # date, ticker, <member z> ...
    ic: dict[str, float] = field(default_factory=dict)  # mean over folds: mean_ic, ic_ir
    member_ic: dict[str, dict[str, float]] = field(default_factory=dict)
    rounds: list[int] = field(default_factory=list)
    audit: dict = field(default_factory=dict)
    diagnostic_models: dict[str, BaseModel] | None = None
    diagnostic_panel: pd.DataFrame | None = None
    fold_models: list[dict[str, BaseModel]] = field(default_factory=list)


@dataclass
class TrainResult:
    models: dict[int, dict[str, BaseModel]]
    cv: dict[int, HorizonCV]
    weights: dict[int, float]
    predictions: pd.DataFrame
    signal: pd.Series
    signal_date: pd.Timestamp
    train_end_effective: pd.Timestamp | None
    importance: pd.Series | None
    backtests: dict[int, dict[str, float]]
    summaries: dict[int, dict]


def union_all_columns(config: DictConfig) -> list[str]:
    """Every column any family might use at ANY horizon: each block's `columns` plus all of its
    `columns_by_horizon` lists, so a horizon-only feature is still loaded from the cube."""
    out: list[str] = []
    for key in _CONFIG_KEYS:
        block = config.get(key)
        if not block:
            continue
        for kind in ("columns", "categoricals"):
            out += list(block.get(kind) or [])
            for cols in (block.get(f"{kind}_by_horizon") or {}).values():
                out += list(cols) if cols else []
    return list(dict.fromkeys(out))


class StepLongShort(Step):
    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube
        self.label_column = str(config.model.label_column)
        self.target_type = str(config.model.get("target_type", "rank"))
        ensemble = config.model.get("ensemble")
        if not ensemble:
            raise ValueError("model.ensemble must list the families to train, e.g. [elasticnet, lgbm, random_forest]")
        self.families = [str(f) for f in ensemble]
        self._classes = {fam: model_class(fam) for fam in self.families}  # validates every name
        self._monitor = Monitor(context, config)
        self._backtest = Backtest(config)
        self._eligibility: dict[int, dict[str, int]] = {}

    def run(self) -> TrainResult:
        return self.run_train()

    # ================================================================== #
    # Training                                                           #
    # ================================================================== #
    def _prepare_training(self, full_history: bool = False) -> tuple[Schema, dict | None]:
        """Resolve inputs and the observation cutoff without fitting or writing artifacts."""
        schema = self._resolve_schema()
        store = self._context.store
        calendar = pd.DatetimeIndex(store.distinct(Tables.cube_part_prices, "date")).normalize().sort_values().drop_duplicates()
        cube_dates = store.distinct(Tables.cube, "date", order="desc", limit=1)
        if calendar.empty or not cube_dates:
            raise ValueError("Training requires the complete price-part calendar and a populated cube")
        latest = min(calendar[-1], pd.Timestamp(cube_dates[0]))
        self._calendar = calendar[calendar <= latest]
        self._training_as_of = latest
        if not full_history and self._config.train.end_date:
            cutoff = pd.Timestamp(self._config.train.end_date)
            sessions = calendar[calendar <= cutoff]
            if sessions.empty:
                raise ValueError("Training cutoff precedes the price calendar")
            endpoint = label_end_dates(calendar, max(schema.horizons)).get(sessions[-1])
            if pd.notna(endpoint):
                self._training_as_of = min(latest, endpoint)
        explicit_cutoff = self._config.train.get("outcome_cutoff")
        if explicit_cutoff is not None and not full_history:
            self._training_as_of = min(latest, pd.Timestamp(explicit_cutoff))
        signature = selection_signature(self._config, schema.feature_cols + schema.categorical_cols, schema.horizons)
        prior = read_metadata(models_dir(self._context, self._config)) if full_history else None
        if prior is not None and not isinstance(prior.get("selection"), dict):
            raise ValueError("Saved model has no fold selection; run modelling train before full-train")
        if prior is not None and (prior.get("selection_signature") != signature or pd.Timestamp(prior["selection_as_of"]) > self._training_as_of):
            raise ValueError("Saved fold selection is incompatible or not yet observable; run modelling train first")
        self._selection_as_of = prior["selection_as_of"] if prior else str(self._training_as_of.date())
        return schema, prior

    def run_train(self, full_history: bool = False) -> TrainResult:
        """Train, evaluate, persist and backtest the per-horizon ensemble (see module docstring)."""
        schema, prior = self._prepare_training(full_history)
        self._monitor.start_run()
        half_life = half_life_years(self._config)
        if half_life is not None:
            self._log.info("Time-decay sample weights enabled (half_life=%.1f years)", half_life)

        models: dict[int, dict[str, BaseModel]] = {}
        cvs: dict[int, HorizonCV] = {}
        score_frames: list[pd.DataFrame] = []
        train_ends: list[pd.Timestamp] = []
        for h in schema.horizons:
            panel = self._load_horizon(h, schema, full_history)
            if panel is None or panel.empty:
                self._log.warning("horizon %s: no labelled rows in the cube or the train window -> skipped", h)
                continue
            self._log.info("horizon %s: %s rows, %s tickers, %s days", h, len(panel), panel["ticker"].nunique(), panel["date"].nunique())
            train_ends.append(panel["date"].max())
            if prior is None:
                cvs[h] = self._cross_validate(h, panel)
            else:
                selected = prior["selection"][str(h)]
                cvs[h] = HorizonCV(folds=selected["folds"], rounds=selected["rounds"], ic=selected["ic"], member_ic=selected["member_ic"])
            if "lgbm" in self.families and len(cvs[h].rounds) != int(self._config.model.cv.n_splits):
                raise ValueError(f"h{h}: require all configured fold iteration counts before final fitting")
            models[h] = self._fit_final(h, panel, cvs[h].rounds)
            cvs[h].audit.update(
                {
                    **self._eligibility[h],
                    "training_as_of": str(self._training_as_of.date()),
                    "fitted_rows": len(panel),
                    "feature_min": str(panel["date"].min().date()),
                    "feature_max": str(panel["date"].max().date()),
                    "label_end_max": str(panel["label_end"].max().date()),
                    "requested_rounds": int(np.median(cvs[h].rounds)) if cvs[h].rounds else None,
                    "achieved_rounds": models[h]["lgbm"].model.current_iteration() if "lgbm" in models[h] else None,
                }
            )
            score_frames.append(self._score(h, models[h], panel))
            if not full_history:
                df_diagnostic = cvs[h].diagnostic_panel
                if df_diagnostic is None:
                    df_diagnostic = panel
                self._monitor.horizon_report(
                    h,
                    df_diagnostic,
                    cvs[h].oos,
                    cvs[h].diagnostic_models or models[h],
                    {**self._horizon_kpis(cvs[h], full_history), "diagnostic_population": "last_outer_test"},
                    self.label_column,
                )
            cvs[h].diagnostic_models = None
            cvs[h].diagnostic_panel = None
            panel = None
            gc.collect()  # hand this horizon's panel back before the next load

        if not models:
            raise RuntimeError("No horizon produced a model (empty cube / train window?).")
        if not full_history and self._config.get("portfolio", {}).get("start"):
            last_label = max(pd.Timestamp(cv.audit["label_end_max"]) for cv in cvs.values())
            if pd.Timestamp(self._config.portfolio.start) <= last_label:
                raise ValueError(f"portfolio.start must be strictly after training label end {last_label.date()}")
        train_end_effective = max(train_ends, default=None)
        self._log.info("Trained %s horizons x %s (%s)", len(models), len(self.families), self.families)

        weights = ir_horizon_weights({h: cvs[h].ic["ic_ir"] for h in models})
        self._log.info("Blend weights: %s", {h: round(w, 3) for h, w in weights.items()})
        predictions, signal, signal_date = self._blend(score_frames, list(models), weights)
        importance = self._monitor.feature_importance(models)
        self._save_models(models, schema, cvs, full_history, train_end_effective)
        self._save_outputs(predictions, cvs, signal, signal_date)

        backtests = {}
        for h, cv in cvs.items() if self._config.model.get("backtest", {}).get("enabled", True) and not full_history else []:
            if cv.oos is None:
                continue
            result = self._backtest.run(cv.oos, label_col=self.label_column, horizon=h)
            self._monitor.backtest_report(h, result)
            backtests[h] = result.summary
        self._monitor.run_report(weights, self.families, self.target_type)
        return TrainResult(
            models, cvs, weights, predictions, signal, signal_date, train_end_effective, importance, backtests, dict(self._monitor.summaries)
        )

    # ---- schema (no data read) ---------------------------------------- #
    def _resolve_schema(self) -> Schema:
        cube_cols = set(self._context.store.columns(Tables.cube))
        if {"f_fwd_eps_yield", "f_forward_earnings_yield", "f_eps_expectation_growth"} & cube_cols:
            raise RuntimeError("Obsolete fundamentals cube; rebuild full fundamentals and assemble the cube before training.")
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
        for by_h in family_cols_by_h.values():
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
            where={tcol: store.NOT_NULL},
            since=start_date,
            until=end_date,
            stream=True,
            feature_cols=schema.panel_cols,
            categorical_features=schema.categorical_cols,
        )
        if raw is None:
            return None
        panel = panel_from_cube(raw, horizon=horizon, label_name=self.label_column, feature_cols=schema.panel_cols, target_type=self.target_type)
        raw = None
        panel["label_end"] = panel["date"].map(ends)
        unknown = int(panel["label_end"].isna().sum())
        immature = int((panel["label_end"] > self._training_as_of).sum())
        self._eligibility[horizon] = {"source_labelled_rows": len(panel), "unknown_label_end_rows": unknown, "immature_rows": immature}
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

    def _cross_validate(self, horizon: int, panel: pd.DataFrame, *, retain_models: bool = False) -> HorizonCV:
        """Purged walk-forward CV: per-fold ensemble + member daily IC, and the OOS predictions."""
        cfg = self._config.model
        retain_models = retain_models or bool(cfg.cv.get("retain_models", False))
        embargo = cfg.cv.embargo or horizon  # embargo must be >= horizon
        windows = cfg.cv.get("test_windows")
        if windows is not None and int(embargo) < horizon:
            raise ValueError("Explicit assessment embargo must be at least the horizon")
        cv = HorizonCV()
        oos_frames, member_frames = [], []
        splits = purged_wf_splits(
            panel["date"], cfg.cv.n_splits, embargo, test_windows=windows, calendar=self._calendar if windows is not None else None
        )
        for fold_no, (train_days, test_days) in enumerate(splits, start=1):
            train = panel[panel["date"].isin(train_days)]
            test = panel[panel["date"].isin(test_days)]
            if train.empty or test.empty:
                continue
            if "label_end" in train:
                train = train.loc[train["label_end"] < test["date"].min()]
            sub_tr, sub_val = temporal_valid_split(train) if LightGBMModel in self._classes.values() else (train, train.iloc[:0])
            fitted: dict[str, BaseModel] = {}
            for fam, cls in self._classes.items():
                model = cls(self._context, self._config, fam, horizon)
                fitted[fam] = model.fit(sub_tr, sub_val) if cls is LightGBMModel else model.fit(train)
            if "lgbm" in fitted:
                booster = fitted["lgbm"].model
                count = int(booster.best_iteration or booster.current_iteration())
                cv.rounds.append(count)
                if booster.training_iterations >= int(self._config.lgbm.num_boost_round):
                    self._log.warning("h%s: selected rounds hit configured cap (%s)", horizon, count)
            preds, members = ensemble_predict(fitted, test)
            # IC IR annualised by the horizon (overlapping labels), else inflated ~sqrt(horizon)
            cv.folds.append(daily_ic(test, preds, self.label_column, horizon=horizon))
            cv.folds[-1].update(
                {
                    "train_rows": len(sub_tr),
                    "validation_rows": len(sub_val),
                    "outer_rows": len(train),
                    "test_start": str(test["date"].min().date()),
                    "test_end": str(test["date"].max().date()),
                    "inner_train_end": str(sub_tr["date"].max().date()),
                    "validation_start": str(sub_val["date"].min().date()) if not sub_val.empty else None,
                    "inner_train_label_end_max": str(sub_tr["label_end"].max().date()),
                    "validation_end": str(sub_val["date"].max().date()) if not sub_val.empty else None,
                    "validation_label_end_max": str(sub_val["label_end"].max().date()) if not sub_val.empty else None,
                    "outer_train_label_end_max": str(train["label_end"].max().date()),
                    "selected_rounds": cv.rounds[-1] if cv.rounds else None,
                    "executed_rounds": fitted["lgbm"].model.training_iterations if cv.rounds else None,
                    "round_cap_hit": fitted["lgbm"].model.training_iterations >= int(self._config.lgbm.num_boost_round) if cv.rounds else False,
                }
            )
            for name, mpred in members.items():
                cv.member_folds.setdefault(name, []).append(daily_ic(test, mpred, self.label_column, horizon=horizon))
            keys = {"date": test["date"].to_numpy(), "ticker": test["ticker"].to_numpy()}
            if retain_models:
                cv.fold_models.append(fitted)
                keys["fold"] = np.full(len(test), fold_no)
            oos_frames.append(pd.DataFrame({**keys, "pred": preds.to_numpy(), self.label_column: test[self.label_column].to_numpy()}))
            member_frames.append(pd.DataFrame({**keys, **{name: z.to_numpy() for name, z in members.items()}}))
            if self._monitor.active and fold_no == int(cfg.cv.n_splits):
                sample = int(cfg.get("diagnostics", {}).get("shap_sample", 2000))
                cv.diagnostic_models = fitted
                cv.diagnostic_panel = test.sample(n=min(sample or len(test), len(test)), random_state=int(self._config.get("seed", 42))).copy()
        if oos_frames:
            cv.oos = pd.concat(oos_frames, ignore_index=True)
            cv.oos_members = pd.concat(member_frames, ignore_index=True)
        if not cv.folds or len(cv.folds) != int(cfg.cv.n_splits):
            raise ValueError("Insufficient usable walk-forward folds")
        cv.ic = {
            "mean_ic": float(np.nanmean([r["mean_ic"] for r in cv.folds])) if cv.folds else np.nan,
            "ic_ir": float(np.nanmean([r["ic_ir"] for r in cv.folds])) if cv.folds else np.nan,
        }
        self._log.info("horizon %s: [ENSEMBLE] CV mean_IC=%+.4f  IC_IR=%+.2f", horizon, cv.ic["mean_ic"], cv.ic["ic_ir"])
        for name, folds in cv.member_folds.items():
            cv.member_ic[name] = {
                "mean_ic": float(np.nanmean([r["mean_ic"] for r in folds])) if folds else np.nan,
                "ic_ir": float(np.nanmean([r["ic_ir"] for r in folds])) if folds else np.nan,
            }
            self._log.info(
                "horizon %s:   [%-10s] CV mean_IC=%+.4f  IC_IR=%+.2f", horizon, name, cv.member_ic[name]["mean_ic"], cv.member_ic[name]["ic_ir"]
            )
        return cv

    def _fit_final(self, horizon: int, panel: pd.DataFrame, counts: list[int]) -> dict[str, BaseModel]:
        """Fresh all-row members, with fold-selected boosting rounds and no validation."""
        if "lgbm" in self.families and (not counts or any(n < 1 for n in counts)):
            raise ValueError("Final LightGBM fit requires positive fold-selected counts")
        return self._fit_members(horizon, panel, None, int(np.median(counts)) if counts else None)

    def _score(self, horizon: int, models: dict[str, BaseModel], panel: pd.DataFrame) -> pd.DataFrame:
        """Small blend input (date, ticker, z_<h>, pred_<member>_h<h>), so the wide panel can be
        freed right after."""
        scores, members = ensemble_predict(models, panel)
        df = panel[["date", "ticker"]].copy()
        df["score"] = scores.to_numpy()
        df["z"] = per_day_zscore(df["score"].to_numpy(), df["date"].to_numpy())  # horizons comparable
        member_cols = []
        for name, mz in members.items():
            col = f"pred_{name}_h{horizon}"
            df[col] = mz.to_numpy()
            member_cols.append(col)
        return df[["date", "ticker", "z"] + member_cols].rename(columns={"z": f"z_{horizon}"})

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
    def _blend(
        self, score_frames: list[pd.DataFrame], horizons: list[int], weights: dict[int, float]
    ) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp]:
        """IR-weighted blend of the per-horizon z-scores -> `combined`, its per-day percentile
        `signal`, and the latest date's signal."""
        blended = None
        for df in score_frames:
            blended = df if blended is None else blended.merge(df, on=["date", "ticker"], how="outer")
        if blended is None:
            raise RuntimeError("No horizon score frame was produced")
        zcols = [f"z_{h}" for h in horizons]
        blended["combined"] = blend_horizons(blended[zcols].to_numpy(), np.array([weights[h] for h in horizons]))
        blended["signal"] = blended.groupby("date")["combined"].rank(pct=True)
        last_date = pd.Timestamp(blended["date"].max())
        latest = blended[blended["date"] == last_date].sort_values("signal", ascending=False)
        signal = latest.set_index("ticker")["signal"]
        self._log.info("Blended signal for %s (%s names)", last_date.date(), signal.notna().sum())
        return blended, signal, last_date

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
        directory = models_dir(self._context, self._config)
        stale = clear_members(directory)
        if stale:
            self._log.info("Removed %d member file(s) of the previous run from %s", len(stale), directory)
        for h, members in models.items():
            for fam, model in members.items():
                save_member(model, member_path(directory, h, fam))
        meta = {
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
            "selection_signature": selection_signature(self._config, schema.feature_cols + schema.categorical_cols, schema.horizons),
            "selection": {int(h): {"folds": cv.folds, "rounds": cv.rounds, "ic": cv.ic, "member_ic": cv.member_ic} for h, cv in cvs.items()},
            "fit_audit": {int(h): cv.audit for h, cv in cvs.items()},
            "train_ic_ir": {int(h): (float(cvs[h].ic["ic_ir"]) if np.isfinite(cvs[h].ic["ic_ir"]) else 0.0) for h in models},
            "library_versions": {"python": platform.python_version(), "lightgbm": version("lightgbm"), "numpy": version("numpy")},
        }
        write_metadata(directory, meta)
        self._log.info("Saved %d horizons x %d members + metadata.json to %s", len(models), len(self.families), directory)

    def _save_outputs(self, predictions: pd.DataFrame, cvs: dict[int, HorizonCV], signal: pd.Series, signal_date: pd.Timestamp) -> None:
        out = self._cfg.output
        if out.get("save_predictions", True):
            self._context.store.replace(Tables.predictions, predictions)
            self._log.info("Saved predictions to DB table '%s'", Tables.predictions)
        if not self._context.save:
            return
        if out.save_cv_results:
            rows = [{"horizon": h, "fold": i, **r} for h, cv in cvs.items() for i, r in enumerate(cv.folds)]
            pd.DataFrame(rows).to_parquet(self._context.paths["CUBE_CV_RESULTS_PATH"], index=False)
        if out.save_signal:
            sig = signal.rename("signal").reset_index()
            sig.insert(0, "date", signal_date)
            self._context.store.replace(Tables.cube_signal, sig)
            self._log.info("Saved blended signal to DB table '%s'", Tables.cube_signal)

    # ================================================================== #
    # Production prediction                                              #
    # ================================================================== #
    def run_predict(self, n_dates: int = 1) -> pd.DataFrame:
        """Score the latest `n_dates` cube dates with the saved members -> `predictions_latest`
        (predicted_at | date | ticker | horizon | model | predicts_for | pred | rank)."""
        directory = models_dir(self._context, self._config)
        meta = read_metadata(directory)
        models = load_ensemble(directory, [int(h) for h in meta["horizons"]], list(meta.get("model_types") or self.families))
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
