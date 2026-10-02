"""
step_long_short.py  (src/modelling/steps/step_long_short.py)
-------------------------------------------------------------
`StepLongShort`: the cross-sectional long/short ensemble, trained and scored through the
transformers.

`run_train(full_history=False)`
  1. resolves horizons, per-family feature lists and the cube projection from the SCHEMA only;
  2. per horizon, loads ONLY that horizon's labelled rows (union projection, `label IS NOT NULL`
     pushed into SQL, float32, train window) -- one horizon resident at a time;
  3. purged walk-forward CV (embargo >= horizon): every `model.ensemble` family is fitted on the
     first 90 % of each training fold (LightGBM early-stops on the last 10 %), the members' per-day
     z-scores are averaged, and the fold's daily IC is recorded for the ensemble and per member;
  4. fits the final members on the whole panel, scores it, and hands the panel to the `Monitor`
     (SHAP / PDP / IC curve / KPIs) before freeing it;
  5. blends the horizons by CV IC_IR (floored at 0), persists the members + `metadata.json`, the
     `predictions` / `cube_signal` tables and the CV results, then runs the label-only `Backtest`
     on each horizon's out-of-sample CV predictions and writes the run-level KPIs.
`full_history=True` is the production refit: no train-window end, metadata records the actual
last trained date.

`run_predict(n_dates=1)` loads the saved members, scores the newest cube dates for every horizon
from a FEATURE-ONLY float64 read (their forward labels have not matured, so a labelled read would
drop them), and writes `predictions_latest` in long form: each member, each horizon's ensemble, and
the IR-weighted blend across horizons.

Exposes `run_train` / `run_predict` (with `run()` = `run_train()`) rather than a single `run`:
training and prediction are separate DAG tasks on different cadences.
"""

from __future__ import annotations

import gc
import platform
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
from src.modelling.transformers.monitor import Monitor
from src.modelling.utils.artifacts import clear_members, load_ensemble, member_path, models_dir, read_metadata, write_metadata
from src.modelling.utils.cv import purged_wf_splits, temporal_valid_split
from src.modelling.utils.ensemble import blend_horizons, ensemble_predict, ir_horizon_weights, prediction_rows
from src.modelling.utils.metrics import daily_ic, per_day_zscore
from src.modelling.utils.panel import load_frame
from src.utils.step import Step

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
        if block.get("columns"):
            out += list(block.columns)
        by_h = block.get("columns_by_horizon")
        if by_h:
            for cols in by_h.values():
                if cols:
                    out += list(cols)
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

    def run(self) -> TrainResult:
        return self.run_train()

    # ================================================================== #
    # Training                                                           #
    # ================================================================== #
    def run_train(self, full_history: bool = False) -> TrainResult:
        """Train, evaluate, persist and backtest the per-horizon ensemble (see module docstring)."""
        schema = self._resolve_schema()
        self._monitor.start_run()
        if self._half_life() is not None:
            self._log.info("Time-decay sample weights enabled (half_life=%.1f years)", self._half_life())

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
            cvs[h] = self._cross_validate(h, panel)
            models[h] = self._fit_final(h, panel)
            score_frames.append(self._score(h, models[h], panel))
            self._monitor.horizon_report(h, panel, cvs[h].oos, models[h], self._horizon_kpis(cvs[h], full_history), self.label_column)
            panel = None
            gc.collect()  # hand this horizon's panel back before the next load

        if not models:
            raise RuntimeError("No horizon produced a model (empty cube / train window?).")
        train_end_effective = max(train_ends, default=None)
        self._log.info("Trained %s horizons x %s (%s)", len(models), len(self.families), self.families)

        weights = ir_horizon_weights({h: cvs[h].ic["ic_ir"] for h in models})
        self._log.info("Blend weights: %s", {h: round(w, 3) for h, w in weights.items()})
        predictions, signal, signal_date = self._blend(score_frames, list(models), weights)
        importance = self._monitor.feature_importance(models)
        self._save_models(models, schema, cvs, full_history, train_end_effective)
        self._save_outputs(predictions, cvs, signal, signal_date)

        backtests = {}
        for h, cv in cvs.items():  # last stage: label-only backtest of the OOS CV predictions
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
        load_cols, dropped = self._select_load_columns(cube_cols)
        horizons = horizons_in(cube_cols, self.target_type)
        if not horizons:
            raise FileNotFoundError(
                f"The cube carries no 'target_{self.target_type}_h*' column at all. Run StepBuildCube first "
                f"(and confirm build_cube.targets.labels includes '{self.target_type}')."
            )
        avail = {c for c in cube_cols if not is_meta_column(c) and c != self.label_column}
        family_cols = {key: [c for c in cls.configured_columns(self._config, None) if c in avail] for key, cls in _FAMILY_BLOCKS.items()}
        family_cols_by_h = {
            key: {h: [c for c in cls.configured_columns(self._config, h) if c in avail] for h in horizons} for key, cls in _FAMILY_BLOCKS.items()
        }
        categorical_cols = [c for c in LightGBMModel.configured_categoricals(self._config) if c in avail]
        allcols = set().union(*family_cols.values())
        for by_h in family_cols_by_h.values():
            for cols in by_h.values():
                allcols |= set(cols)
        feature_cols = sorted(allcols)
        if not family_cols["linear"] and not family_cols["lgbm"]:
            raise ValueError("No configured features present in the cube; check linear_modelling.yml / lgbm_modelling.yml `columns`.")
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
        requested = union_all_columns(self._config) + LightGBMModel.configured_categoricals(self._config)
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
        raw = load_frame(store, Tables.cube, columns=self._load_cols_for(schema, tcol), where={tcol: store.NOT_NULL})
        if raw is None:
            return None
        panel = panel_from_cube(raw, horizon=horizon, label_name=self.label_column, feature_cols=schema.panel_cols, target_type=self.target_type)
        raw = None
        tr = self._config.get("train", None)
        if tr:
            start_date = tr.start_date
            end_date = None if full_history else tr.end_date
            if start_date:
                panel = panel.loc[panel["date"] >= start_date]
            if end_date:
                panel = panel.loc[panel["date"] <= end_date]
        return panel if not panel.empty else None

    def _fit_members(self, horizon: int, train: pd.DataFrame, valid: pd.DataFrame | None) -> dict[str, BaseModel]:
        """Fresh members of every family, in `model.ensemble` order."""
        return {fam: cls(self._context, self._config, fam, horizon).fit(train, valid) for fam, cls in self._classes.items()}

    def _cross_validate(self, horizon: int, panel: pd.DataFrame) -> HorizonCV:
        """Purged walk-forward CV: per-fold ensemble + member daily IC, and the OOS predictions."""
        cfg = self._config.model
        embargo = cfg.cv.embargo or horizon  # embargo must be >= horizon
        cv = HorizonCV()
        oos_frames, member_frames = [], []
        for train_days, test_days in purged_wf_splits(panel["date"], cfg.cv.n_splits, embargo):
            train = panel[panel["date"].isin(train_days)]
            test = panel[panel["date"].isin(test_days)]
            if train.empty or test.empty:
                continue
            sub_tr, sub_val = temporal_valid_split(train)
            preds, members = ensemble_predict(self._fit_members(horizon, sub_tr, sub_val), test)
            # IC IR annualised by the horizon (overlapping labels), else inflated ~sqrt(horizon)
            cv.folds.append(daily_ic(test, preds, self.label_column, horizon=horizon))
            for name, mpred in members.items():
                cv.member_folds.setdefault(name, []).append(daily_ic(test, mpred, self.label_column, horizon=horizon))
            keys = {"date": test["date"].to_numpy(), "ticker": test["ticker"].to_numpy()}
            oos_frames.append(pd.DataFrame({**keys, "pred": preds.to_numpy(), self.label_column: test[self.label_column].to_numpy()}))
            member_frames.append(pd.DataFrame({**keys, **{name: z.to_numpy() for name, z in members.items()}}))
        if oos_frames:
            cv.oos = pd.concat(oos_frames, ignore_index=True)
            cv.oos_members = pd.concat(member_frames, ignore_index=True)
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

    def _fit_final(self, horizon: int, panel: pd.DataFrame) -> dict[str, BaseModel]:
        sub_tr, sub_val = temporal_valid_split(panel, train_frac=0.9)
        return self._fit_members(horizon, sub_tr, sub_val)

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
        train_end = (datetime.today() - timedelta(days=30 + 30)).strftime("%Y-%m-%d") if full_history else self._config.train.end_date
        self._log.info("Train from %s to %s", self._config.train.start_date, train_end)
        return {
            "cv_mean_ic": cv.ic.get("mean_ic"),
            "cv_ic_ir": cv.ic.get("ic_ir"),
            "target_type": self.target_type,
            "label_column": self.label_column,
            "train_start": self._config.train.start_date,
            "train_end": train_end,
            "full_history": bool(full_history),
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
                model.save(member_path(directory, h, fam))
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
            "train_ic_ir": {int(h): (float(cvs[h].ic["ic_ir"]) if np.isfinite(cvs[h].ic["ic_ir"]) else 0.0) for h in models},
            "library_versions": {"python": platform.python_version(), "lightgbm": version("lightgbm"), "numpy": version("numpy")},
        }
        write_metadata(directory, meta)
        self._log.info("Saved %d horizons x %d members + metadata.json to %s", len(models), len(self.families), directory)

    def _save_outputs(self, predictions: pd.DataFrame, cvs: dict[int, HorizonCV], signal: pd.Series, signal_date: pd.Timestamp) -> None:
        out = self._cfg.output
        self._context.store.replace(Tables.predictions, predictions)  # full rebuild each run
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

    def _half_life(self) -> float | None:
        wd = self._config.model.get("weight_decay")
        return float(wd.half_life_years) if (wd and wd.get("enabled", False)) else None

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

        dates = sorted(pd.Timestamp(d).normalize() for d in store.distinct(Tables.cube, "date", order="desc", limit=int(n_dates)))
        if not dates:
            raise RuntimeError("cube is empty -> nothing to predict.")
        start = min(dates)
        cube_cols = set(store.columns(Tables.cube))
        want = [c for c in dict.fromkeys(feat_cols + cat_cols) if c in cube_cols]
        # feature-only, float64: the newest rows have no matured label, and the linear member
        # scores in float64
        cube = load_frame(store, Tables.cube, columns=list(dict.fromkeys(["date", "ticker"] + want)), since=start, downcast=False)
        if cube is None or cube.empty:
            raise RuntimeError(f"No cube rows on/after {start.date()}.")
        present = [c for c in (feat_cols + cat_cols) if c in cube.columns]
        missing = [c for c in (feat_cols + cat_cols) if c not in cube.columns]
        panel = cube[["date", "ticker"] + present].copy()
        if missing:  # an absent model feature scores as NaN
            panel = pd.concat([panel, pd.DataFrame(np.nan, index=panel.index, columns=missing)], axis=1)
        keys = panel[["date", "ticker"]]

        long_rows: list[pd.DataFrame] = []
        ens_wide = None
        for h, members in models.items():
            scores, member_preds = ensemble_predict(members, panel)
            per_model = {**{name: p.to_numpy() for name, p in member_preds.items()}, PREDICTION_MODEL_ENSEMBLE: scores.to_numpy()}
            for name, raw in per_model.items():
                long_rows.append(prediction_rows(keys, raw, h, name, predicted_at))
            ez = keys.copy()
            ez[f"z{h}"] = per_day_zscore(scores.to_numpy(), keys["date"].to_numpy())
            ens_wide = ez if ens_wide is None else ens_wide.merge(ez, on=["date", "ticker"], how="outer")
        if ens_wide is None:
            raise RuntimeError("No horizon produced a prediction for the latest cube date(s).")

        hs = [int(h) for h in models if f"z{h}" in ens_wide.columns]
        irs = {h: max(0.0, train_ic.get(h, 0.0)) for h in hs}
        tot = sum(irs.values())
        w = {h: (irs[h] / tot if tot > 0 else 1.0 / len(hs)) for h in hs}
        blend = blend_horizons(ens_wide[[f"z{h}" for h in hs]].to_numpy(), np.array([w[h] for h in hs]))
        # the blend is stamped with the IR-weighted average horizon: how far ahead it is about
        blend_h = int(round(sum(w[h] * h for h in hs))) if hs else 0
        long_rows.append(prediction_rows(ens_wide[["date", "ticker"]], blend, blend_h, PREDICTION_MODEL_BLENDED, predicted_at))

        out = pd.concat(long_rows, ignore_index=True)
        out = out.sort_values(["date", "model", "horizon", "rank"], ascending=[True, True, True, False]).reset_index(drop=True)
        store.replace(Tables.predictions_latest, out)
        last = out[(out["date"] == out["date"].max()) & (out["model"] == PREDICTION_MODEL_BLENDED)]
        self._log.info(
            "run_predict: %d row(s) -> '%s' | as-of %s | horizons %s x models %s | blend weights %s",
            len(out),
            Tables.predictions_latest,
            [str(d.date()) for d in dates],
            sorted(out["horizon"].unique()),
            sorted(out["model"].unique()),
            {h: round(w[h], 3) for h in hs},
        )
        self._log.info(
            "blended (h~%d) on %s: %d names, predicts_for %s, pred range [%.3f, %.3f]",
            blend_h,
            out["date"].max().date(),
            len(last),
            last["predicts_for"].max().date() if not last.empty else None,
            float(last["pred"].min()),
            float(last["pred"].max()),
        )
        return out
