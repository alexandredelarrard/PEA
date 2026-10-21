"""Assess saved original folds and a fixed cutoff ensemble; fitting belongs to StepLongShort."""

from __future__ import annotations

import gc
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.assemble.cube import target_column
from src.data_store.schema import Tables
from src.modelling.transformers import BaseModel, LinearRegression
from src.modelling.transformers.assessment_diagnostics import explain_linear, explain_tree
from src.modelling.transformers.assessment_report import render_assessment
from src.modelling.utils.artifacts import (
    METADATA_FILE,
    file_identity,
    load_ensemble,
    read_metadata,
)
from src.modelling.utils.assessment import assessment_workspace, balanced_sample_positions, publish_assessment, write_json
from src.modelling.utils.assessment_metrics import temporal_diagnostics
from src.modelling.utils.cv import label_end_dates
from src.modelling.utils.ensemble import ensemble_predict
from src.modelling.utils.panel import load_frame
from src.utils.modelling_runs import select_run
from src.utils.step import Step


class _ReadOnlyStore:
    def __init__(self, store: Any) -> None:
        self.source = store

    def __getattr__(self, name: str) -> Any:
        if name not in {"columns", "distinct", "NOT_NULL", "load", "iter_load"}:
            raise RuntimeError(f"Assessment application operation refused: {name}")
        return getattr(self.source, name)


class StepAssessLongShort(Step):
    def __init__(self, context: Context, config: DictConfig, run_id: str | None = None) -> None:
        super().__init__(context=context, config=config)
        self._options = config.model.assessment
        self._requested_run_id = run_id
        self._store = _ReadOnlyStore(context.store)

    def run(self) -> Path:
        """Recompute every required analysis from saved models and current data before publication."""
        self._output, training = select_run(self._context, self._config, self._requested_run_id)
        completed = datetime.fromisoformat(training["training_completed_at_utc"])
        now = datetime.now(UTC)
        self._log.info("Assessment selected run_id=%s trained_at_utc=%s", training["run_id"], completed.isoformat())
        if now - completed > timedelta(days=1):
            self._log.warning("Selected model is more than one day old: run_id=%s trained_at_utc=%s", training["run_id"], completed.isoformat())
        cutoff = read_metadata(self._output / "cutoff")
        self._check_training(training, cutoff)
        current = pd.DatetimeIndex(self._store.distinct(Tables.cube_part_prices, "date")).normalize().sort_values().drop_duplicates()
        cube_dates = self._store.distinct(Tables.cube, "date", order="desc", limit=1)
        if current.empty or not cube_dates:
            raise ValueError("Assessment requires the price calendar and a populated cube")
        calendar = current[current <= pd.Timestamp(cube_dates[0])]
        if calendar.empty:
            raise ValueError("Assessment cube and price calendar do not overlap")
        diagnostics = self._config.model.diagnostics
        state = {
            "run_id": training["run_id"],
            "training_completed_at_utc": training["training_completed_at_utc"],
            "assessment_started_at_utc": now.isoformat(),
            "observation_frontier": str(calendar[-1].date()),
            "diagnostics": {key: diagnostics[key] for key in ("shap_sample", "dependence_sample", "pdp_grid", "pdp_background", "rolling_window")},
            "evaluation_start": str(self._options.evaluation_start),
            "seed": int(self._config.seed),
            "fitted_feature_cols": training["feature_cols"],
            "horizons": {},
            "windows": {},
            "complete": False,
            "no_fitting": True,
        }
        with assessment_workspace(self._output, self._log) as workspace:
            self._analysis_path = workspace
            for horizon in training["horizons"]:
                self._assess_horizon(int(horizon), training, cutoff, calendar, state)
            state["assessment_completed_at_utc"] = datetime.now(UTC).isoformat()
            write_json(workspace / METADATA_FILE, state)
            report = render_assessment(self._output, assessment_directory=workspace, report_path=workspace / "report.html")
            state["report_file"] = file_identity(report)
            result = publish_assessment(self._output, workspace, state, self._log)
        self._log.info("Fresh saved-model assessment complete: %s", result)
        return result

    def _check_training(self, training: dict[str, Any], cutoff: dict[str, Any]) -> None:
        """Validate fitted role/schema and chronology without requiring current fitting settings."""
        if cutoff.get("run_id") != training["run_id"] or cutoff.get("full_history") is not False or cutoff.get("role") != "cutoff":
            raise ValueError("Assessment requires this run's cutoff models")
        if training["target_type"] != "zscore":
            raise ValueError("Saved-model assessment requires zscore targets")
        if cutoff["horizons"] != training["horizons"] or cutoff["model_types"] != training["model_types"]:
            raise ValueError("Cutoff horizon/family inventory differs from saved training")
        if pd.Timestamp(self._options.evaluation_start) <= pd.Timestamp(cutoff["training_as_of"]):
            raise ValueError("Post-2022 assessment must start after every cutoff training outcome")
        required = set(cutoff["feature_cols"] + cutoff["categorical_cols"])
        if missing := required - set(self._store.columns(Tables.cube)):
            raise ValueError(f"Fitted model inputs absent from current cube: {sorted(missing)}")
        self._fitted_inputs = required

    def _windows(self, horizon: int, cutoff: dict[str, Any], calendar: pd.DatetimeIndex) -> list[tuple[int, pd.Timestamp, pd.Timestamp, int]]:
        end_dates = label_end_dates(calendar, horizon)
        mature = end_dates.index[end_dates.notna() & (end_dates <= pd.Timestamp(cutoff["training_as_of"]))]
        windows = []
        for number, fold in enumerate(cutoff["selection"][str(horizon)]["folds"], 1):
            start = fold.get("configured_test_start", fold["test_start"])
            stop = fold.get("configured_test_end", fold["test_end"])
            stop = min(pd.Timestamp(stop), mature[-1], pd.Timestamp(cutoff["train_end"]))
            windows.append((number, pd.Timestamp(start), stop))
        counts = np.array([int(((calendar >= start) & (calendar <= stop)).sum()) for _, start, stop in windows])
        budget = int(self._config.model.diagnostics.shap_sample)
        if not counts.all() or budget < len(windows):
            raise ValueError("Each fold needs observed dates and a positive share of the diagnostic sample")
        quotas = np.maximum(1, np.floor(budget * counts / counts.sum()).astype(int))
        while quotas.sum() > budget:
            quotas[np.argmax(quotas)] -= 1
        for index in np.argsort(-(budget * counts / counts.sum() - quotas))[: budget - int(quotas.sum())]:
            quotas[index] += 1
        result = [(number, start, stop, int(quota)) for (number, start, stop), quota in zip(windows, quotas, strict=True)]
        evaluation_start, evaluation_end = pd.Timestamp(self._options.evaluation_start), calendar[-horizon - 1]
        if evaluation_end < evaluation_start:
            raise ValueError(f"h{horizon}: no mature fixed-cutoff evaluation")
        return [*result, (0, evaluation_start, evaluation_end, budget)]

    def _assess_horizon(
        self, horizon: int, training: dict[str, Any], cutoff: dict[str, Any], calendar: pd.DatetimeIndex, state: dict[str, Any]
    ) -> None:
        output = self._analysis_path / f"h{horizon}"
        output.mkdir(parents=True, exist_ok=True)
        frames = [
            self._score_window(horizon, fold, start, stop, budget, training, state)
            for fold, start, stop, budget in self._windows(horizon, cutoff, calendar)
        ]
        scores = pd.concat(frames, ignore_index=True)
        # All cross-sections are scored and saved before reading a single future outcome.
        target = target_column(training["target_type"], horizon)
        labels = load_frame(
            self._store,
            Tables.cube,
            columns=["date", "ticker", target],
            since=scores.date.min(),
            until=scores.date.max(),
            downcast=False,
            stream=True,
            optional=False,
            chunksize=int(self._options.read_chunksize),
        )
        if labels is None:
            raise ValueError(f"h{horizon}: missing evaluation targets")
        labels = labels.rename(columns={target: "y"})
        predictions = scores.merge(labels, on=["date", "ticker"], how="left", validate="one_to_one")
        predictions["label_end"] = predictions.date.map(label_end_dates(calendar, horizon))
        known = predictions.label_end.notna() & ((predictions.fold == 0) | (predictions.label_end <= pd.Timestamp(cutoff["training_as_of"])))
        predictions.loc[~known | ~np.isfinite(predictions.y), "y"] = np.nan
        predictions.loc[predictions.fold > 0, "y"] = predictions.loc[predictions.fold > 0, "y"].astype("float32")
        predictions.to_parquet(output / "predictions.parquet", index=False)
        models = [*training["model_types"], "ensemble"]
        metrics, daily_frames = {}, []
        for population, mask in [("validation", predictions.fold > 0), ("backtest", predictions.fold == 0)]:
            metrics[population], daily = temporal_diagnostics(
                predictions.loc[mask], models, horizon, int(self._config.model.diagnostics.rolling_window)
            )
            daily["population"], daily["horizon"] = population, horizon
            daily_frames.append(daily)
        pd.concat(daily_frames, ignore_index=True).to_parquet(output / "daily.parquet", index=False)
        write_json(output / "metrics.json", metrics)
        state["horizons"][str(horizon)] = {
            "scoring_rows": len(predictions),
            "paired_rows": int(np.isfinite(predictions.y).sum()),
            "effective_fold_test_end": str(predictions.loc[(predictions.fold > 0) & np.isfinite(predictions.y), "date"].max()),
            "metrics": metrics,
        }
        self._log.info("h%s saved fold-test/post-2022 IC and IC-IR for %s", horizon, models)

    def _score_window(
        self, horizon: int, fold: int, start: pd.Timestamp, stop: pd.Timestamp, budget: int, training: dict[str, Any], state: dict[str, Any]
    ) -> pd.DataFrame:
        directory = self._output / (f"folds/h{horizon}/fold{fold}" if fold else "cutoff")
        meta = read_metadata(directory)
        if (
            meta.get("run_id") != training["run_id"]
            or meta.get("model_types") != training["model_types"]
            or meta.get("role") != ("fold" if fold else "cutoff")
            or horizon not in meta["horizons"]
            or (fold and (meta.get("fold") != fold or meta["horizons"] != [horizon]))
        ):
            raise ValueError("Wrong run/role/horizon/fold model metadata")
        if meta.get("member_files") is None:
            raise ValueError("Missing original model member inventory")
        members = load_ensemble(directory, [horizon], training["model_types"], metadata=meta)[horizon]
        for name, member in members.items():
            if member.family != name or member.horizon != horizon or member.task != "regression":
                raise ValueError("Saved member identity/task differs from the requested assessment")
        absent = {feature for member in members.values() for feature in member.features} - self._fitted_inputs
        if absent:
            raise ValueError(f"Fitted member inputs absent from saved cutoff schema: {sorted(absent)}")
        features = list(dict.fromkeys(feature for member in members.values() for feature in member.features))
        categories = list(dict.fromkeys(feature for member in members.values() for feature in member.categoricals))
        panel = load_frame(
            self._store,
            Tables.cube,
            columns=["date", "ticker", *features],
            since=start,
            until=stop,
            downcast=bool(fold),
            stream=True,
            feature_cols=features,
            categorical_features=categories,
            chunksize=int(self._options.read_chunksize),
            optional=False,
        )
        if panel is None or panel.empty or panel.duplicated(["date", "ticker"]).any():
            raise ValueError(f"h{horizon} fold{fold}: missing or duplicate feature-only population")
        panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
        identity = {"rows": len(panel), "start": str(start), "stop": str(stop), "sample_budget": budget}
        output = self._analysis_path / f"h{horizon}/fold{fold}"
        output.mkdir(parents=True, exist_ok=True)
        score, member_scores = ensemble_predict(members, panel)
        frame = panel[["date", "ticker"]].copy()
        for name, prediction in member_scores.items():
            frame[name] = prediction.to_numpy()
        frame["ensemble"], frame["fold"] = score.to_numpy(), fold
        frame.to_parquet(output / "scores.parquet", index=False)
        sample = panel.iloc[balanced_sample_positions(panel, budget, int(self._config.seed))]
        sample.to_parquet(output / "sample.parquet", index=False)
        for name, member in members.items():
            self._explain_member(member, sample, output / name, pdp=fold == 0)
        write_json(output / METADATA_FILE, {**identity, "sample_rows": len(sample), "future_outcomes_read": False})
        state["windows"].setdefault(str(horizon), {})[str(fold)] = identity
        self._log.info("h%s fold%s: scored %s feature-only rows; explained %s rows", horizon, fold, len(panel), len(sample))
        del members, panel
        gc.collect()
        return frame

    def _explain_member(self, member: BaseModel, sample: pd.DataFrame, output: Path, *, pdp: bool) -> None:
        """Always compute explanations for this newly scored window."""
        if isinstance(member, LinearRegression):
            explain_linear(member, sample, output, self._config, self._log, pdp=pdp)
        elif member.supports_shap:
            explain_tree(member, sample, output, self._config, self._log, pdp=pdp)
        else:
            raise ValueError(f"Required explanations unavailable for {member.family}")
