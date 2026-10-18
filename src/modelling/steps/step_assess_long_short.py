"""Assess saved original folds and a fixed cutoff ensemble; fitting belongs to StepLongShort."""

from __future__ import annotations

import gc
import json
import platform
from importlib.metadata import version
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
    fitting_source_identity,
    load_ensemble,
    read_metadata,
    read_run,
    run_directory,
    selection_signature,
    verify_inventory,
)
from src.modelling.utils.assessment import balanced_sample_positions, frame_digest, write_json
from src.modelling.utils.assessment_metrics import temporal_diagnostics
from src.modelling.utils.cv import label_end_dates
from src.modelling.utils.ensemble import ensemble_predict
from src.modelling.utils.panel import load_frame
from src.utils.step import Step


class _ReadOnlyStore:
    def __init__(self, store: Any) -> None:
        self.source = store
        self.attempted_mutations: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name not in {"columns", "distinct", "NOT_NULL", "load", "iter_load"}:
            self.attempted_mutations.append(name)
            raise RuntimeError(f"Assessment application operation refused: {name}")
        return getattr(self.source, name)


class StepAssessLongShort(Step):
    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._options = config.model.assessment
        self._output = run_directory(context, config)
        self._store = _ReadOnlyStore(context.store)

    def run(self) -> Path:
        """Read this exact run, produce required analysis and publish only a complete report."""
        training = read_run(self._output)
        verify_inventory(self._output, {"cutoff/metadata.json": training["cutoff_metadata"], "training/calendar.parquet": training["calendar_file"]})
        cutoff = read_metadata(self._output / "cutoff")
        self._check_training(training, cutoff)
        identity = self._identity(training)
        state_path = self._output / "assessment/metadata.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["identity"] != identity:
                raise ValueError("Assessment configuration, model or diagnostic source identity changed")
            if not self._options.get("resume", False) and not self._options.get("report_only", False):
                raise FileExistsError("Assessment exists; use matching --resume or --report-only")
        else:
            if self._options.get("report_only", False):
                raise ValueError("Report-only requires a completed saved assessment")
            state = {"run_id": training["run_id"], "identity": identity, "horizons": {}, "windows": {}, "complete": False}
            write_json(state_path, state)
        if self._options.get("report_only", False):
            if state.get("complete") is not True:
                raise ValueError("Report-only requires every horizon's completed analysis")
            self._verify_analysis(state, training)
            return self._publish_report(state)

        calendar = pd.DatetimeIndex(pd.read_parquet(self._output / "training/calendar.parquet").date)
        current = pd.DatetimeIndex(self._store.distinct(Tables.cube_part_prices, "date")).normalize().sort_values().drop_duplicates()
        cube_dates = self._store.distinct(Tables.cube, "date", order="desc", limit=1)
        if not cube_dates:
            raise ValueError("Assessment requires a populated cube")
        current = current[current <= pd.Timestamp(cube_dates[0])]
        if not np.array_equal(calendar.to_numpy(dtype="datetime64[ns]"), current.to_numpy(dtype="datetime64[ns]")):
            raise ValueError("Observed calendar/frontier changed; start a new modelling run")
        for horizon in training["horizons"]:
            self._assess_horizon(int(horizon), training, cutoff, calendar, state)
            write_json(state_path, state)
        self._verify_analysis(state, training)
        return self._publish_report(state)

    def _publish_report(self, state: dict[str, Any]) -> Path:
        """Publish the successful reference only after the required HTML is written."""
        report = render_assessment(self._output, self._config)
        state.update({"complete": True, "report_file": file_identity(report), "no_fitting": True})
        state_path = self._output / "assessment/metadata.json"
        write_json(state_path, state)
        write_json(
            self._output.parent / "latest.json",
            {
                "run_id": state["run_id"],
                "directory": self._output.name,
                "assessment_metadata": file_identity(state_path),
                "report_file": state["report_file"],
            },
        )
        self._log.info("Saved-model assessment complete: %s", report)
        return report

    def _check_training(self, training: dict[str, Any], cutoff: dict[str, Any]) -> None:
        if cutoff.get("run_id") != training["run_id"] or cutoff.get("full_history") is not False:
            raise ValueError("Assessment requires this run's cutoff models")
        if training["source_hashes"] != fitting_source_identity():
            raise ValueError("Saved fitting/scoring implementation changed")
        versions = {"python": platform.python_version(), **{name: version(name) for name in ("lightgbm", "numpy", "pandas")}}
        if training["library_versions"] != versions:
            raise ValueError("Saved training library identity changed")
        if training["target_type"] != "zscore" or self._config.model.target_type != "zscore":
            raise ValueError("Saved-model assessment requires zscore targets")
        if list(self._config.model.horizons) != training["horizons"] or list(self._config.model.ensemble) != training["model_types"]:
            raise ValueError("Configured horizons or families differ from the saved run")
        signature = selection_signature(self._config, cutoff["feature_cols"] + cutoff["categorical_cols"], training["horizons"])
        if signature != training["selection_signature"] or cutoff["selection_signature"] != signature:
            raise ValueError("Configured fitting inputs differ from the saved run")
        if pd.Timestamp(self._options.evaluation_start) <= pd.Timestamp(cutoff["training_as_of"]):
            raise ValueError("Backtest must start after every cutoff training outcome")
        verify_inventory(self._output / "cutoff", cutoff["member_files"])
        for horizon in training["horizons"]:
            folds = training["folds"].get(str(horizon), {})
            if set(folds) != {str(i) for i in range(1, int(self._config.model.cv.n_splits) + 1)}:
                raise ValueError(f"h{horizon}: incomplete original fold inventory")
            for fold, expected in folds.items():
                directory = self._output / f"folds/h{horizon}/fold{fold}"
                verify_inventory(directory, {METADATA_FILE: expected})
                meta = read_metadata(directory)
                if (
                    meta["run_id"] != training["run_id"]
                    or meta["horizons"] != [horizon]
                    or meta["fold"] != int(fold)
                    or meta["model_types"] != training["model_types"]
                    or meta["role"] != "fold"
                ):
                    raise ValueError("Wrong-run/horizon/fold model metadata")
                verify_inventory(directory, meta["member_files"])
                verify_inventory(directory, meta["evidence_files"])

    def _identity(self, training: dict[str, Any]) -> dict[str, Any]:
        source_root = Path(__file__).resolve().parents[3]
        sources = [
            "steps/step_assess_long_short.py",
            "transformers/assessment_diagnostics.py",
            "transformers/monitor.py",
            "utils/assessment.py",
            "utils/assessment_metrics.py",
        ]
        diagnostics = self._config.model.diagnostics
        return {
            "run_id": training["run_id"],
            "cutoff_metadata": training["cutoff_metadata"],
            "folds": training["folds"],
            "calendar_file": training["calendar_file"],
            "evaluation_start": str(self._options.evaluation_start),
            "seed": int(self._config.seed),
            "diagnostics": {key: diagnostics[key] for key in ("shap_sample", "dependence_sample", "pdp_grid", "pdp_background", "rolling_window")},
            "source_hashes": {name: file_identity(source_root / "src/modelling" / name)["sha256"] for name in sources},
            "library_versions": {name: version(name) for name in ("numpy", "pandas", "lightgbm", "shap")},
        }

    def _windows(self, horizon: int, cutoff: dict[str, Any], calendar: pd.DatetimeIndex) -> list[tuple[int, pd.Timestamp, pd.Timestamp, int]]:
        end_dates = label_end_dates(calendar, horizon)
        mature = end_dates.index[end_dates.notna() & (end_dates <= pd.Timestamp(cutoff["training_as_of"]))]
        windows = []
        configured = self._config.model.cv.get("test_windows")
        for number, fold in enumerate(cutoff["selection"][str(horizon)]["folds"], 1):
            start, stop = configured[number - 1] if configured is not None else (fold["test_start"], fold["test_end"])
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
        output = self._output / f"assessment/h{horizon}"
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
        digest = frame_digest(predictions, ["y", "label_end"], downcast=False)
        previous = state["horizons"].get(str(horizon))
        if previous is not None:
            if previous["label_digest"] != digest:
                raise ValueError("Historical evaluation outcomes changed; refuse stale assessment reuse")
            verify_inventory(output, previous["files"])
            return
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
            "label_digest": digest,
            "files": {name: file_identity(output / name) for name in ("predictions.parquet", "daily.parquet", "metrics.json")},
        }
        self._log.info("h%s saved validation/backtest IC and IC-IR for %s", horizon, models)

    def _score_window(
        self, horizon: int, fold: int, start: pd.Timestamp, stop: pd.Timestamp, budget: int, training: dict[str, Any], state: dict[str, Any]
    ) -> pd.DataFrame:
        directory = self._output / (f"folds/h{horizon}/fold{fold}" if fold else "cutoff")
        meta = read_metadata(directory)
        members = load_ensemble(directory, [horizon], training["model_types"], metadata=meta)[horizon]
        for name, member in members.items():
            if member.family != name or member.horizon != horizon or member.task != "regression":
                raise ValueError("Saved member identity/task differs from the requested assessment")
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
        identity = {
            "input_digest": frame_digest(panel, features, downcast=False),
            "rows": len(panel),
            "start": str(start),
            "stop": str(stop),
            "budget": budget,
            "model_metadata": file_identity(directory / METADATA_FILE),
        }
        output = self._output / f"assessment/h{horizon}/fold{fold}"
        output.mkdir(parents=True, exist_ok=True)
        previous = state["windows"].get(str(horizon), {}).get(str(fold))
        if previous is not None:
            verify_inventory(output, {METADATA_FILE: previous})
            saved = json.loads((output / METADATA_FILE).read_text(encoding="utf-8"))
            if saved["identity"] != identity:
                raise ValueError("Prepared feature/universe inputs changed; refuse stale scoring reuse")
            verify_inventory(output, saved["files"])
            return pd.read_parquet(output / "scores.parquet")
        score, member_scores = ensemble_predict(members, panel)
        frame = panel[["date", "ticker"]].copy()
        for name, prediction in member_scores.items():
            frame[name] = prediction.to_numpy()
        frame["ensemble"], frame["fold"] = score.to_numpy(), fold
        frame.to_parquet(output / "scores.parquet", index=False)
        sample = panel.iloc[balanced_sample_positions(panel, budget, int(self._config.seed))]
        sample.to_parquet(output / "sample.parquet", index=False)
        for name, member in members.items():
            self._explain_member(member, sample, output / name, identity, pdp=fold == 0)
        files = {
            str(path.relative_to(output)).replace("\\", "/"): file_identity(path)
            for path in output.rglob("*")
            if path.is_file() and path != output / METADATA_FILE
        }
        write_json(output / METADATA_FILE, {"identity": identity, "files": files, "future_outcomes_read": False})
        state["windows"].setdefault(str(horizon), {})[str(fold)] = file_identity(output / METADATA_FILE)
        write_json(self._output / "assessment/metadata.json", state)
        self._log.info("h%s fold%s: scored %s feature-only rows; explained %s rows", horizon, fold, len(panel), len(sample))
        del members, panel
        gc.collect()
        return frame

    def _explain_member(self, member: BaseModel, sample: pd.DataFrame, output: Path, identity: dict[str, Any], *, pdp: bool) -> None:
        marker = output / "complete.json"
        if marker.exists():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved["identity"] != identity:
                raise ValueError("Saved explanation has changed model/sample inputs")
            verify_inventory(output, saved["files"])
            return
        if isinstance(member, LinearRegression):
            explain_linear(member, sample, output, self._config, self._log, pdp=pdp)
        elif member.supports_shap:
            explain_tree(member, sample, output, self._config, self._log, pdp=pdp)
        else:
            raise ValueError(f"Required explanations unavailable for {member.family}")
        files = {path.name: file_identity(path) for path in output.iterdir() if path.is_file() and path != marker}
        write_json(marker, {"identity": identity, "files": files})

    def _verify_analysis(self, state: dict[str, Any], training: dict[str, Any]) -> None:
        for horizon in training["horizons"]:
            if str(horizon) not in state["horizons"]:
                raise ValueError("Every requested horizon must complete before reporting")
            output = self._output / f"assessment/h{horizon}"
            verify_inventory(output, state["horizons"][str(horizon)]["files"])
            windows = state["windows"].get(str(horizon), {})
            if set(windows) != {"0", *training["folds"][str(horizon)]}:
                raise ValueError("Every original fold and cutoff explanation must complete")
            for fold, expected in windows.items():
                folder = output / f"fold{fold}"
                verify_inventory(folder, {METADATA_FILE: expected})
                saved = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
                verify_inventory(folder, saved["files"])
