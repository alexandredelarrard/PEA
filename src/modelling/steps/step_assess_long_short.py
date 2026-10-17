"""Run isolated fixed-parameter feature selection, then sealed h60 evaluation and reporting."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.context import Context
from src.data_store.schema import Tables
from src.modelling.steps.step_long_short import StepLongShort
from src.modelling.transformers import BaseModel, model_class
from src.modelling.transformers.assessment_diagnostics import explain_tree
from src.modelling.transformers.assessment_report import render_assessment
from src.modelling.transformers.monitor import shap_row_values, shap_score_scale
from src.modelling.utils.artifacts import load_member, save_member
from src.modelling.utils.assessment import (
    assess_candidate,
    audit_features,
    balanced_sample_positions,
    frame_digest,
    prune_pool,
    set_horizon_features,
    write_json,
)
from src.modelling.utils.assessment_metrics import formation_cohorts, forward_returns, matched_cohort_returns, temporal_diagnostics
from src.modelling.utils.cv import label_end_dates
from src.modelling.utils.ensemble import ensemble_predict
from src.modelling.utils.features import design_matrix
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
        self._output = Path(str(self._options.output_dir)).resolve()
        root = Path(context.paths["ROOT"]).resolve()
        if not self._output.is_relative_to(root / "reports" / "validate"):
            raise ValueError("Assessment output must stay under the project's reports/validate directory")
        if int(self._options.horizon) != 60 or str(config.model.target_type) != "zscore":
            raise ValueError("This assessment requires the approved h60 z-score target")

    def run(self) -> Path:
        """Keep application state isolated even if a candidate or diagnostic fails."""
        original_store, original_paths = self._context.store, self._context.paths
        store = _ReadOnlyStore(original_store)
        self._output.mkdir(parents=True, exist_ok=True)
        try:
            self._context.store = cast(Any, store)
            self._context.paths = dict(original_paths)
            self._context.paths["OUTPUT_DIR"] = self._output / "_out"
            stage = str(self._options.get("stage", "development"))
            if stage in {"all", "development"}:
                self._development()
            if stage in {"all", "evaluation"}:
                self._evaluation()
            if stage in {"all", "report"}:
                self._report()
            if stage not in {"all", "development", "evaluation", "report"}:
                raise ValueError(f"Unknown assessment stage: {stage}")
            if store.attempted_mutations:
                raise RuntimeError("Application mutation attempted during assessment")
        finally:
            self._context.store = original_store
            self._context.paths = original_paths
            write_json(self._output / "_out" / "isolation.json", {"attempted_mutations": store.attempted_mutations})
        return self._output

    def _runtime_config(self, family: str | None = None, features: list[str] | None = None, categoricals: list[str] | None = None) -> DictConfig:
        config = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(self._config, resolve=True)))
        config.model.horizons = [60]
        config.model.read_chunksize = int(self._options.read_chunksize)
        config.model.cv.test_windows = self._options.test_windows
        config.model.cv.retain_models = True
        config.model.cv.input_digest = True
        config.model.diagnostics.enabled = False
        config.model.backtest.enabled = False
        config.train.start_date = "1999-01-01"
        config.train.end_date = "2022-01-01"
        config.train.outcome_cutoff = str(self._options.outcome_cutoff)
        config.build_cube.output.save_predictions = False
        config.build_cube.output.save_signal = False
        config.build_cube.output.save_cv_results = False
        if family is not None:
            config.model.ensemble = [family]
            block = config[model_class(family).config_key]
            block.columns_by_horizon = {**(block.get("columns_by_horizon") or {}), 60: features or []}
            if family != "elasticnet":
                block.categoricals_by_horizon = {**(block.get("categoricals_by_horizon") or {}), 60: categoricals or []}
        return config

    def _development_panel(self, eligible: list[str], categoricals: list[str]) -> tuple[StepLongShort, pd.DataFrame]:
        config = self._runtime_config()
        for family in config.model.ensemble:
            block = config[model_class(str(family)).config_key]
            block.columns_by_horizon = {60: [name for name in eligible if name not in categoricals]}
            if str(family) != "elasticnet":
                block.categoricals_by_horizon = {60: categoricals}
        step = StepLongShort(context=self._context, config=config)
        schema, _ = step._prepare_training()
        panel = step._load_horizon(60, schema, False)
        if panel is None or panel.empty or not np.isfinite(panel[step.label_column].to_numpy(dtype=float)).all():
            raise ValueError("Development requires finite matured target rows")
        return step, panel

    def _development(self) -> None:
        if (self._output / "_out" / "eval" / "started.json").exists():
            raise ValueError("Sealed evaluation has started; development is permanently closed for this run")
        audit_path = self._output / "_out" / "eligibility.json"
        baseline_path = self._output / "_out" / "b0" / "result.json"
        if not audit_path.exists():
            self._audit()
        if not baseline_path.exists():
            self._baseline()
        ledger_path = self._output / "_out" / "selection.json"
        resume = bool(self._options.get("resume", False))
        if ledger_path.exists() and not resume:
            raise ValueError("Selection already exists; explicit --resume is required")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline["source_hashes"] != self._fit_source_identity() or baseline["seed"] != int(self._config.seed):
            raise ValueError("Baseline fit source or seed changed; refuse mixed-experiment comparison")
        self._check_baseline_controls(baseline_path.parent)
        eligible = [row["feature"] for row in audit["features"] if row["role"] != "excluded"]
        categorical = [row["feature"] for row in audit["features"] if row["role"] == "categorical"]
        original_columns = baseline["audit"]["input_digest_columns"]
        extra = [name for name in original_columns if name not in {"date", "ticker", "y", "label_end"}]
        load_columns = list(dict.fromkeys(eligible + extra))
        step, panel = self._development_panel(load_columns, categorical)
        if frame_digest(panel, original_columns) != baseline["audit"]["input_digest"]:
            raise ValueError("Current baseline and wide development inputs differ; refuse mixed-snapshot comparison")
        input_id = frame_digest(panel, [*eligible, step.label_column, "label_end"])
        ledger = (
            json.loads(ledger_path.read_text(encoding="utf-8"))
            if ledger_path.exists()
            else {
                "input_digest": input_id,
                "features": eligible,
                "categoricals": categorical,
                "families": {},
                "frozen": False,
                "fixed_parameters": self._fixed_parameters(),
                "controls": self._controls(),
                "target": "target_zscore_h60",
                "outcome_cutoff": str(self._options.outcome_cutoff),
            }
        )
        if ledger["input_digest"] != input_id or ledger["fixed_parameters"] != self._fixed_parameters() or ledger["controls"] != self._controls():
            raise ValueError("Resume input/fixed-parameter identity mismatch")
        write_json(ledger_path, ledger)
        expected = self._expected_oof(panel)
        for family in ["lgbm", "random_forest", "elasticnet"]:
            if family not in ledger["families"]:
                ledger["families"][family] = self._search(family, panel, expected, eligible, categorical, baseline)
                write_json(ledger_path, ledger)
        ledger["frozen"] = True
        ledger["frozen_at"] = str(pd.Timestamp.now())
        selected = {family: state["best"] for family, state in ledger["families"].items()}
        ledger["selected"] = selected
        write_json(ledger_path, ledger)
        self._write_selected_config(selected)
        final_columns = list(dict.fromkeys([name for candidate in selected.values() for name in candidate["features"]]))
        ledger["final_input_digest"] = frame_digest(panel, [*final_columns, step.label_column, "label_end"])
        write_json(ledger_path, ledger)
        del panel
        gc.collect()
        self._fit_selected(ledger)

    def _fixed_parameters(self) -> dict[str, Any]:
        ignored = {"columns", "columns_by_horizon", "categoricals", "categoricals_by_horizon"}
        return {
            key: {name: value for name, value in cast(dict, OmegaConf.to_container(self._config[key], resolve=True)).items() if name not in ignored}
            for key in ["linear", "lgbm", "random_forest"]
        }

    def _fit_source_identity(self) -> dict[str, str]:
        root = Path(__file__).resolve().parents[3]
        paths = [
            *[
                root / "src" / "modelling" / "transformers" / name
                for name in ["__init__.py", "base.py", "lightgbm_model.py", "random_forest.py", "linear_regression.py"]
            ],
            *[
                root / "src" / "modelling" / "utils" / name
                for name in ["features.py", "panel.py", "cv.py", "ensemble.py", "metrics.py", "artifacts.py"]
            ],
            root / "src" / "modelling" / "steps" / "step_long_short.py",
        ]
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    def _controls(self) -> dict[str, Any]:
        config = self._runtime_config()
        return {
            "seed": int(config.seed),
            "source_hashes": self._fit_source_identity(),
            "shared": {
                key: OmegaConf.to_container(config.model[key], resolve=True) if OmegaConf.is_config(config.model[key]) else config.model[key]
                for key in ["cv", "weight_decay", "label_column", "target_type", "ensemble"]
            },
            "assessment": {
                key: OmegaConf.to_container(self._options[key], resolve=True) if OmegaConf.is_config(self._options[key]) else self._options[key]
                for key in ["horizon", "outcome_cutoff", "evaluation_start", "test_windows", "prune_fraction", "max_rounds", "patience"]
            },
        }

    def _check_baseline_controls(self, output: Path) -> None:
        started = json.loads((output / "started.json").read_text(encoding="utf-8"))
        original = started["config"]
        ignored = {"columns", "columns_by_horizon", "categoricals", "categoricals_by_horizon"}
        original_parameters = {
            key: {name: value for name, value in original[key].items() if name not in ignored} for key in ["linear", "lgbm", "random_forest"]
        }
        if original_parameters != self._fixed_parameters():
            raise ValueError("Current-config baseline used different fixed family parameters")
        current = self._controls()["shared"]
        if any(original["model"][key] != value for key, value in current.items()):
            raise ValueError("Current-config baseline used different shared model controls")

    def _audit(self) -> None:
        config = self._runtime_config()
        columns = [str(column) for column in self._context.store.columns(Tables.cube)]
        config.lgbm.columns_by_horizon = {60: columns}
        step = StepLongShort(context=self._context, config=config)
        projected, _ = step._select_load_columns(set(columns))
        excluded = {
            name: "key" if name in {"date", "ticker"} else "target" if name.startswith("target_") else "operational metadata"
            for name in columns
            if name not in projected[2:]
        }
        excluded.update(
            {
                str(name): "raw/adjusted level or share-unit control; historical adjustment vintage not certified"
                for name in self._options.excluded_columns
                if name in columns
            }
        )
        features = [name for name in projected[2:] if name not in excluded]
        step._prepare_training()
        ends = label_end_dates(step._calendar, 60)
        mature = ends.index[ends.notna() & (ends <= step._training_as_of)]
        if mature.empty:
            raise ValueError("No mature development dates")
        chunks = self._context.store.iter_load(
            Tables.cube,
            columns=["date", "ticker", *features],
            since="1999-01-01",
            until=mature[-1],
            where={"target_zscore_h60": self._context.store.NOT_NULL},
            chunksize=int(self._options.read_chunksize),
        )
        self._log.info("Audit %d predictors on full development population through %s", len(features), mature[-1])
        report, sample = audit_features(
            chunks, features, columns, list(self._options.categorical_features), excluded, int(config.model.diagnostics.shap_sample), int(config.seed)
        )
        report.update(
            {
                "feature_end": str(mature[-1].date()),
                "label_end_cutoff": str(self._options.outcome_cutoff),
                "population": "mature labelled development rows",
                "application_writes": 0,
                "fitting": False,
                "sealed_outcomes_read": False,
            }
        )
        write_json(self._output / "_out" / "eligibility.json", report)
        sample.to_parquet(self._output / "_out" / "eligibility_sample.parquet", index=False)

    def _baseline(self) -> None:
        output = self._output / "_out" / "b0"
        if output.exists():
            raise ValueError("Incomplete baseline directory exists; preserve it and use a fresh assessment run")
        output.mkdir()
        config = self._runtime_config()
        config.model.models_dir = str(output / "m")
        write_json(
            output / "started.json",
            {
                "seed": int(config.seed),
                "source_hashes": self._fit_source_identity(),
                "config": {key: OmegaConf.to_container(config[key], resolve=True) for key in ["model", "linear", "lgbm", "random_forest", "train"]},
            },
        )
        started = time.perf_counter()
        fitted = StepLongShort(context=self._context, config=config).run()
        cv = fitted.cv[60]
        assert cv.oos is not None and cv.oos_members is not None
        oof = cv.oos.rename(columns={"pred": "ensemble"}).merge(cv.oos_members, on=["date", "ticker", "fold"], validate="one_to_one")
        oof.to_parquet(output / "oof.parquet", index=False)
        metrics = {}
        for name in [*fitted.models[60], "ensemble"]:
            metrics[name], series = assess_candidate(oof[["date", "ticker", "y"]], oof[["date", "ticker", name]].rename(columns={name: "pred"}), 60)
            series.rename("ic").to_frame().to_parquet(output / f"ic_{name}.parquet")
        for index, members in enumerate(cv.fold_models, 1):
            for family, member in members.items():
                save_member(member, output / f"f{index}_{family}.pkl")
        write_json(
            output / "result.json",
            {
                "duration_seconds": time.perf_counter() - started,
                "metrics": metrics,
                "folds": cv.folds,
                "member_folds": cv.member_folds,
                "rounds": cv.rounds,
                "audit": cv.audit,
                "features": {name: member.features for name, member in fitted.models[60].items()},
                "categoricals": {name: member.categoricals for name, member in fitted.models[60].items()},
                "source_hashes": self._fit_source_identity(),
                "seed": int(config.seed),
                "attempted_mutations": [],
            },
        )

    def _expected_oof(self, panel: pd.DataFrame) -> pd.DataFrame:
        dates = pd.to_datetime(panel["date"])
        keep = np.zeros(len(panel), dtype=bool)
        for start, end in self._options.test_windows:
            keep |= ((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))).to_numpy()
        return panel.loc[keep, ["date", "ticker", self._config.model.label_column]].copy()

    def _search(self, family: str, panel: pd.DataFrame, expected: pd.DataFrame, eligible: list[str], categorical: list[str], baseline: dict) -> dict:
        names = [name for name in eligible if family != "elasticnet" or name not in categorical]
        old = baseline["metrics"][family]
        best = (
            {"id": "b0", "family": family, "features": baseline["features"][family], "categoricals": baseline["categoricals"][family], "metrics": old}
            if old["eligible"]
            else None
        )
        candidates, failures = [], 0
        prefix = {"lgbm": "l", "random_forest": "r", "elasticnet": "e"}[family]
        for round_no in range(int(self._options.max_rounds) + 1):
            identifier = f"{prefix}{round_no}"
            candidate = self._candidate(identifier, family, names, categorical, panel, expected)
            candidates.append(candidate)
            objective = candidate["metrics"]["objective"]
            improved = objective is not None and (best is None or objective > best["metrics"]["objective"])
            tie = objective is not None and best is not None and objective == best["metrics"]["objective"] and len(names) < len(best["features"])
            if improved or tie:
                best = candidate
            if round_no:
                failures = 0 if improved else failures + 1
            self._log.info(
                "%s round %d: %d inputs, pooled IC=%s IR=%s, eligible=%s, failures=%d",
                family,
                round_no,
                len(names),
                candidate["metrics"]["mean_ic"],
                candidate["metrics"]["ic_ir"],
                candidate["metrics"]["eligible"],
                failures,
            )
            if failures >= int(self._options.patience) or round_no == int(self._options.max_rounds):
                break
            importance = json.loads((self._output / "_out" / identifier / "importance.json").read_text(encoding="utf-8"))
            names, removed, reason = prune_pool(names, importance, float(self._options.prune_fraction))
            candidate["pruning"] = {"removed": removed, "reason": reason}
            if not removed:
                break
        if best is None:
            raise ValueError(f"{family}: no candidate has a finite complete-population objective; no hyperparameter fallback permitted")
        return {"best": best, "candidates": candidates, "non_improving_rounds": failures}

    def _candidate(self, identifier: str, family: str, names: list[str], categorical: list[str], panel: pd.DataFrame, expected: pd.DataFrame) -> dict:
        output = self._output / "_out" / identifier
        completed = output / "result.json"
        if completed.exists():
            if not self._options.get("resume", False):
                raise ValueError("Candidate exists without explicit resume")
            result = json.loads(completed.read_text(encoding="utf-8"))
            if result["features"] != names:
                raise ValueError("Candidate resume feature identity mismatch")
            return result
        output.mkdir(parents=True, exist_ok=True)
        cats = [name for name in names if name in categorical] if family != "elasticnet" else []
        config = self._runtime_config(family, [name for name in names if name not in cats], cats)
        step = StepLongShort(context=self._context, config=config)
        step._prepare_training()
        started = time.perf_counter()
        cv = step._cross_validate(60, panel, retain_models=True)
        assert cv.oos is not None
        stats, series = assess_candidate(expected, cv.oos, 60, str(config.model.label_column))
        cv.oos.to_parquet(output / "oof.parquet", index=False)
        series.rename("ic").to_frame().to_parquet(output / "ic.parquet")
        for index, members in enumerate(cv.fold_models, 1):
            save_member(members[family], output / f"f{index}.pkl")
        importance = self._importance(cv.oos, cv.fold_models, family, panel, output)
        write_json(output / "importance.json", importance)
        result = {
            "id": identifier,
            "family": family,
            "features": names,
            "categoricals": cats,
            "metrics": stats,
            "folds": cv.folds,
            "rounds": cv.rounds,
            "duration_seconds": time.perf_counter() - started,
        }
        write_json(completed, result)
        return result

    def _importance(self, oof: pd.DataFrame, folds: list[dict[str, BaseModel]], family: str, panel: pd.DataFrame, output: Path) -> dict[str, float]:
        sampled = oof.iloc[balanced_sample_positions(oof, int(self._config.model.diagnostics.shap_sample), int(self._config.seed))]
        features = folds[0][family].features
        totals = pd.Series(0.0, index=features)
        n_dates = oof["date"].nunique()
        values_frames, feature_frames = [], []
        for index, members in enumerate(folds, 1):
            member = members[family]
            dates = oof.loc[oof["fold"] == index, "date"].nunique()
            if not member.supports_shap:
                totals += member.importance().reindex(features) * dates / n_dates
                continue
            keys = sampled.loc[sampled["fold"] == index, ["date", "ticker", "fold"]]
            rows = keys.merge(panel[["date", "ticker", *features]], on=["date", "ticker"], validate="one_to_one")
            matrix = design_matrix(rows, features, member.categoricals, downcast=False)
            got = shap_row_values(member.model, matrix, features, sample=0, logger=self._log)
            if got is None:
                raise RuntimeError("Required held-out SHAP failed")
            values, positions = got
            native = np.asarray(member.model.predict(matrix, pred_contrib=True)) * shap_score_scale(member.model)
            predictions = member.predict(rows).to_numpy()
            if (
                not np.array_equal(positions, np.arange(len(rows)))
                or not np.allclose(native.sum(axis=1), predictions, rtol=1e-6, atol=1e-7)
                or not np.allclose(values, native[:, :-1], rtol=1e-6, atol=1e-7)
            ):
                raise ValueError("SHAP/scoring additivity or input parity failed")
            df_values = pd.DataFrame(values, columns=features)
            df_values["date"] = rows["date"].to_numpy()
            totals += df_values.groupby("date")[features].apply(lambda group: group.abs().mean()).sum() / n_dates
            df_values.insert(1, "ticker", rows["ticker"].to_numpy())
            df_values["fold"] = index
            df_values["base"] = native[:, -1]
            values_frames.append(df_values)
            feature_frames.append(rows)
        if values_frames:
            pd.concat(values_frames, ignore_index=True).to_parquet(output / "shap.parquet", index=False)
            pd.concat(feature_frames, ignore_index=True).to_parquet(output / "x.parquet", index=False)
        return {str(name): float(value) for name, value in totals.items()}

    def _write_selected_config(self, selected: dict) -> None:
        before = self._fixed_parameters()
        for family, candidate in selected.items():
            key = model_class(family).config_key
            path = self._config_dir / "models" / f"{key}_modelling.yml"
            original = path.read_text(encoding="utf-8")
            cats = candidate["categoricals"]
            numeric = [name for name in candidate["features"] if name not in cats]
            updated = set_horizon_features(original, key, "columns_by_horizon", numeric)
            if family != "elasticnet":
                updated = set_horizon_features(updated, key, "categoricals_by_horizon", cats)
            block = cast(DictConfig, OmegaConf.create(updated))[key]
            ignored = {"columns", "columns_by_horizon", "categoricals", "categoricals_by_horizon"}
            fixed = {name: value for name, value in cast(dict, OmegaConf.to_container(block, resolve=True)).items() if name not in ignored}
            if fixed != before[key]:
                raise ValueError("Feature YAML edit changed fixed model parameters")
            path.write_text(updated, encoding="utf-8")

    def _winning_oof(self, candidate: dict) -> pd.DataFrame:
        path = self._output / "_out" / candidate["id"] / "oof.parquet"
        frame = pd.read_parquet(path)
        if candidate["id"] == "b0":
            frame = frame[["date", "ticker", "fold", "y", candidate["family"]]].rename(columns={candidate["family"]: "pred"})
        return frame

    def _fit_selected(self, ledger: dict) -> None:
        final = self._output / "_out" / "final"
        if (final / "result.json").exists() and self._options.get("resume", False):
            result = json.loads((final / "result.json").read_text(encoding="utf-8"))
            if result["input_digest"] != ledger["final_input_digest"]:
                raise ValueError("Final artifact input identity mismatch")
            return
        config = self._runtime_config()
        config.model.models_dir = str(final / "m")
        for family, candidate in ledger["selected"].items():
            block = config[model_class(family).config_key]
            cats = candidate["categoricals"]
            block.columns_by_horizon = {60: [name for name in candidate["features"] if name not in cats]}
            if family != "elasticnet":
                block.categoricals_by_horizon = {60: cats}
        fitted = StepLongShort(context=self._context, config=config).run()
        cv = fitted.cv[60]
        if cv.audit["input_digest"] != ledger["final_input_digest"]:
            raise ValueError("Final training data changed after selection froze")
        assert cv.oos is not None and cv.oos_members is not None
        combined = cv.oos.rename(columns={"pred": "ensemble"}).merge(cv.oos_members, on=["date", "ticker", "fold"], validate="one_to_one")
        for family, candidate in ledger["selected"].items():
            original = self._winning_oof(candidate)
            compared = combined[["date", "ticker", family]].merge(original[["date", "ticker", "pred"]], on=["date", "ticker"], validate="one_to_one")
            if len(compared) != len(combined) or not np.array_equal(compared[family], compared["pred"], equal_nan=True):
                raise ValueError(f"{family}: frozen winning-fold predictions did not reproduce")
        lgbm_rounds = ledger["selected"]["lgbm"].get("rounds")
        if lgbm_rounds is None:
            lgbm_rounds = json.loads((self._output / "_out" / "b0" / "result.json").read_text(encoding="utf-8"))["rounds"]
        if cv.rounds != lgbm_rounds:
            raise ValueError("Frozen LGBM fold round counts changed")
        combined.to_parquet(final / "oof.parquet", index=False)
        for index, members in enumerate(cv.fold_models, 1):
            for family, member in members.items():
                save_member(member, final / f"f{index}_{family}.pkl")
        metrics = {}
        expected = combined[["date", "ticker", "y"]]
        for name in [*fitted.models[60], "ensemble"]:
            metrics[name], _ = assess_candidate(expected, combined[["date", "ticker", name]].rename(columns={name: "pred"}), 60)
        artifacts = {str(path.relative_to(final)): hashlib.sha256(path.read_bytes()).hexdigest() for path in final.rglob("*.pkl")}
        write_json(
            final / "result.json",
            {
                "input_digest": ledger["final_input_digest"],
                "metrics": metrics,
                "rounds": cv.rounds,
                "artifacts": artifacts,
                "audit": cv.audit,
                "selection_frozen_at": ledger["frozen_at"],
            },
        )

    def _evaluation(self) -> None:
        output = self._output / "_out" / "eval"
        ledger = json.loads((self._output / "_out" / "selection.json").read_text(encoding="utf-8"))
        final = self._output / "_out" / "final"
        fitted = json.loads((final / "result.json").read_text(encoding="utf-8"))
        if not ledger["frozen"] or fitted["selection_frozen_at"] != ledger["frozen_at"]:
            raise ValueError("Evaluation requires the frozen selection and matching cutoff artifacts")
        if ledger["controls"] != self._controls() or ledger["fixed_parameters"] != self._fixed_parameters():
            raise ValueError("Evaluation experiment controls changed after development")
        actual = {str(path.relative_to(final)): hashlib.sha256(path.read_bytes()).hexdigest() for path in final.rglob("*.pkl")}
        if actual != fitted["artifacts"]:
            raise ValueError("Frozen model artifacts changed before evaluation")
        if (output / "result.json").exists():
            if not self._options.get("resume", False):
                raise ValueError("Evaluation exists; use explicit --resume for the same frozen models")
            return
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            output / "started.json",
            {
                "frozen_at": ledger["frozen_at"],
                "models": actual,
                "started_at": pd.Timestamp.now(),
                "policy": "cohorts committed before outcome reads; no further fitting or feature selection",
            },
        )
        scores, cohorts, calendar = self._score_universes(ledger, output)
        self._attach_outcomes(scores, cohorts, calendar, output)
        from_result = pd.read_parquet(output / "predictions.parquet")
        models = [*self._config.model.ensemble, "ensemble"]
        metrics = {}
        diagnostic = self._config.model.diagnostics
        for name, population in [("development", from_result.loc[from_result.fold > 0]), ("sealed", from_result.loc[from_result.fold == 0])]:
            metrics[name], daily = temporal_diagnostics(
                population,
                models,
                60,
                int(diagnostic.rolling_window),
                int(diagnostic.bootstrap_repetitions),
                float(diagnostic.confidence),
                int(self._config.seed),
            )
            daily.to_parquet(output / f"daily_{name}.parquet", index=False)
        self._explanations(output)
        write_json(
            output / "result.json",
            {
                "metrics": metrics,
                "frozen_at": ledger["frozen_at"],
                "model_artifacts": actual,
                "formation_end": str(scores.date.max().date()),
                "calendar_end": str(calendar[-1].date()),
                "score_rows": len(scores),
                "cohort_rows": len(cohorts),
                "no_refitting": True,
            },
        )

    def _score_universes(self, ledger: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
        calendar = pd.DatetimeIndex(self._context.store.distinct(Tables.cube_part_prices, "date")).normalize().sort_values().unique()
        cube_latest = pd.Timestamp(self._context.store.distinct(Tables.cube, "date", order="desc", limit=1)[0])
        calendar = calendar[calendar <= cube_latest]
        if len(calendar) <= 60:
            raise ValueError("Insufficient observed calendar for mature evaluation")
        end = calendar[-61]
        windows = [
            (i, pd.Timestamp(start), min(pd.Timestamp(stop), pd.Timestamp(ledger["outcome_cutoff"])))
            for i, (start, stop) in enumerate(self._options.test_windows, 1)
        ]
        development_end = label_end_dates(calendar, 60)
        windows = [
            (i, start, min(stop, development_end.index[(development_end <= pd.Timestamp(ledger["outcome_cutoff"])) & development_end.notna()][-1]))
            for i, start, stop in windows
        ]
        windows.append((0, pd.Timestamp(self._options.evaluation_start), end))
        final = self._output / "_out" / "final"
        features = list(dict.fromkeys([name for candidate in ledger["selected"].values() for name in candidate["features"]]))
        categories = list(dict.fromkeys([name for candidate in ledger["selected"].values() for name in candidate["categoricals"]]))
        projections = list(dict.fromkeys(["date", "ticker", "sector", *features]))
        score_frames, identities = [], []
        total_development_dates = sum(int(((calendar >= start) & (calendar <= stop)).sum()) for fold, start, stop in windows if fold)
        for fold, start, stop in windows:
            members = {
                str(name): load_member(final / (f"f{fold}_{name}.pkl" if fold else f"m/model_h60_{name}.pkl")) for name in self._config.model.ensemble
            }
            panel = load_frame(
                self._context.store,
                Tables.cube,
                columns=projections,
                since=start,
                until=stop,
                downcast=bool(fold),
                stream=True,
                feature_cols=features,
                categorical_features=categories,
                chunksize=int(self._options.read_chunksize),
                optional=False,
            )
            if panel is None or panel.empty:
                raise ValueError(f"No feature-only assessment universe for fold {fold}")
            scores, member_scores = ensemble_predict(members, panel)
            frame = panel[["date", "ticker"]].copy()
            for family, prediction in member_scores.items():
                frame[family] = prediction.to_numpy()
            frame["ensemble"], frame["fold"] = scores.to_numpy(), fold
            score_frames.append(frame)
            budget = int(self._config.model.diagnostics.shap_sample)
            if fold:
                budget = max(1, round(budget * panel.date.nunique() / total_development_dates))
            positions = balanced_sample_positions(panel, budget, int(self._config.seed))
            panel.iloc[positions].to_parquet(output / f"s{fold}.parquet", index=False)
            self._coverage(panel, features, categories).to_parquet(output / f"q{fold}.parquet", index=False)
            identities.append(
                {
                    "fold": fold,
                    "rows": len(panel),
                    "feature_min": start,
                    "feature_max": stop,
                    "digest": frame_digest(panel, features),
                    "sample_rows": len(positions),
                    "future_outcome_predicates": False,
                }
            )
            self._log.info("Feature-only scoring fold %d: %d rows, %d dates; no outcome filter", fold, len(panel), panel.date.nunique())
            del panel, members
            gc.collect()
        combined = pd.concat(score_frames, ignore_index=True)
        cohorts = formation_cohorts(combined, [*self._config.model.ensemble, "ensemble"])
        combined.to_parquet(output / "scores.parquet", index=False)
        cohorts.to_parquet(output / "cohorts.parquet", index=False)
        write_json(
            output / "feature_prediction_manifest.json",
            {
                "windows": identities,
                "cohort_assignment_complete": True,
                "future_labels_read": False,
                "policy": "highest predicted ceil(N/10), equal weights, ticker tie-break",
            },
        )
        return combined, cohorts, calendar

    def _coverage(self, panel: pd.DataFrame, features: list[str], categories: list[str]) -> pd.DataFrame:
        values = panel[features].notna()
        for category in categories:
            values[category] &= panel[category].ge(0)
        identifiers = [panel.date.dt.year.rename("year"), panel.sector.rename("sector")]
        observed = (
            values.groupby(identifiers, dropna=False).sum().reset_index().melt(id_vars=["year", "sector"], var_name="feature", value_name="observed")
        )
        population = panel.groupby(identifiers, dropna=False).size().rename("rows").reset_index()
        return observed.merge(population, on=["year", "sector"], validate="many_to_one")

    def _attach_outcomes(self, scores: pd.DataFrame, cohorts: pd.DataFrame, calendar: pd.DatetimeIndex, output: Path) -> None:
        if not (output / "feature_prediction_manifest.json").exists() or not (output / "cohorts.parquet").exists():
            raise ValueError("Commit feature-only cohorts before loading outcomes")
        start, stop = scores.date.min(), scores.date.max()
        labels = load_frame(
            self._context.store,
            Tables.cube,
            columns=["date", "ticker", "target_zscore_h60"],
            since=start,
            until=stop,
            stream=True,
            downcast=False,
            chunksize=int(self._options.read_chunksize),
            optional=False,
        )
        assert labels is not None
        labels = labels.rename(columns={"target_zscore_h60": "y"})
        labels["label_end"] = labels.date.map(label_end_dates(calendar, 60))
        predictions = scores.merge(labels, on=["date", "ticker"], how="left", validate="one_to_one")
        predictions.to_parquet(output / "predictions.parquet", index=False)
        price_stop = label_end_dates(calendar, 60).loc[stop]
        prices = load_frame(
            self._context.store,
            Tables.cube_part_prices,
            columns=["date", "ticker", "ret", "close_total"],
            since=start,
            until=price_stop,
            stream=True,
            downcast=False,
            chunksize=int(self._options.read_chunksize),
            optional=False,
        )
        assert prices is not None
        scoped_calendar = calendar[(calendar >= start) & (calendar <= price_stop)]
        outcomes = forward_returns(prices, scoped_calendar, 60)
        outcomes.to_parquet(output / "return_windows.parquet", index=False)
        macro = load_frame(
            self._context.store,
            Tables.prices_macro,
            columns=["date", "ticker", "close"],
            where={"ticker": "equity_tr"},
            since=start,
            until=price_stop,
            downcast=False,
            optional=False,
        )
        assert macro is not None
        macro = macro.set_index("date").reindex(scoped_calendar)
        year_ends = macro.close.groupby(pd.DatetimeIndex(macro.index).year).last()
        write_json(output / "market_calendar_years.json", {str(year): float(value) for year, value in year_ends.pct_change(fill_method=None).items()})
        macro["ret"] = macro.close.pct_change(fill_method=None)
        macro["ticker"] = "equity_tr"
        market = forward_returns(macro.rename(columns={"close": "close_total"}).reset_index(names="date"), scoped_calendar, 60)
        market.to_parquet(output / "market_windows.parquet", index=False)
        market = market[["date", "realized_return", "valid"]].rename(columns={"realized_return": "market_return", "valid": "market_valid"})
        matched = matched_cohort_returns(cohorts, outcomes, market)
        matched["period"] = np.where(matched.date >= pd.Timestamp(self._options.evaluation_start), "sealed", "development")
        matched.to_parquet(output / "matched_returns.parquet", index=False)

    def _explanations(self, output: Path) -> None:
        final = self._output / "_out" / "final"
        for fold in range(6):
            panel = pd.read_parquet(output / f"s{fold}.parquet")
            for family in ["lgbm", "random_forest"]:
                member = load_member(final / (f"f{fold}_{family}.pkl" if fold else f"m/model_h60_{family}.pkl"))
                destination = output / f"{family}_{fold}"
                digest = frame_digest(panel, member.features)
                if (destination / "curves.json").exists() and self._options.get("resume", False):
                    saved = json.loads((destination / "identity.json").read_text(encoding="utf-8"))
                    if saved["input_digest"] != digest:
                        raise ValueError("Explanation sample changed on resume")
                    continue
                explain_tree(member, panel, destination, self._config, self._log, pdp=fold == 0)
                write_json(destination / "identity.json", {"input_digest": digest, "fold": fold, "family": family})
        linear = load_member(final / "m/model_h60_elasticnet.pkl")
        write_json(
            output / "linear.json",
            {
                "features": linear.features,
                "coefficient": linear.coef_,
                "mean": linear.mean_,
                "std": linear.std_,
                "intercept": linear.intercept_,
                "policy": "coefficient per training-standardized feature; missing imputed to training mean",
            },
        )

    def _report(self) -> None:
        render_assessment(self._output, self._config)
