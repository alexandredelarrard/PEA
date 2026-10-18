"""Independently reconcile saved model runs with arithmetic, native predictions and scoped sources.

No modelling helper is imported. Hashes are checked before loading a fitted object; all source
reads use context.store. The check never fits, writes application tables or promotes models.
"""

from __future__ import annotations

import gc
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr

from src.context import Context
from src.data_store.schema import Tables
from src.validate.result import CheckResult, Finding


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Non-standard JSON {value}")))


def _file_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(handle, "sha256").hexdigest()}


def _inventory(directory: Path, inventory: dict[str, Any]) -> None:
    if not inventory:
        raise ValueError("Missing required artifact inventory")
    for relative, expected in inventory.items():
        path = (directory / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(directory.resolve()) or path == directory.resolve():
            raise ValueError("Artifact path must stay inside its declared folder")
        if _file_identity(path) != expected:
            raise ValueError(f"Artifact identity/hash changed: {relative}")


def _finding(findings: list[Finding], condition: bool, observed: str, expected: str, **evidence: Any) -> None:
    if not condition:
        findings.append(Finding.at(8, observed, expected, **evidence))


def independent_daily(predictions: pd.DataFrame, model: str) -> pd.Series:
    """Compute finite-pair rank correlation independently of producer metric helpers."""
    output = {}
    for date, group in predictions.groupby("date", sort=True):
        x, y = group[model].to_numpy(dtype=float), group.y.to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 3 and len(np.unique(x[valid])) >= 3 and len(np.unique(y[valid])) >= 3:
            output[date] = float(cast(Any, spearmanr(x[valid], y[valid])).statistic)
    result = pd.Series(output, name="ic", dtype=float).rename_axis("date")
    result.index = pd.DatetimeIndex(result.index).as_unit("ns")
    return result


def _reconcile_configuration(config: DictConfig, training: dict, cutoff: dict, findings: list[Finding]) -> None:
    shared = config.model
    payload = {
        "features": cutoff["feature_cols"] + cutoff["categorical_cols"],
        "horizons": training["horizons"],
        "blocks": {key: OmegaConf.to_container(config[key], resolve=True) for key in ("linear", "lgbm", "random_forest", "train") if key in config},
        "shared": {
            key: OmegaConf.to_container(shared[key], resolve=True) if OmegaConf.is_config(shared.get(key)) else shared.get(key)
            for key in ("ensemble", "cv", "weight_decay", "target_type", "label_column")
        },
        "seed": config.get("seed"),
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    valid = signature == training["selection_signature"] == cutoff["selection_signature"]
    valid &= list(config.model.horizons) == training["horizons"] and list(config.model.ensemble) == training["model_types"]
    _finding(
        findings,
        valid,
        "Current fitting configuration differs from saved run",
        "Identical features, horizons, parameters, cutoff, folds and seed",
        field="configuration",
    )


def _reconcile_metrics(output: Path, horizon: int, models: list[str], window: int, findings: list[Finding]) -> dict[str, Any]:
    predictions, saved, summaries = (
        pd.read_parquet(output / "predictions.parquet"),
        pd.read_parquet(output / "daily.parquet"),
        _read(output / "metrics.json"),
    )
    measured = {}
    _finding(findings, set(saved.horizon) == {horizon}, "Mixed daily metric horizons", "Only the declared horizon", horizon=horizon)
    _finding(
        findings,
        set(saved.population) == {"validation", "backtest"} and set(saved.model) == set(models),
        "Missing/mixed metric series",
        "Every family and ensemble in both samples",
        horizon=horizon,
    )
    for population, mask in [("validation", predictions.fold > 0), ("backtest", predictions.fold == 0)]:
        panel = predictions.loc[mask]
        if panel.duplicated(["date", "ticker"]).any():
            raise ValueError("Duplicate prediction keys")
        sessions = pd.DatetimeIndex(sorted(panel.date.unique())).as_unit("ns")
        measured[population] = {}
        for model in models:
            expected = independent_daily(panel, model).reindex(sessions)
            actual = saved.loc[(saved.population == population) & (saved.model == model)].sort_values("date")
            if actual.date.duplicated().any() or not np.array_equal(
                actual.date.to_numpy(dtype="datetime64[ns]"), sessions.to_numpy(dtype="datetime64[ns]")
            ):
                _finding(
                    findings,
                    False,
                    "Dropped/duplicate observed IC sessions",
                    "Exact scoring-session slots including undefined IC",
                    horizon=horizon,
                    population=population,
                    model=model,
                )
                continue
            finite = np.isfinite(panel.y) & np.isfinite(panel[model])
            paired = panel.loc[finite]
            rolling = expected.rolling(window, min_periods=window).mean()
            dispersion = expected.rolling(window, min_periods=window).std(ddof=0)
            ratio = rolling / dispersion.where(dispersion > 0) * np.sqrt(252 / horizon)
            arrays = {
                "ic": expected,
                "rolling_ic": rolling,
                "rolling_ir": ratio,
                "paired": paired.groupby("date").size().reindex(sessions, fill_value=0),
                "universe": panel.groupby("date").size().reindex(sessions, fill_value=0),
                "prediction_unique": paired.groupby("date")[model].nunique().reindex(sessions, fill_value=0),
                "target_unique": paired.groupby("date").y.nunique().reindex(sessions, fill_value=0),
            }
            for name, values in arrays.items():
                _finding(
                    findings,
                    np.allclose(actual[name], values, equal_nan=True, atol=1e-10, rtol=1e-8),
                    f"Saved {name} differs",
                    "Independent finite-pair observed-session calculation",
                    horizon=horizon,
                    population=population,
                    model=model,
                )
            values = expected.dropna().to_numpy()
            mean, std = (float(values.mean()), float(values.std())) if len(values) else (np.nan, np.nan)
            stats = {"mean_ic": mean, "ic_std": std, "ic_ir": mean / std * np.sqrt(252 / horizon) if std > 0 else np.nan, "n_days": len(values)}
            measured[population][model] = stats
            for name, expected_value in stats.items():
                actual_value = summaries[population][model][name]
                _finding(
                    findings,
                    np.isclose(np.nan if actual_value is None else actual_value, expected_value, equal_nan=True, atol=1e-10),
                    f"Saved pooled {name} differs",
                    "Independent pooled daily-IC summary",
                    horizon=horizon,
                    population=population,
                    model=model,
                )
    return measured


def _calendar_ends(calendar: pd.DatetimeIndex, horizon: int) -> pd.Series:
    values = calendar.to_numpy(dtype="datetime64[ns]")
    return pd.Series(np.r_[values[horizon:], np.full(horizon, np.datetime64("NaT", "ns"))], index=calendar)


def _source_horizon(
    context: Context, run: Path, training: dict, cutoff: dict, horizon: int, calendar: pd.DatetimeIndex, chunksize: int, findings: list[Finding]
) -> int:
    panel = pd.read_parquet(run / f"assessment/h{horizon}/predictions.parquet")
    target = f"target_{training['target_type']}_h{horizon}"
    chunks = list(
        context.store.iter_load(Tables.cube, columns=["date", "ticker", target], since=panel.date.min(), until=panel.date.max(), chunksize=chunksize)
    )
    source = pd.concat(chunks, ignore_index=True).rename(columns={target: "source_y"})
    joined = panel.merge(source, on=["date", "ticker"], how="left", validate="one_to_one", indicator=True)
    ends = joined.date.map(_calendar_ends(calendar, horizon))
    known = ends.notna() & ((joined.fold == 0) | (ends <= pd.Timestamp(cutoff["training_as_of"])))
    expected = joined.source_y.where(known & np.isfinite(joined.source_y))
    _finding(
        findings,
        bool((joined["_merge"] == "both").all()) and np.allclose(joined.y, expected, equal_nan=True, atol=1e-12),
        "Saved outcome population/value differs from current source",
        "Independent scoped source target/maturity replay",
        horizon=horizon,
        rows=len(panel),
    )
    _finding(
        findings,
        np.array_equal(pd.to_datetime(joined.label_end).to_numpy(dtype="datetime64[ns]"), ends.to_numpy(dtype="datetime64[ns]"), equal_nan=True),
        "Saved label endpoints differ",
        "Complete observed calendar shifted by the horizon",
        horizon=horizon,
    )
    for number, audit in enumerate(cutoff["selection"][str(horizon)]["folds"], 1):
        outer = pd.Timestamp(audit["outer_train_label_end_max"]) < pd.Timestamp(audit["test_start"])
        inner = audit["validation_start"] is None or pd.Timestamp(audit["inner_train_label_end_max"]) < pd.Timestamp(audit["validation_start"])
        validation = audit["validation_label_end_max"] is None or pd.Timestamp(audit["validation_label_end_max"]) < pd.Timestamp(audit["test_start"])
        _finding(
            findings,
            outer and inner and validation,
            "Training/early-stop outcomes overlap validation",
            "Strict inner and outer label-end purges",
            horizon=horizon,
            fold=number,
        )
    _finding(
        findings,
        pd.Timestamp(cutoff["fit_audit"][str(horizon)]["label_end_max"])
        <= pd.Timestamp(cutoff["training_as_of"])
        < panel.loc[panel.fold == 0, "date"].min(),
        "Cutoff model observes evaluation outcomes",
        "All cutoff outcomes known strictly before backtest",
        horizon=horizon,
    )
    return len(panel)


def _raw_prediction(member: Any, matrix: np.ndarray) -> np.ndarray:
    if member.family in {"elasticnet", "ridge"}:
        scaled = np.nan_to_num((matrix - member.mean_) / member.std_, nan=0.0)
        return scaled @ member.coef_ + member.intercept_
    return np.asarray(member.model.predict(matrix))


def _linear_signed_checks(member: Any, matrix: np.ndarray, phi: np.ndarray, curves: dict, config: DictConfig, findings: list[Finding]) -> None:
    """Replay chart coordinates from raw rows and each member's saved training scaler."""
    budget = int(config.model.diagnostics.dependence_sample)
    expected_rows = budget if 0 < budget < len(matrix) else len(matrix)
    for index, feature in enumerate(member.features):
        values = curves["features"][feature]
        scatter = values.get("scatter", {})
        rows = np.asarray(scatter.get("row", []))
        valid_rows = (
            rows.ndim == 1
            and rows.dtype.kind in "iu"
            and len(rows) == expected_rows
            and bool(((rows >= 0) & (rows < len(matrix))).all())
            and bool((np.diff(rows) > 0).all())
        )
        _finding(findings, valid_rows, "Invalid linear scatter row provenance", "Ordered sampled positions in saved raw inputs", feature=feature)
        if not valid_rows:
            continue
        column = matrix[:, index]
        signed_x = (column - member.mean_[index]) / member.std_[index]
        frame = pd.DataFrame({"x": signed_x, "raw_x": column, "mean": phi[:, index]})
        frame = frame.loc[np.isfinite(column)]
        bins = frame.x if frame.x.nunique() <= 10 else pd.qcut(frame.x, 10, duplicates="drop")
        expected = frame.groupby(bins, observed=True).agg(x=("x", "median"), raw_x=("raw_x", "median"), mean=("mean", "mean"), count=("mean", "size"))
        comparisons = [(scatter, {"x": signed_x[rows], "raw_x": column[rows], "shap": phi[rows, index]})]
        comparisons.append((values.get("binned", {}), {key: expected[key].to_numpy() for key in expected.columns}))
        valid = values.get("coefficient") == member.coef_[index]
        for actual, wanted in comparisons:
            for key, array in wanted.items():
                saved = np.asarray(actual.get(key, []), dtype=float)
                valid &= saved.shape == array.shape and np.allclose(saved, array, equal_nan=True, atol=1e-10, rtol=1e-8)
        _finding(
            findings,
            bool(valid),
            "Linear signed chart coordinates/hover differ from saved scaling",
            "Per-member training-standardized scatter/bin x, native contributions, raw hover and signed beta; missing raw states excluded from bins",
            feature=feature,
            family=member.family,
            horizon=member.horizon,
        )


def _effect_checks(folder: Path, member: Any, config: DictConfig, cutoff: bool, findings: list[Finding]) -> int:
    features = member.features
    x = pd.read_parquet(folder / "x.parquet")
    curves = _read(folder / "curves.json")
    linear = member.family in {"elasticnet", "ridge"}
    raw = pd.read_parquet(folder / ("contributions.parquet" if linear else "shap.parquet"))
    _finding(
        findings,
        set(curves["features"]) == set(features) and list(raw.columns) == ["date", "ticker", *features, "base", "prediction"],
        "Missing/misordered retained variable effects",
        "Every actual fitted feature including zeros",
        family=member.family,
        horizon=member.horizon,
    )
    if not x[["date", "ticker"]].equals(raw[["date", "ticker"]]):
        raise ValueError("Explanation input and contribution keys differ")
    matrix = x[features].to_numpy(dtype=float)
    prediction = _raw_prediction(member, matrix)
    if linear:
        phi = np.nan_to_num((matrix - member.mean_) / member.std_, nan=0.0) * member.coef_
        _linear_signed_checks(member, matrix, phi, curves, config, findings)
        base = np.full(len(matrix), member.intercept_)
        importance = np.abs(member.coef_)
        coefficients = pd.read_parquet(folder / "coefficients.parquet")
        _finding(
            findings,
            coefficients.feature.tolist() == features
            and np.allclose(coefficients[["coefficient", "mean", "std"]], np.column_stack([member.coef_, member.mean_, member.std_])),
            "Saved coefficient order/scaler differs",
            "Actual fitted standardized coefficients/scaler",
            family=member.family,
            horizon=member.horizon,
        )
    else:
        native = np.asarray(member.model.predict(matrix, pred_contrib=True))
        shape = member.model.dump_model(num_iteration=1)
        if shape["average_output"] and shape["objective"] == "regression":
            native = native / member.model.num_trees()
        phi, base = native[:, :-1], native[:, -1]
        importance = raw[features].abs().groupby(raw.date.to_numpy()).mean().mean().to_numpy()
    _finding(
        findings,
        np.allclose(raw[features], phi, atol=1e-7, rtol=1e-6)
        and np.allclose(raw.base, base, atol=1e-7)
        and np.allclose(raw.prediction, prediction, atol=1e-7)
        and np.allclose(phi.sum(axis=1) + base, prediction, atol=1e-7, rtol=1e-6),
        "Signed contributions do not reconstruct native predictions",
        "Exact linear decomposition or RF-scale-correct native tree additivity",
        family=member.family,
        horizon=member.horizon,
    )
    _finding(
        findings,
        np.allclose([curves["features"][feature]["importance"] for feature in features], importance, atol=1e-10),
        "Saved variable importance differs",
        "abs(beta) for linear; date-balanced mean abs(SHAP) for trees",
        family=member.family,
        horizon=member.horizon,
    )
    if cutoff:
        options = config.model.diagnostics
        background = (
            np.random.default_rng(int(config.seed)).choice(len(matrix), int(options.pdp_background), replace=False)
            if len(matrix) > int(options.pdp_background)
            else np.arange(len(matrix))
        )
        selected = {0, len(features) // 2, len(features) - 1}
        selected.update(features.index(name) for name in member.categoricals)
        if linear and (np.asarray(member.coef_) == 0).any():
            selected.add(int(np.flatnonzero(np.asarray(member.coef_) == 0)[0]))
        for index, feature in enumerate(features):
            column = matrix[:, index]
            good = np.isfinite(column) & ((column >= 0) if feature in member.categoricals else True)
            grid = np.unique(column[good])
            if feature not in member.categoricals and len(grid) > int(options.pdp_grid):
                grid = np.unique(np.quantile(column[good], np.linspace(0.02, 0.98, int(options.pdp_grid))))
            saved = curves["features"][feature]["pdp"]
            _finding(
                findings,
                (saved["x"] is None and not len(grid)) or (saved["x"] is not None and np.array_equal(saved["x"], grid)),
                "PDP grid invents/drops observed states",
                "Observed categorical/low-cardinality states or configured quantiles",
                feature=feature,
                family=member.family,
                horizon=member.horizon,
            )
            if index in selected and len(grid):
                for point in {0, len(grid) // 2, len(grid) - 1}:
                    work = matrix[background].copy()
                    work[:, index] = grid[point]
                    _finding(
                        findings,
                        np.isclose(saved["mean"][point], _raw_prediction(member, work).mean(), atol=1e-7, rtol=1e-6),
                        "PDP response differs from native prediction",
                        "Direct same-background native prediction with only this input replaced",
                        feature=feature,
                        family=member.family,
                        horizon=member.horizon,
                    )
    return len(raw)


def _model_checks(
    context: Context, run: Path, training: dict, cutoff: dict, horizon: int, state: dict, config: DictConfig, findings: list[Finding]
) -> int:
    explained, hashes = 0, {name: [] for name in training["model_types"]}
    for fold in range(int(config.model.cv.n_splits) + 1):
        fitted = run / (f"folds/h{horizon}/fold{fold}" if fold else "cutoff")
        metadata = _read(fitted / "metadata.json")
        inventory = metadata["member_files"]
        expected = {f"model_h{horizon}_{name}.pkl" for name in training["model_types"]}
        _finding(findings, expected <= set(inventory), "Missing fitted member", "All declared family/horizon members", horizon=horizon, fold=fold)
        _inventory(fitted, inventory)
        if fold:
            _inventory(fitted, metadata["evidence_files"])
            _finding(
                findings,
                metadata["run_id"] == training["run_id"] and metadata["horizons"] == [horizon] and metadata["fold"] == fold,
                "Mixed original model identity",
                "Named run, horizon and original fold",
                horizon=horizon,
                fold=fold,
            )
        analysis = run / f"assessment/h{horizon}/fold{fold}"
        _inventory(analysis, {"metadata.json": state["windows"][str(horizon)][str(fold)]})
        window = _read(analysis / "metadata.json")
        _inventory(analysis, window["files"])
        sample = pd.read_parquet(analysis / "sample.parquet")
        date = sample.date.min()
        source = context.store.load(Tables.cube, columns=list(sample.columns), since=date, until=date, optional=False)
        if source is None:
            raise ValueError("Missing source diagnostic sample")
        categories = set(name for names in cutoff["categorical_cols_by_model_h"][str(horizon)].values() for name in names)
        for column in set(sample.columns) - {"date", "ticker"}:
            numeric = pd.to_numeric(source[column], errors="raise")
            source[column] = numeric.fillna(-1).astype("int32") if column in categories else numeric.astype("float32") if fold else numeric
        paired = sample.loc[sample.date == date].merge(
            source, on=["date", "ticker"], how="left", suffixes=("", "_source"), validate="one_to_one", indicator=True
        )
        columns = list(set(sample.columns) - {"date", "ticker"})
        _finding(
            findings,
            (paired["_merge"] == "both").all()
            and np.allclose(paired[columns], paired[[name + "_source" for name in columns]], equal_nan=True, atol=0, rtol=0),
            "Prepared explanation inputs differ from source",
            "Independent bounded date/key/category/precision replay",
            horizon=horizon,
            fold=fold,
        )
        oof = pd.read_parquet(fitted / "oof.parquet") if fold else None
        if oof is not None:
            day = oof.date.min()
            oof = oof.loc[oof.date == day].sort_values(["date", "ticker"])
            replay = context.store.load(Tables.cube, columns=list(sample.columns), since=day, until=day, optional=False)
            if replay is None:
                raise ValueError("Missing original OOF replay inputs")
            for column in set(sample.columns) - {"date", "ticker"}:
                numeric = pd.to_numeric(replay[column], errors="raise")
                replay[column] = numeric.fillna(-1).astype("int32") if column in categories else numeric.astype("float32")
            replay = oof[["date", "ticker"]].merge(replay, on=["date", "ticker"], how="left", validate="one_to_one")
        else:
            replay = None
        zs = []
        for name in training["model_types"]:
            path = fitted / f"model_h{horizon}_{name}.pkl"
            hashes[name].append(inventory[path.name]["sha256"])
            with path.open("rb") as handle:
                member = pickle.load(handle)
            family_key = {"elasticnet": "linear", "ridge": "linear", "lgbm": "lgbm", "random_forest": "rf"}[name]
            declared_categories = cutoff["categorical_cols_by_model_h"][str(horizon)][name]
            declared_features = list(dict.fromkeys(cutoff[f"{family_key}_cols_by_h"][str(horizon)] + declared_categories))
            _finding(
                findings,
                member.features == declared_features and member.categoricals == declared_categories,
                "Actual fitted pool differs from resolved configuration",
                "Exact configured retained numeric/category order",
                horizon=horizon,
                fold=fold,
                family=name,
            )
            _finding(
                findings,
                member.horizon == horizon and member.family == name and member.task == "regression",
                "Wrong fitted member identity",
                "Declared family/horizon regression member",
                horizon=horizon,
                fold=fold,
            )
            explained += _effect_checks(analysis / name, member, config, fold == 0, findings)
            if replay is not None and oof is not None:
                raw = _raw_prediction(member, replay[member.features].to_numpy(dtype=float))
                std = float(np.std(raw, ddof=1))
                z = (raw - np.mean(raw)) / std if np.isfinite(std) and std > 0 else np.full(len(raw), np.nan)
                _finding(
                    findings,
                    np.allclose(z, oof[name], equal_nan=True, atol=1e-8),
                    "Original OOF member fails reloaded native prediction",
                    "Exact saved original model on its original labelled cross-section",
                    horizon=horizon,
                    fold=fold,
                    family=name,
                )
                zs.append(z)
            del member
        if oof is not None:
            values = np.column_stack(zs)
            count = np.isfinite(values).sum(axis=1)
            average = np.nansum(values, axis=1) / np.where(count > 0, count, 1)
            average[count == 0] = np.nan
            _finding(
                findings,
                np.allclose(average, oof.pred, equal_nan=True, atol=1e-8),
                "Original OOF ensemble differs",
                "Independent mean of standardized original members",
                horizon=horizon,
                fold=fold,
            )
        del sample, source
        gc.collect()
    for name, values in hashes.items():
        _finding(
            findings,
            len(set(values)) == len(values),
            "Repeated fold/cutoff model bytes",
            "Distinct original fitted objects, not copies of the cutoff model",
            horizon=horizon,
            family=name,
        )
    return explained


def _full_history_checks(context: Context, run: Path, training: dict, cutoff: dict, config: DictConfig, findings: list[Finding]) -> dict[str, Any]:
    stage = training.get("full_history", {})
    if stage.get("complete") is not True:
        _finding(findings, False, "Full-history refit/publication not complete", "Final named-run refit and isolated publication completed")
        return {}
    snapshot = (run / "full_history" / stage["attempt_id"]).resolve()
    if not snapshot.is_relative_to(run.resolve()):
        raise ValueError("Full-history attempt path escapes the run")
    _inventory(snapshot, {"metadata.json": stage["metadata_file"], "kpis.json": stage["kpis_file"]})
    meta = _read(snapshot / "metadata.json")
    _inventory(snapshot, meta["member_files"])
    root = Path(str(config.model.models_dir))
    root = (root if root.is_absolute() else Path(context.paths["DATA_STORE"]) / root).resolve()
    if stage["published_root"] != str(root):
        raise ValueError("Full-history publication root differs from configured model output")
    _inventory(root, {str(Path(stage["published_directory"]) / name): identity for name, identity in meta["member_files"].items()})
    _finding(
        findings,
        meta["run_id"] == training["run_id"]
        and meta["full_history"] is True
        and meta["model_types"] == training["model_types"]
        and meta["horizons"] == training["horizons"],
        "Full-history identity differs",
        "Same named run/families/horizons and full-history role",
    )
    calendar = pd.DatetimeIndex(context.store.distinct(Tables.cube_part_prices, "date")).normalize().sort_values().drop_duplicates()
    calendar = calendar[calendar <= pd.Timestamp(meta["training_as_of"])]
    coverage = {}
    for horizon in training["horizons"]:
        target = f"target_{training['target_type']}_h{horizon}"
        ends = _calendar_ends(calendar, horizon)
        count, last = 0, pd.Timestamp.min
        for chunk in context.store.iter_load(
            Tables.cube,
            columns=["date", "ticker", target],
            where={target: context.store.NOT_NULL},
            since=meta["train_start"],
            until=meta["training_as_of"],
            chunksize=int(config.model.assessment.read_chunksize),
        ):
            endpoint = chunk.date.map(ends)
            valid = np.isfinite(chunk[target]) & endpoint.notna()
            count += int(valid.sum())
            if valid.any():
                last = max(last, chunk.loc[valid, "date"].max())
        audit = meta["fit_audit"][str(horizon)]
        counts = cutoff["selection"][str(horizon)]["rounds"]
        valid_counts = meta["selection"][str(horizon)]["rounds"] == counts and (
            audit["requested_rounds"] == int(np.median(counts)) if counts else audit["requested_rounds"] is None
        )
        _finding(
            findings,
            valid_counts and count == audit["fitted_rows"] and str(last.date()) == audit["feature_max"],
            "Full-history coverage/count selection differs",
            "All latest finite mature labels and the named cutoff's saved fold counts, without repeated CV",
            horizon=horizon,
            rows=count,
        )
        coverage[str(horizon)] = {"rows": count, "last_feature_date": str(last.date()), "rounds_reused": valid_counts}
    return coverage


def _report_checks(run: Path, training: dict, state: dict, findings: list[Finding]) -> None:
    _inventory(run, {"report.html": state["report_file"]})
    text = (run / "report.html").read_text(encoding="utf-8")
    match = re.search(r'<script id="assessment-data" type="application/json">(.*?)</script>', text, re.S)
    if match is None:
        raise ValueError("Missing report payload")
    payload = json.loads(match.group(1))
    _finding(
        findings,
        payload["run_id"] == training["run_id"]
        and payload["horizons"] == training["horizons"]
        and payload["families"] == training["model_types"]
        and text.count("<section ") == 2,
        "Report run/chart scope differs",
        "Named run with every configured horizon/family and exactly two sections",
    )
    for horizon in training["horizons"]:
        _temporal_payload_checks(run, horizon, training["model_types"], payload["analysis"][str(horizon)]["temporal"], findings)
        effects = payload["analysis"][str(horizon)]["effects"]
        for family in training["model_types"]:
            _finding(
                findings,
                set(effects[family]) == {"0", *training["folds"][str(horizon)]},
                "Report dropped an original-fold curve",
                "All five original fold curves and cutoff curve",
                horizon=horizon,
                family=family,
            )
            for fold in effects[family]:
                expected = _read(run / f"assessment/h{horizon}/fold{fold}/{family}/curves.json")
                for values in expected["features"].values():
                    if fold == "0":
                        values["scatter"] = {key: values["scatter"][key] for key in ("x", "shap", "raw_x") if key in values["scatter"]}
                    else:
                        values.pop("scatter", None)
                _finding(
                    findings,
                    expected == effects[family][fold],
                    "HTML variable effects differ from saved evidence",
                    "Unmodified retained-variable curves/importance and displayed sample",
                    horizon=horizon,
                    family=family,
                    fold=fold,
                )
    manifest = _read(run / "assessment/report_manifest.json")
    expected_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    _finding(
        findings,
        manifest["report_sha256"] == _file_identity(run / "report.html")["sha256"] and manifest["payload_sha256"] == expected_hash,
        "HTML/JSON report hashes differ",
        "Independent report identity",
    )


def _temporal_payload_checks(run: Path, horizon: int, families: list[str], temporal: dict, findings: list[Finding]) -> None:
    daily = pd.read_parquet(run / f"assessment/h{horizon}/daily.parquet")
    if set(temporal) != {"validation", "backtest"} or any(set(series) != {*families, "ensemble"} for series in temporal.values()):
        raise ValueError("HTML dropped a temporal population/model")
    for population, series in temporal.items():
        for model, values in series.items():
            rows = daily.loc[(daily.population == population) & (daily.model == model)].sort_values("date")
            dates_match = np.array_equal(pd.to_datetime(values["date"]).to_numpy(dtype="datetime64[ns]"), rows.date.to_numpy(dtype="datetime64[ns]"))
            numbers_match = all(
                np.allclose(np.asarray(values[name], dtype=float), rows[name], equal_nan=True)
                for name in ("ic", "rolling_ic", "rolling_ir", "paired", "universe")
            )
            _finding(
                findings,
                dates_match and numbers_match,
                "HTML temporal series differs from saved evidence",
                "Every saved date/statistic/support value",
                horizon=horizon,
                population=population,
                model=model,
            )


def check_modelling(context: Context, *, out: str | Path, config: DictConfig) -> CheckResult:
    run = Path(out).resolve()
    required = [run / name for name in ("metadata.json", "cutoff/metadata.json", "assessment/metadata.json", "report.html")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return CheckResult.abstained("modelling", "saved_ensemble", "Saved run assessment/report is incomplete", missing=missing)
    findings: list[Finding] = []
    metrics: dict[str, Any] = {}
    rows, explained = 0, 0
    try:
        training, cutoff, state = _read(required[0]), _read(required[1]), _read(required[2])
        if training.get("training_complete") is not True or state.get("complete") is not True:
            return CheckResult.abstained("modelling", "saved_ensemble", "Training or assessment stage is incomplete")
        if training["run_id"] != run.name or cutoff["run_id"] != run.name or state["run_id"] != run.name:
            raise ValueError("Wrong-run model/assessment identity")
        _inventory(run, {"cutoff/metadata.json": training["cutoff_metadata"], "training/calendar.parquet": training["calendar_file"]})
        _reconcile_configuration(config, training, cutoff, findings)
        if findings:
            return CheckResult.measured("modelling", "saved_ensemble", findings, scope={"run": str(run), "source_reads": 0}, metrics={})
        source_root = Path(__file__).resolve().parents[3]
        for name, digest in {
            **training["source_hashes"],
            **{f"src/modelling/{name}": value for name, value in state["identity"]["source_hashes"].items()},
        }.items():
            source = (source_root / name).resolve()
            if not source.is_relative_to(source_root) or _file_identity(source)["sha256"] != digest:
                raise ValueError("Saved implementation source identity changed")
        if dict(state["identity"]["diagnostics"]) != {key: config.model.diagnostics[key] for key in state["identity"]["diagnostics"]} or state[
            "identity"
        ]["evaluation_start"] != str(config.model.assessment.evaluation_start):
            raise ValueError("Diagnostic/evaluation configuration differs")
        calendar = pd.DatetimeIndex(pd.read_parquet(run / "training/calendar.parquet").date).as_unit("ns")
        for horizon in training["horizons"]:
            output = run / f"assessment/h{horizon}"
            _inventory(output, state["horizons"][str(horizon)]["files"])
            models = [*training["model_types"], "ensemble"]
            metrics[str(horizon)] = _reconcile_metrics(output, horizon, models, int(config.model.diagnostics.rolling_window), findings)
            rows += _source_horizon(context, run, training, cutoff, horizon, calendar, int(config.model.assessment.read_chunksize), findings)
            for fold, identity in training["folds"][str(horizon)].items():
                _inventory(run / f"folds/h{horizon}/fold{fold}", {"metadata.json": identity})
            explained += _model_checks(context, run, training, cutoff, horizon, state, config, findings)
            context.log.info("Independent h%s reconciliation: metrics, source targets, original models and effects complete", horizon)
        metrics["full_history"] = _full_history_checks(context, run, training, cutoff, config, findings)
        _report_checks(run, training, state, findings)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AssertionError, pickle.UnpicklingError) as exc:
        findings.append(
            Finding.at(8, f"{type(exc).__name__}: {exc}", "Complete, matching saved-run artifacts and independently reproducible evidence")
        )
    return CheckResult.measured(
        "modelling",
        "saved_ensemble",
        findings,
        scope={
            "run": str(run),
            "rows": rows,
            "explained_rows": explained,
            "no_fitting": True,
            "source_scope": "All saved target rows; one complete feature cross-section per fold/cutoff and original OOF first date; configured-grid native PDP spot checks",
        },
        metrics=metrics,
        reason="Conditional historical assessment: retrospective classifications, peer vintages and current-universe bias remain; no performance threshold or generalization guarantee.",
    )
