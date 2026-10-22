"""Render saved multi-horizon IC and variable effects as one self-contained offline report."""

from __future__ import annotations

import base64
import gzip
import hashlib
import html
import json
import tempfile
from itertools import product
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from src.modelling.utils.artifacts import file_identity
from src.modelling.utils.assessment import json_value, write_json
from src.utils.modelling_runs import read_run


def _temporal_payload(output: Path, families: list[str]) -> dict[str, Any]:
    daily = pd.read_parquet(output / "daily.parquet").assign(universe_size="all")
    tails = pd.read_parquet(output / "universe_daily.parquet")
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    tail_metrics = json.loads((output / "universe_metrics.json").read_text(encoding="utf-8"))
    temporal: dict[str, Any] = {population: {} for population in ("validation", "backtest")}
    combined = pd.concat([daily, tails], ignore_index=True)
    expected = {(p, u, m) for p in temporal for u in ("all", "10", "50", "100") for m in [*families, "ensemble"]}
    groups = combined.groupby(["population", "universe_size", "model"])
    if set(groups.groups) != expected:
        raise ValueError("Incomplete temporal chart cohort/model inventory")
    for key, rows in groups:
        population, size, model = cast(tuple[str, str, str], key)
        rows = rows.sort_values("date")
        if rows.date.duplicated().any():
            raise ValueError("Duplicate temporal chart sessions")
        summary = metrics[population][model] if size == "all" else tail_metrics[population][size][model]
        temporal[population].setdefault(size, {})[model] = {
            **{name: rows[name].tolist() for name in ("date", "fold", "ic", "rolling_ic", "rolling_ir", "paired", "universe")},
            "summary": {name: summary[name] for name in ("mean_ic", "ic_std", "ic_ir", "n_days")},
        }
    return temporal


def _error_display(errors: pd.DataFrame) -> pd.DataFrame:
    """Cap saved native metrics from opposite ends of one ascending RMSE/ticker order."""
    ordered = errors.loc[np.isfinite(errors.rmse) & (errors.n > 0)].sort_values(["rmse", "ticker"])
    worst = ordered.iloc[::-1].head(100).assign(group="worst")
    best = ordered.head(50).assign(group="best")
    return pd.concat([worst, best], ignore_index=True).drop_duplicates("ticker").reset_index(drop=True)


def _native_payload(output: Path, families: list[str]) -> dict[str, Any]:
    predictions = pd.read_parquet(output / "predictions.parquet", columns=["date", "ticker", "fold", "y", *(f"raw_{m}" for m in families)])
    errors = pd.read_parquet(output / "ticker_errors.parquet")
    native: dict[str, Any] = {}
    for population in ("validation", "backtest"):
        sample = predictions.loc[(predictions.fold > 0) == (population == "validation")]
        sessions = sample.groupby("date").fold.first()
        native[population] = {}
        for model in families:
            statistics = errors.loc[(errors.population == population) & (errors.model == model)]
            selected = _error_display(statistics)
            columns = [f"raw_{model}", "y"]
            paired = sample.loc[sample.ticker.isin(selected.ticker), ["date", "ticker", *columns]]
            matrix = (
                paired.pivot(index="ticker", columns="date", values=columns)
                .reindex(index=selected.ticker, columns=pd.MultiIndex.from_product([columns, sessions.index]))
                .to_numpy()
                .reshape(len(selected), 2, len(sessions))
            )
            matrix = np.where(np.all(np.isfinite(matrix), axis=1, keepdims=True), matrix, np.nan).astype("<f8")
            native[population][model] = {
                "date": sessions.index.tolist(),
                "fold": sessions.tolist(),
                "eligible_tickers": int((np.isfinite(statistics.rmse) & (statistics.n > 0)).sum()),
                "tickers": selected[["ticker", "n", "mae", "bias", "rmse", "group"]].to_dict("records"),
                "data": base64.b64encode(gzip.compress(matrix.tobytes(), compresslevel=6, mtime=0)).decode("ascii"),
            }
    return native


def _effect_payload(output: Path, families: list[str], windows: dict[str, Any], folds: list[int]) -> dict[str, Any]:
    if set(windows) != {"0", *(str(fold) for fold in folds)}:
        raise ValueError("Report requires all original folds and cutoff effects")
    effects: dict[str, Any] = {name: {} for name in families}
    for fold, family in product(windows, families):
        curves = json.loads((output / f"fold{fold}" / family / "curves.json").read_text(encoding="utf-8"))
        if not curves.get("features"):
            raise ValueError("A required model has no retained variable effects")
        for values in curves["features"].values():
            if fold == "0":
                values["scatter"] = {key: values["scatter"][key] for key in ("x", "shap", "raw_x") if key in values["scatter"]}
            else:
                values.pop("scatter", None)
        effects[family][fold] = curves
    for family, original in effects.items():
        retained = set(original["0"]["features"])
        if any(set(curves["features"]) != retained for curves in original.values()):
            raise ValueError(f"{family}: original-fold effect feature inventory differs")
    return effects


def _payload(run: Path, assessment_directory: Path | None = None) -> dict[str, Any]:
    training = read_run(run)
    assessment_directory = assessment_directory or run / "assessment"
    state = json.loads((assessment_directory / "metadata.json").read_text(encoding="utf-8"))
    if state["run_id"] != training["run_id"]:
        raise ValueError("Report and training run identities differ")
    analysis = {}
    for horizon in training["horizons"]:
        output = assessment_directory / f"h{horizon}"
        families = training["model_types"]
        analysis[str(horizon)] = {
            "temporal": _temporal_payload(output, families),
            "native": _native_payload(output, families),
            "effects": _effect_payload(output, families, state["windows"][str(horizon)], training["folds"][str(horizon)]),
        }
    return json_value(
        {
            "run_id": training["run_id"],
            "horizons": training["horizons"],
            "families": training["model_types"],
            "training_as_of": training["training_as_of"],
            "frontier": state["observation_frontier"],
            "trained_at_utc": training["training_completed_at_utc"],
            "assessed_at_utc": state["assessment_completed_at_utc"],
            "effective_fold_test_ends": {h: evidence["effective_fold_test_end"] for h, evidence in state["horizons"].items()},
            "analysis": analysis,
        }
    )


def _document(payload: dict[str, Any]) -> str:
    """Keep compressed histories outside the initial JSON and decode one selection on demand."""
    analysis, blocks = {}, []
    for horizon, evidence in payload["analysis"].items():
        native: dict[str, Any] = {}
        for population, models in evidence["native"].items():
            native[population] = {}
            for model, values in models.items():
                block = f"native-{horizon}-{population}-{model}"
                native[population][model] = {**{key: value for key, value in values.items() if key != "data"}, "block": block}
                blocks.append(f'<script id="{html.escape(block, quote=True)}" type="application/octet-stream">{values["data"]}</script>')
        analysis[horizon] = {**evidence, "native": native}
    public = {**payload, "analysis": analysis}
    encoded = json.dumps(public, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    return template.replace("__PAYLOAD__", encoded).replace("__NATIVE_BLOCKS__", "".join(blocks)).replace("__PLOTLY__", get_plotlyjs())


def render_assessment(run: Path, *, assessment_directory: Path | None = None, report_path: Path | None = None) -> Path:
    """Read only saved evidence; no database, fitted-object loading or provider is needed."""
    assessment_directory = assessment_directory or run / "assessment"
    payload = _payload(run, assessment_directory)
    document = _document(payload)
    output = report_path or run / "report.html"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=run, prefix=".report.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(document)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    write_json(
        assessment_directory / "report_manifest.json",
        {
            "report_sha256": file_identity(output)["sha256"],
            "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "horizons": payload["horizons"],
            "families": payload["families"],
            "plot_areas": ["temporal", "target_prediction", "importance", "ticker_rmse", "effect"],
        },
    )
    return output
