"""Pin one assessed cutoff ensemble and score full daily feature cross-sections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_store.schema import Tables
from src.modelling.utils.artifacts import load_ensemble, read_metadata
from src.modelling.utils.ensemble import blend_horizons, ensemble_predict, ir_horizon_weights
from src.modelling.utils.metrics import per_day_zscore
from src.utils.modelling_runs import select_run
from src.utils.universe import load_universe_tickers


def select_cutoff(context: Context, config: DictConfig, run_id: str | None) -> tuple[Path, dict[str, Any]]:
    """Validate identity and all observed outcome dates before loading any pickle."""
    run, state = select_run(context, config, run_id, assessed=True)
    meta = read_metadata(run / "cutoff")
    if meta.get("run_id") != state["run_id"] or meta.get("full_history") is not False or meta.get("role") != "cutoff":
        raise ValueError("Portfolio backtest requires the named cutoff models")
    if not meta.get("member_files"):
        raise ValueError("Missing cutoff model member inventory")
    for key in ("horizons", "model_types", "feature_cols", "categorical_cols", "target_type", "label_column"):
        if key not in meta or meta[key] != state[key]:
            raise ValueError(f"Cutoff/run {key} mismatch")
    cutoff = pd.Timestamp(config.portfolio.training_cutoff)
    if pd.isna(cutoff) or cutoff >= pd.Timestamp(config.portfolio.start):
        raise ValueError("Training cutoff must precede the backtest")

    def check_date(record: dict[str, Any], key: str) -> None:
        value = record.get(key)
        if value is None:
            raise ValueError(f"Cutoff {key} missing or later than {cutoff.date()}")
        date = pd.Timestamp(value)
        if pd.isna(date) or date > cutoff:
            raise ValueError(f"Cutoff {key} missing or later than {cutoff.date()}")

    for key in ("training_as_of", "selection_as_of", "train_end"):
        check_date(meta, key)
    if state.get("training_as_of") != meta["training_as_of"]:
        raise ValueError("Run/cutoff observation frontier mismatch")
    for horizon in meta["horizons"]:
        audit = meta.get("fit_audit", {}).get(str(horizon), {})
        for key in ("training_as_of", "feature_max", "label_end_max"):
            check_date(audit, key)
        folds = meta.get("selection", {}).get(str(horizon), {}).get("folds", [])
        if len(folds) != len(state["folds"][str(horizon)]):
            raise ValueError("Incomplete cutoff selection audit")
        for fold in folds:
            for key in ("train_end", "train_label_end_max", "test_end", "test_label_end_max"):
                check_date(fold, key)
            if pd.Timestamp(fold["train_label_end_max"]) >= pd.Timestamp(fold["test_start"]):
                raise ValueError("Training labels overlap selection test")
    return run, meta


def score_panel(panel: pd.DataFrame, models: dict[int, dict[str, Any]], metadata: dict[str, Any]) -> pd.DataFrame:
    """No labels or future-period correlations participate in horizon weights."""
    horizons = list(models)
    training_ir = metadata.get("train_ic_ir", {})
    weights = ir_horizon_weights({h: float(training_ir.get(str(h), training_ir.get(h, np.nan))) for h in horizons})
    signals = []
    for horizon in horizons:
        score, _ = ensemble_predict(models[horizon], panel)
        signals.append(per_day_zscore(score.to_numpy(), panel.date.to_numpy()))
    result = panel[["date", "ticker"]].copy()
    result["score"] = blend_horizons(np.column_stack(signals), np.array([weights[h] for h in horizons]))
    return result


def load_scores(context: Context, config: DictConfig, directory: Path, metadata: dict[str, Any], end: pd.Timestamp) -> pd.DataFrame:
    """Stream one bounded month at a time, then normalize complete daily cross-sections."""
    models = load_ensemble(directory, metadata["horizons"], metadata["model_types"], metadata=metadata)
    columns = list(dict.fromkeys(["date", "ticker", *metadata["feature_cols"], *metadata["categorical_cols"]]))
    missing = set(columns) - set(context.store.columns(Tables.cube))
    if missing:
        raise ValueError(f"Missing model input columns: {sorted(missing)}")
    names = load_universe_tickers(context)
    start = pd.Timestamp(config.portfolio.start)
    rows = []
    for month in pd.period_range(start, end, freq="M"):
        chunks = list(
            context.store.iter_load(
                Tables.cube,
                columns=columns,
                where={"ticker": names},
                since=max(start, month.start_time),
                until=min(end, month.end_time),
                chunksize=int(config.portfolio.read_chunksize),
            )
        )
        if not chunks:
            continue
        panel = pd.concat(chunks, ignore_index=True)
        panel["date"] = pd.to_datetime(panel.date)
        if panel.duplicated(["date", "ticker"]).any():
            raise ValueError("Duplicate feature grain (date,ticker)")
        rows.append(score_panel(panel, models, metadata))
    if not rows:
        raise ValueError("No feature observations in the backtest interval")
    return pd.concat(rows).pivot(index="date", columns="ticker", values="score").sort_index()
