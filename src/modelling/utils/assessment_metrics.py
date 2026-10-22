"""Finite-pair daily IC and fully supported observed-session rolling IC/IC-IR."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.modelling.utils.metrics import daily_ic_series, summarize_ic


def _temporal_model(predictions: pd.DataFrame, sessions: pd.DataFrame, model: str, horizon: int, window: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Use original sessions even when a selected cohort has no comparable predictions."""
    target = predictions.loc[np.isfinite(predictions.y) & np.isfinite(predictions[model])]
    scored = target[["date", "ticker", "y", model]].rename(columns={model: "pred"})
    series = daily_ic_series(scored, "y")
    daily = series.rename("ic").reindex(sessions.index).to_frame().join(sessions)
    daily["paired"] = target.groupby("date").size().reindex(daily.index, fill_value=0)
    daily["prediction_unique"] = target.groupby("date")[model].nunique().reindex(daily.index, fill_value=0)
    daily["target_unique"] = target.groupby("date").y.nunique().reindex(daily.index, fill_value=0)
    daily["rolling_ic"] = daily.ic.rolling(window, min_periods=window).mean()
    dispersion = daily.ic.rolling(window, min_periods=window).std(ddof=0)
    daily["rolling_ir"] = daily.rolling_ic / dispersion.where(dispersion > 0) * np.sqrt(252 / horizon)
    worst = daily.rolling_ic.dropna().idxmin() if daily.rolling_ic.notna().any() else None
    weakest = daily.ic.loc[:worst].tail(window) if worst is not None else daily.ic.iloc[:0]
    stats = {
        **summarize_ic(series, horizon),
        "years": {str(year): summarize_ic(group.ic, horizon) for year, group in daily.groupby(pd.DatetimeIndex(daily.index).year)},
        "folds": {str(fold): summarize_ic(group.ic, horizon) for fold, group in daily.groupby("fold")},
        "worst_rolling_end": worst,
        "worst_rolling_start": weakest.index[0] if worst is not None else None,
        "worst_rolling_mean_ic": daily.at[worst, "rolling_ic"] if worst is not None else None,
        "worst_rolling_ic_ir": daily.at[worst, "rolling_ir"] if worst is not None else None,
        "worst_rolling_n_days": len(weakest),
        "rolling_window": window,
        "label_rows": len(target),
        "scored_rows": len(predictions),
        "excluded_label_rows": len(predictions) - len(target),
    }
    daily["model"] = model
    return stats, daily.rename_axis("date").reset_index()


def temporal_diagnostics(predictions: pd.DataFrame, models: list[str], horizon: int, window: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Retain all observed sessions; rolling windows require a full set of finite daily ICs."""
    if predictions.duplicated(["date", "ticker"]).any() or window < 1:
        raise ValueError("Temporal diagnostics require unique prediction keys and a positive window")
    sessions = predictions.groupby("date").agg(fold=("fold", "first"), universe=("ticker", "size"))
    stats, frames = {}, []
    for model in models:
        stats[model], daily = _temporal_model(predictions, sessions, model, horizon, window)
        frames.append(daily)
    return stats, pd.concat(frames, ignore_index=True)


def tail_universes(predictions: pd.DataFrame, model: str, sizes: tuple[int, ...]) -> dict[int, pd.DataFrame]:
    """Sort each day's finite predictions once; ticker breaks ties before either tail is taken."""
    ordered = predictions.loc[np.isfinite(predictions[model])].sort_values(["date", model, "ticker"])
    groups = ordered.groupby("date")
    position = groups.cumcount()
    count = groups[model].transform("size")
    return {size: ordered.loc[(position < size) | (position >= count - size)] for size in sizes}


def universe_diagnostics(
    predictions: pd.DataFrame, models: list[str], horizon: int, window: int, sizes: tuple[int, ...]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate ex-ante tail IC using the same arithmetic and rolling support as All."""
    sessions = predictions.groupby("date").agg(fold=("fold", "first"))
    stats: dict[str, Any] = {str(size): {} for size in sizes}
    frames = []
    for model in models:
        for size, selected in tail_universes(predictions, model, sizes).items():
            cohort = sessions.assign(universe=selected.groupby("date").size().reindex(sessions.index, fill_value=0))
            stats[str(size)][model], daily = _temporal_model(selected, cohort, model, horizon, window)
            daily["universe_size"] = str(size)
            frames.append(daily)
    return stats, pd.concat(frames, ignore_index=True)


def native_error_diagnostics(predictions: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Retain every sample/ticker's native support independently of ranking eligibility."""
    frames = []
    for model in models:
        raw = predictions[f"raw_{model}"]
        residual = (raw.astype(float) - predictions.y).where(np.isfinite(raw) & np.isfinite(predictions.y))
        errors = predictions[["ticker"]].assign(
            population=np.where(predictions.fold > 0, "validation", "backtest"),
            residual=residual,
            absolute=residual.abs(),
            squared=residual**2,
        )
        grouped = errors.groupby(["population", "ticker"]).agg(
            n=("residual", "count"), mae=("absolute", "mean"), bias=("residual", "mean"), mse=("squared", "mean")
        )
        grouped["rmse"] = np.sqrt(grouped.pop("mse"))
        grouped["model"] = model
        frames.append(grouped.reset_index())
    return pd.concat(frames, ignore_index=True)
