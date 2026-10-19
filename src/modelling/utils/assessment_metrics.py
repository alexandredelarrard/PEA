"""Finite-pair daily IC and fully supported observed-session rolling IC/IC-IR."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.modelling.utils.assessment import summarize_ic
from src.modelling.utils.metrics import daily_ic_series


def temporal_diagnostics(predictions: pd.DataFrame, models: list[str], horizon: int, window: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Retain all observed sessions; rolling windows require a full set of finite daily ICs."""
    stats, frames = {}, []
    if predictions.duplicated(["date", "ticker"]).any() or window < 1:
        raise ValueError("Temporal diagnostics require unique prediction keys and a positive window")
    sessions = predictions.groupby("date").agg(fold=("fold", "first"), universe=("ticker", "size"))
    for model in models:
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
        years = {str(year): summarize_ic(group.ic, horizon) for year, group in daily.groupby(pd.DatetimeIndex(daily.index).year)}
        folds = {str(fold): summarize_ic(group.ic, horizon) for fold, group in daily.groupby("fold")}
        worst = daily.rolling_ic.dropna().idxmin() if daily.rolling_ic.notna().any() else None
        weakest = daily.ic.loc[:worst].tail(window) if worst is not None else daily.ic.iloc[:0]
        stats[model] = {
            **summarize_ic(series, horizon),
            "years": years,
            "folds": folds,
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
        frames.append(daily.rename_axis("date").reset_index())
    return stats, pd.concat(frames, ignore_index=True)
