"""Temporal IC uncertainty, ex-ante cohorts and fully supported matched return windows."""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from src.modelling.utils.assessment import summarize_ic
from src.modelling.utils.metrics import daily_ic_series


def formation_cohorts(scores: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Highest predicted ceil(N/10) names, with fixed weights, before outcomes are attached."""
    keys = ["date", "ticker"]
    if scores[keys].isna().any().any() or scores.duplicated(keys).any():
        raise ValueError("Cohort formation requires unique nonnull feature-date keys")
    cohorts = []
    for model in models:
        if not np.isfinite(scores[model].to_numpy(dtype=float)).all():
            raise ValueError(f"{model}: cannot silently exclude an unscored contemporaneous stock")
        ranked = scores[keys + [model]].sort_values(["date", model, "ticker"], ascending=[True, False, True])
        ranked["population"] = ranked.groupby("date")["ticker"].transform("size")
        ranked["selected_count"] = (ranked["population"] + 9) // 10
        ranked["position"] = ranked.groupby("date").cumcount() + 1
        chosen = ranked.loc[ranked.position <= ranked.selected_count].rename(columns={model: "score"}).copy()
        chosen["model"], chosen["weight"] = model, 1 / chosen["selected_count"]
        cohorts.append(chosen)
    return pd.concat(cohorts, ignore_index=True)


def forward_returns(prices: pd.DataFrame, calendar: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """Require every persisted future return and reconcile compounding to total-return prices."""
    if prices.duplicated(["date", "ticker"]).any() or horizon < 1:
        raise ValueError("Return paths require unique keys and a positive horizon")
    calendar = pd.DatetimeIndex(calendar).sort_values().unique()
    frames = []
    for ticker, group in prices.groupby("ticker", sort=False):
        path = group.set_index("date").reindex(calendar)
        close = pd.to_numeric(path["close_total"], errors="raise")
        ret = pd.to_numeric(path["ret"], errors="raise")
        good = pd.Series(np.isfinite(ret.to_numpy()) & (ret.to_numpy() > -1), index=path.index)
        logs = pd.Series(np.log1p(ret.where(good).to_numpy()), index=path.index)
        compound = pd.Series(np.expm1(logs.rolling(horizon, min_periods=horizon).sum().shift(-horizon).to_numpy()), index=path.index)
        endpoint = close.shift(-horizon) / close - 1
        origin_ok = np.isfinite(close) & (close > 0)
        endpoint_ok = np.isfinite(close.shift(-horizon)) & (close.shift(-horizon) > 0)
        complete = good.astype(int).rolling(horizon, min_periods=horizon).sum().shift(-horizon).eq(horizon)
        reconciled = np.isclose(compound, endpoint, rtol=1e-5, atol=1e-6)
        valid = pd.Series(np.asarray(origin_ok & endpoint_ok & complete & reconciled), index=path.index)
        reasons = np.select(
            [~origin_ok, ~endpoint_ok, ~complete, ~reconciled],
            ["missing_origin", "missing_endpoint", "incomplete_path", "endpoint_path_mismatch"],
            default="supported",
        )
        ends = pd.Series(calendar, index=calendar).shift(-horizon)
        frames.append(
            pd.DataFrame(
                {
                    "date": calendar,
                    "ticker": ticker,
                    "label_end": ends.to_numpy(),
                    "valid": valid.to_numpy(),
                    "reason": reasons,
                    "realized_return": compound.where(valid).to_numpy(),
                    "endpoint_return": endpoint.to_numpy(),
                    "future_valid_bars": good.astype(int).rolling(horizon).sum().shift(-horizon).to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def matched_cohort_returns(cohorts: pd.DataFrame, outcomes: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Missing selected outcomes remain missing fixed weights; expose supported contribution separately."""
    attached = cohorts.merge(outcomes[["date", "ticker", "realized_return", "valid"]], on=["date", "ticker"], how="left", validate="many_to_one")
    attached["valid"] = attached["valid"].eq(True) & np.isfinite(attached["realized_return"])
    attached["known_weight"] = attached.weight.where(attached.valid, 0.0)
    attached["known_contribution"] = (attached.weight * attached.realized_return).where(attached.valid, 0.0)
    grouped = attached.groupby(["date", "model"], as_index=False).agg(
        selected=("ticker", "size"),
        supported=("valid", "sum"),
        known_weight=("known_weight", "sum"),
        known_contribution=("known_contribution", "sum"),
    )
    grouped["supported_mean"] = grouped["known_contribution"] / grouped["known_weight"].replace(0, np.nan)
    grouped["cohort_return"] = grouped.known_contribution.where(grouped.selected == grouped.supported)
    grouped = grouped.merge(market, on="date", how="left", validate="many_to_one")
    grouped["market_return"] = grouped.market_return.where(grouped.market_valid.eq(True))
    grouped["excess_return"] = grouped.cohort_return - grouped.market_return
    return grouped


def moving_block_intervals(daily: pd.DataFrame, horizon: int, repetitions: int, confidence: float, seed: int) -> dict[str, Any]:
    """Resample contiguous blocks within each original fold; do not bridge fit boundaries."""
    rng = np.random.default_rng(seed)
    groups = [group.ic.dropna().to_numpy(dtype=float) for _, group in daily.groupby("fold")]
    groups = [group for group in groups if len(group)]
    answer = {}
    for block in [horizon, 2 * horizon]:
        samples = []
        for _ in range(repetitions):
            resampled = []
            for values in groups:
                length = min(block, len(values))
                starts = rng.integers(0, len(values) - length + 1, size=ceil(len(values) / length))
                resampled.append(np.concatenate([values[start : start + length] for start in starts])[: len(values)])
            stats = summarize_ic(pd.Series(np.concatenate(resampled)), horizon) if resampled else {}
            samples.append([stats.get("mean_ic", np.nan), stats.get("ic_ir", np.nan)])
        alpha = (1 - confidence) / 2
        bounds = np.nanquantile(np.asarray(samples), [alpha, 1 - alpha], axis=0) if samples else np.full((2, 2), np.nan)
        answer[str(block)] = {
            "block_sessions": block,
            "repetitions": repetitions,
            "confidence": confidence,
            "mean_ic": bounds[:, 0],
            "ic_ir": bounds[:, 1],
            "policy": "within-fold moving blocks; overlapping h60 outcomes",
        }
    return answer


def temporal_diagnostics(
    predictions: pd.DataFrame, models: list[str], horizon: int, window: int, repetitions: int, confidence: float, seed: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Daily, year, fold and full-window rolling metrics on finite paired labels."""
    stats, frames = {}, []
    target = predictions.loc[np.isfinite(predictions.y)].copy()
    for model in models:
        scored = target[["date", "ticker", "y", model]].rename(columns={model: "pred"})
        series = daily_ic_series(scored, "y")
        daily = series.rename("ic").to_frame().join(predictions.groupby("date").agg(fold=("fold", "first"), universe=("ticker", "size")))
        daily["paired"] = target.groupby("date").size()
        daily["rolling_ic"] = daily.ic.rolling(window, min_periods=window).mean()
        daily["rolling_ir"] = daily.rolling_ic / daily.ic.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252 / horizon)
        years = {str(year): summarize_ic(group.ic, horizon) for year, group in daily.groupby(pd.DatetimeIndex(daily.index).year)}
        folds = {str(fold): summarize_ic(group.ic, horizon) for fold, group in daily.groupby("fold")}
        worst = daily.rolling_ic.dropna().idxmin() if daily.rolling_ic.notna().any() else None
        weakest = daily.ic.loc[:worst].tail(window) if worst is not None else daily.ic.iloc[:0]
        stats[model] = {
            **summarize_ic(series, horizon),
            "years": years,
            "folds": folds,
            "intervals": moving_block_intervals(daily, horizon, repetitions, confidence, seed),
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
        frames.append(daily.reset_index())
    return stats, pd.concat(frames, ignore_index=True)


def trend_summary(x: Any, y: Any) -> dict[str, Any]:
    """Describe measured curve shape with numerical tolerance, without making a causal claim."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2 or len(np.unique(x)) < 2:
        return {"shape": "unsupported", "points": len(x), "rho": None, "span": None}
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = float(np.ptp(y))
    tolerance = max(1e-8, float(np.max(np.abs(y))) * 1e-6)
    if span <= tolerance:
        return {"shape": "flat", "points": len(x), "rho": None, "span": span}
    changes = np.diff(y)
    increasing, decreasing = float(np.mean(changes >= -tolerance)), float(np.mean(changes <= tolerance))
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    shape = "increasing" if increasing >= 0.9 else "decreasing" if decreasing >= 0.9 else "nonmonotonic"
    return {"shape": shape, "points": len(x), "rho": rho, "span": span, "increasing_fraction": increasing, "decreasing_fraction": decreasing}
