"""
ensemble.py  (src/modelling/utils/ensemble.py)
----------------------------------------------
Combining predictions: the per-day-standardised member average (one horizon), the horizon
weights (IR-floored, or correlation-aware Grinold-Kahn), the weighted nan-mean blend across
horizons, and the long-form `predictions_latest` rows. A "model" here is anything with
`predict(panel) -> array-like` aligned to `panel`'s rows, so a stub works in tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from src.modelling.utils.metrics import per_day_zscore

PREDICTION_COLUMNS = ["predicted_at", "date", "ticker", "horizon", "model", "predicts_for", "pred", "rank"]


class Scorer(Protocol):
    def predict(self, panel: pd.DataFrame) -> Any: ...


def ensemble_predict(models: Mapping[str, Scorer], panel: pd.DataFrame) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Average the per-day z-scored predictions of several members into one score per row.

    Each member's raw output is z-scored within each day BEFORE averaging, so members with very
    different output scales (a GBDT and a linear model) share one ranking scale. Members are
    averaged in mapping order with a manual nan-mean, so a row no member can standardise (a
    single-name day) is NaN instead of a 'Mean of empty slice' warning. Returns
    (blended score, {member name: its per-day z})."""
    if not models:
        raise ValueError("ensemble_predict received no models")
    dates = panel["date"].to_numpy()
    members: dict[str, pd.Series] = {}
    zs = []
    for name, model in models.items():
        raw = np.asarray(model.predict(panel))
        z = per_day_zscore(raw, dates)
        zs.append(z)
        members[str(name)] = pd.Series(z, index=panel.index, name=str(name))
    stack = np.column_stack(zs)
    cnt = np.isfinite(stack).sum(axis=1)
    avg = np.where(cnt > 0, np.nansum(stack, axis=1) / np.where(cnt > 0, cnt, 1), np.nan)
    return pd.Series(avg, index=panel.index, name="score"), members


def ir_horizon_weights(ic_ir: Mapping[int, float]) -> dict[int, float]:
    """Blend weight per horizon ∝ max(0, IC_IR); a NaN IR counts as 0; equal weights when no
    horizon has a positive IR. Keys keep the mapping's (horizon) order."""
    irs = {h: max(0.0, ir) for h, ir in ic_ir.items() if np.isfinite(ir)}
    total = sum(irs.values())
    if total <= 0:
        return {h: 1.0 / len(ic_ir) for h in ic_ir}
    return {h: irs.get(h, 0.0) / total for h in ic_ir}


def blend_horizons(z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Row-wise weighted nan-mean of per-horizon z columns: a horizon missing on a row drops out
    and the remaining weights renormalise; NaN where no horizon is present."""
    mask = ~np.isnan(z)
    wsum = np.where(mask, weights, 0.0).sum(axis=1)
    return np.where(wsum > 0, np.nansum(np.where(mask, z * weights, 0.0), axis=1) / np.where(wsum > 0, wsum, 1), np.nan)


def _pairwise_corr(matrix: np.ndarray) -> np.ndarray:
    """Pairwise-complete correlation of the columns of ``matrix`` (rows with a NaN in either
    column are ignored); degenerate pairs (constant / < 3 obs) -> 0."""
    n = matrix.shape[1]
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = matrix[:, i], matrix[:, j]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 2 and a[m].std() > 0 and b[m].std() > 0:
                c = float(np.corrcoef(a[m], b[m])[0, 1])
                corr[i, j] = corr[j, i] = c if np.isfinite(c) else 0.0
    return corr


def optimal_forecast_weights(signals: dict[int, np.ndarray], ir: dict[int, float], shrink: float = 0.5) -> dict[int, float]:
    """Correlation-aware horizon weights (Grinold-Kahn): w ∝ Σ⁻¹ · IR.

    Correlated horizons share weight instead of triple-counting their common component, and a
    weak but diversifying horizon can still earn weight. Σ is shrunk toward the identity by
    `shrink`; an unestimable (NaN) IR takes the mean of the finite IRs as a neutral prior;
    negative weights are floored at 0 and renormalised; degenerate cases fall back to equal."""
    hs = list(signals)
    n = len(hs)
    if n == 0:
        return {}
    if n == 1:
        return {hs[0]: 1.0}

    ir_vec = np.array([ir.get(h, np.nan) for h in hs], dtype=float)
    finite = np.isfinite(ir_vec)
    if not finite.any():
        return {h: 1.0 / n for h in hs}
    ir_vec[~finite] = ir_vec[finite].mean()
    mu = np.clip(ir_vec, 0.0, None)
    if mu.sum() <= 0:
        return {h: 1.0 / n for h in hs}

    matrix = np.column_stack([np.asarray(signals[h], float) for h in hs])
    corr = _pairwise_corr(matrix)
    corr = (1.0 - shrink) * corr + shrink * np.eye(n)
    try:
        w = np.linalg.solve(corr, mu)
    except np.linalg.LinAlgError:
        w = mu.copy()
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w = mu.copy()
    w = w / w.sum()
    return {h: float(w[i]) for i, h in enumerate(hs)}


def predicts_for(as_of: pd.Timestamp | date | str, horizon: float) -> pd.Timestamp:
    """The date a horizon-`h` prediction is ABOUT: `as_of` + h business days (the cube target is
    a forward return over h trading-day rows). Market holidays are ignored, so treat it as the
    target date +/- a few sessions."""
    return pd.Timestamp(as_of) + pd.tseries.offsets.BDay(int(round(float(horizon))))


def prediction_rows(keys: pd.DataFrame, raw: np.ndarray, horizon: float, model: str, predicted_at: pd.Timestamp) -> pd.DataFrame:
    """One (horizon, model) slice of `predictions_latest`: per-day z-scored `pred`, its per-day
    percentile `rank`, stamped with when it was predicted and the date it is about."""
    df = keys.copy()
    df["horizon"] = int(horizon)
    df["model"] = str(model)
    df["predicted_at"] = predicted_at
    df["predicts_for"] = df["date"].map(lambda d: predicts_for(cast(pd.Timestamp, d), horizon))
    df["pred"] = per_day_zscore(np.asarray(raw, dtype="float64"), df["date"].to_numpy())
    df["rank"] = df.groupby("date")["pred"].rank(pct=True)
    return df[PREDICTION_COLUMNS]
