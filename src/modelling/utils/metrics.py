"""
metrics.py  (src/modelling/utils/metrics.py)
--------------------------------------------
Cross-sectional evaluation primitives: daily Spearman IC (summary and series), per-day
z-scoring, max drawdown of a cumulative curve, and rank-based ROC AUC. Every function is
RuntimeWarning-free on degenerate input (constant predictions, single-name days, empty series),
because a strongly regularised member legitimately produces a constant prediction.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def _min_label_values(label: pd.Series) -> int:
    """Distinct label values a day needs to count in the daily IC: 3 for a continuous (rank /
    z-score / return) label, 2 for a binary {0, 1} classification label, whose days always have
    exactly two. Spearman against a binary label is the rank-biserial correlation."""
    return 2 if set(np.unique(label.dropna().to_numpy())) <= {0, 1} else 3


def daily_ic(panel: pd.DataFrame, preds: pd.Series, label_name: str = "y", horizon: int = 1, trading_days_per_year: int = 252) -> dict:
    """Daily cross-sectional IC (Spearman) and its annualized information ratio.

    The label is an `horizon`-day forward return, so consecutive daily ICs overlap; the IR is
    annualised by the number of INDEPENDENT windows per year:
        ic_ir = mean(IC) / std(IC) * sqrt(252 / horizon)
    which reduces to the classic sqrt(252) daily IR at horizon=1."""
    df = panel[["date", label_name]].copy()
    df["pred"] = preds.to_numpy()
    min_label = _min_label_values(df[label_name])
    ics = []
    for _, g in df.groupby("date", sort=True):
        if g["pred"].nunique() > 2 and g[label_name].nunique() >= min_label:
            ic, _ = spearmanr(g["pred"], g[label_name])
            ic_value = cast(float, ic)
            if np.isfinite(ic_value):
                ics.append(ic_value)
    ics = np.asarray(ics, dtype=float)
    n = len(ics)
    periods_per_year = trading_days_per_year / max(1, int(horizon))
    mean_ic = float(ics.mean()) if n else np.nan
    std_ic = float(ics.std()) if n else np.nan
    ic_ir = (mean_ic / std_ic * np.sqrt(periods_per_year)) if (n and std_ic > 0) else np.nan
    return {"mean_ic": mean_ic, "ic_std": std_ic, "ic_ir": ic_ir, "n_days": n}


def daily_ic_series(oos: pd.DataFrame, label_name: str, pred_col: str = "pred") -> pd.Series:
    """Per-day cross-sectional Spearman IC over a (date, pred, label) frame, date-sorted."""
    rows = {}
    min_label = _min_label_values(oos[label_name])
    for d, g in oos.groupby("date", sort=True):
        if g[pred_col].nunique() > 2 and g[label_name].nunique() >= min_label:
            ic, _ = spearmanr(g[pred_col], g[label_name])
            ic_value = cast(float, ic)
            if np.isfinite(ic_value):
                rows[d] = ic_value
    return pd.Series(rows, name="ic").sort_index()


def per_day_zscore(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Cross-sectionally z-score ``values`` within each day (ddof=1).

    Days with < 2 names, or zero / undefined dispersion (e.g. a member whose coefficients were
    all shrunk to zero -> a constant prediction), return NaN."""
    df = pd.DataFrame({"date": np.asarray(dates), "v": np.asarray(values, dtype=float)})

    def _z(s: pd.Series) -> pd.Series:
        if len(s) < 2:
            return pd.Series(np.nan, index=s.index)
        sd = s.std()
        if not np.isfinite(sd) or sd <= 0.0:
            return pd.Series(np.nan, index=s.index)
        return (s - s.mean()) / sd

    return df.groupby("date")["v"].transform(_z).to_numpy()


def max_drawdown(cumulative: pd.Series) -> tuple[float, int]:
    """(depth, length) of the worst peak-to-trough fall of an ADDITIVE cumulative curve (e.g.
    cumulative IC or cumulative spread): depth = min(curve - running peak) <= 0, length = the
    number of observations from that peak to the trough. (0.0, 0) for an empty or monotone curve."""
    curve = cumulative.dropna().to_numpy(dtype=float)
    if curve.size == 0:
        return 0.0, 0
    level = np.concatenate([[0.0], curve])  # an additive curve starts from 0
    peak = np.maximum.accumulate(level)
    drawdown = level - peak
    trough = int(np.argmin(drawdown))
    depth = float(drawdown[trough])
    if depth >= 0.0:
        return 0.0, 0
    since_peak = int(np.argmax(level[trough::-1] == peak[trough]))  # last time the peak was hit
    return depth, since_peak


def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC for a {0,1} label via the Mann-Whitney rank formula (ties count one half);
    NaN when either class is absent."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    keep = np.isfinite(y) & np.isfinite(s)
    y, s = y[keep], s[keep]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
