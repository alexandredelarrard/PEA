"""
trend_signal.py  (src/modelling/utils_trend_asset/trend_signal.py)
------------------------------------------------------------------
Pure building blocks for the multi-asset TIME-SERIES-MOMENTUM (trend-following / CTA) sleeve.

Time-series momentum is the OPPOSITE of the equity book's cross-sectional alpha: each asset is
compared to its OWN past (not to peers), sized long if it has been trending up and short if down,
across whole asset classes (equity index, commodities, FX, ...). It is DIRECTIONAL (net long/short)
and — crucially — has POSITIVE skew / "crisis alpha": it tends to profit in sustained sell-offs
(2008, 2022) when a long-biased equity book loses, so it diversifies by PAYOFF PROFILE.

Method (Carver-style combined forecast, price-only — NO macro inputs):
  1. per asset, per lookback L: standardized trend = (P_t / P_{t-L} - 1) / (daily_vol * sqrt(L))
     -- a Sharpe-like trend strength, comparable across assets and horizons;
  2. combine lookbacks (mean) and CAP to [-cap, +cap] to bound leverage on runaway moves;
  3. vol-scale to a per-asset ex-ante vol target: weight_i = forecast_i * vol_target / ann_vol_i;
  4. hold between rebalances; realized sleeve return nets turnover cost.
All functions are point-in-time (a signal at t uses only prices up to t; it earns the t->t+1 return).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The pure trend building blocks live in the SHARED util so the post-processing CTA
# strategy + allocation backtest reuse them without cross-importing this modelling package.
# Re-exported here so existing modelling imports keep working unchanged.
from src.utils.trend import (                                       # noqa: F401
    daily_vol, combined_forecast, vol_scaled_positions, sleeve_returns,
    rebalanced as _rebalanced,
)

_ANN: float = 252.0


def value_forecast(close: pd.DataFrame, lookback: int = 1260, vol_window: int = 63,
                   cap: float = 2.0) -> pd.DataFrame:
    """Cross-asset VALUE = long-horizon mean reversion: cheap vs its own multi-year path -> long.
    It is the NEGATIVE of a long-lookback trend, vol-normalized exactly like `combined_forecast`
    so the two combine directly. Being long-horizon reversal, it is NEGATIVELY correlated with the
    3-12m trend -> it diversifies the whipsaw that hurt trend-alone in a choppy regime. date x asset."""
    dvol = daily_vol(close, vol_window)
    long_ret = close.pct_change(lookback, fill_method=None)          # P_t/P_{t-L} - 1 over ~5y
    return (-long_ret / (dvol * np.sqrt(float(lookback)))).clip(lower=-cap, upper=cap)


def carry_forecast(carry: pd.DataFrame, close: pd.DataFrame, vol_window: int = 63,
                   cap: float = 2.0) -> pd.DataFrame:
    """Standardize a per-asset annualized CARRY (date x asset — e.g. bond curve slope 10y-3m, FX
    short-rate differential, commodity roll) into a comparable capped forecast: carry / annual vol
    (a carry-to-risk ratio). Positive carry -> long. Assets/dates with no carry stay NaN (then the
    blend just uses trend+value for them). Reads external rates (FRED `macro` in production)."""
    ann_vol = daily_vol(close, vol_window) * np.sqrt(_ANN)
    c = carry.reindex(index=close.index).ffill().reindex(columns=close.columns)
    return (c / ann_vol).clip(lower=-cap, upper=cap)


def combine_signals(forecasts: dict[str, pd.DataFrame], weights: dict[str, float] | None = None,
                    cap: float = 2.0) -> pd.DataFrame:
    """NaN-aware weighted average of several capped forecasts (trend / value / carry), per cell.
    An asset missing a signal (NaN) falls back to the signals it does have. Re-capped. date x asset."""
    names = list(forecasts)
    if not names:
        raise ValueError("combine_signals: no forecasts given")
    w = {n: float((weights or {}).get(n, 1.0)) for n in names}
    base = forecasts[names[0]]
    stack = np.stack([forecasts[n].reindex_like(base).to_numpy(float) for n in names])
    wv = np.array([w[n] for n in names]).reshape(-1, 1, 1)
    mask = np.isfinite(stack)
    num = np.nansum(np.where(mask, stack * wv, 0.0), axis=0)
    den = np.nansum(np.where(mask, wv, 0.0), axis=0)
    out = np.where(den > 0, num / den, np.nan)
    return pd.DataFrame(out, index=base.index, columns=base.columns).clip(lower=-cap, upper=cap)


def apply_class_budget(weights: pd.DataFrame, asset_class: dict[str, str],
                       class_budgets: dict[str, float] | None = None) -> pd.DataFrame:
    """Risk-budget the per-asset trend weights ACROSS asset classes: each class carries its target
    budget (default = equal risk per class), split equally across the instruments within the class.
    Stops a class with many instruments (e.g. 3 commodities) from dominating one with few (1 equity
    index) — the sleeve then times equity / bonds / commodities / FX on an equal footing. date x asset."""
    groups: dict[str, list[str]] = {}
    for a in weights.columns:
        groups.setdefault(str(asset_class.get(a, "other")), []).append(a)
    if not groups:
        return weights
    default_b = 1.0 / len(groups)
    budgets = dict(class_budgets) if class_budgets else {}
    out = weights.copy()
    for cls, assets in groups.items():
        out[assets] = weights[assets] * (float(budgets.get(cls, default_b)) / len(assets))
    return out


def realized_ann_vol(ret: pd.Series) -> float:
    """Annualized realized volatility of a daily return series."""
    r = ret.dropna()
    return float(r.std() * np.sqrt(_ANN)) if len(r) > 2 else 0.0


def vol_target_scalar(ret: pd.Series, target_ann_vol: float) -> float:
    """Calibration scalar so a daily return series hits `target_ann_vol` (>=0; 1.0 if degenerate)."""
    v = realized_ann_vol(ret)
    return float(target_ann_vol / v) if v > 0 else 1.0
