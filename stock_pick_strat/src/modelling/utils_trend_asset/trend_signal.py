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

_ANN: float = 252.0


def daily_vol(close: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """Trailing daily-return volatility per asset (date x asset). NaN-tolerant."""
    mp = int(min_periods) if min_periods is not None else max(10, window // 2)
    return close.pct_change(fill_method=None).rolling(window, min_periods=mp).std()


def combined_forecast(close: pd.DataFrame, lookbacks: list[int], vol_window: int,
                      cap: float = 2.0) -> pd.DataFrame:
    """Capped multi-lookback vol-normalized trend forecast (date x asset), in [-cap, +cap].
    Positive = uptrend (go long), negative = downtrend (go short)."""
    if not lookbacks:
        raise ValueError("combined_forecast: `lookbacks` must be non-empty")
    dvol = daily_vol(close, vol_window)
    parts: list[pd.DataFrame] = []
    for lb in lookbacks:
        mom = close.pct_change(lb, fill_method=None)                 # P_t/P_{t-lb} - 1
        parts.append(mom / (dvol * np.sqrt(float(lb))))              # standardized trend strength
    forecast = sum(parts) / float(len(parts))
    return forecast.clip(lower=-cap, upper=cap)


def vol_scaled_positions(forecast: pd.DataFrame, close: pd.DataFrame, vol_window: int,
                         per_asset_vol_target: float) -> pd.DataFrame:
    """Turn forecasts into weights sized so each asset's ex-ante annual vol ~
    forecast * per_asset_vol_target: weight_i = forecast_i * vol_target / ann_vol_i (date x asset)."""
    ann_vol = daily_vol(close, vol_window) * np.sqrt(_ANN)
    ann_vol = ann_vol.where(np.isfinite(ann_vol) & (ann_vol > 0))
    w = forecast * (per_asset_vol_target / ann_vol)
    return w.replace([np.inf, -np.inf], np.nan)


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


def _rebalanced(weights: pd.DataFrame, rebalance_freq: int) -> pd.DataFrame:
    """Update target weights only every `rebalance_freq` rows; hold (ffill) in between."""
    if rebalance_freq <= 1:
        return weights
    mask = np.zeros(len(weights), dtype=bool)
    mask[::rebalance_freq] = True
    held = weights.where(pd.Series(mask, index=weights.index), other=np.nan)
    return held.ffill()


def sleeve_returns(weights: pd.DataFrame, close: pd.DataFrame, fee_bps: float = 1.0,
                   spread_bps: float = 5.0, rebalance_freq: int = 5) -> pd.DataFrame:
    """Daily NET return of the trend sleeve from target weights (held between rebalances).
    Point-in-time: weights decided at t-1 earn the t-1->t return; turnover at t costs (fee+spread).
    Returns date-indexed DataFrame [ret, gross, turnover]."""
    cost_rate = (fee_bps + spread_bps) / 1e4
    r = close.pct_change(fill_method=None)
    w = _rebalanced(weights, rebalance_freq).reindex(close.index).fillna(0.0)
    w_held = w.shift(1)                                              # held into today's return
    gross_ret = (w_held * r).sum(axis=1, skipna=True)
    turnover = (w - w.shift(1)).abs().sum(axis=1, skipna=True)
    net_ret = gross_ret - turnover * cost_rate
    out = pd.DataFrame({"ret": net_ret, "gross": w.abs().sum(axis=1), "turnover": turnover})
    return out.iloc[1:]                                             # drop the first all-NaN row


def realized_ann_vol(ret: pd.Series) -> float:
    """Annualized realized volatility of a daily return series."""
    r = ret.dropna()
    return float(r.std() * np.sqrt(_ANN)) if len(r) > 2 else 0.0


def vol_target_scalar(ret: pd.Series, target_ann_vol: float) -> float:
    """Calibration scalar so a daily return series hits `target_ann_vol` (>=0; 1.0 if degenerate)."""
    v = realized_ann_vol(ret)
    return float(target_ann_vol / v) if v > 0 else 1.0
