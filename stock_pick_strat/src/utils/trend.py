"""
trend.py  (src/utils/trend.py)
------------------------------
Pure, reusable time-series-momentum (trend) building blocks, shared by BOTH the
modelling trend sleeve (src/modelling/utils_trend_asset) and the post-processing
multi-asset allocation backtest (src/post_processing/utils/strategies_alloc) — so the
two never cross-import each other's package (a src/ subfolder rule) and there is one
definition of the vol-normalized trend forecast.

  * daily_vol            -- trailing daily-return volatility per asset (date x asset)
  * combined_forecast    -- capped multi-lookback vol-normalized trend, in [-cap, +cap]
  * trend_scale_long_only -- map the forecast to a LONG-ONLY allocation scale in
                             [floor, 1] (linear or binary), for scaling % weights toward
                             cash when an asset rolls over (crisis de-risking).
All functions are point-in-time: a value at t uses only prices up to t.
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
    Positive = uptrend (go long), negative = downtrend (go short). Each lookback L gives a
    Sharpe-like trend strength (P_t/P_{t-L} - 1) / (daily_vol * sqrt(L)); the mean over
    lookbacks is capped to bound leverage on runaway moves."""
    if not lookbacks:
        raise ValueError("combined_forecast: `lookbacks` must be non-empty")
    dvol = daily_vol(close, vol_window)
    parts: list[pd.DataFrame] = []
    for lb in lookbacks:
        mom = close.pct_change(lb, fill_method=None)                 # P_t/P_{t-lb} - 1
        parts.append(mom / (dvol * np.sqrt(float(lb))))              # standardized trend strength
    forecast = sum(parts) / float(len(parts))
    return forecast.clip(lower=-cap, upper=cap)


def trend_scale_long_only(close: pd.DataFrame, lookbacks: list[int], vol_window: int,
                          scheme: str = "linear", floor: float = 0.0, cap: float = 2.0) -> pd.DataFrame:
    """LONG-ONLY trend overlay scale in [floor, 1] per asset per date (date x asset).

    For a % ALLOCATION (not a long/short book) the trend signal should not go short, and it
    should NOT cut a healthy uptrend — it keeps FULL weight while the asset trends up and
    only scales DOWN toward `floor` (freed weight -> cash) as the asset rolls over. Two shapes:
      * 'linear' -- scale = clip(1 + forecast/cap, floor, 1): full (1) for any positive
                    trend, ramping linearly down to `floor` as the forecast falls to -cap
                    (smooth de-risking on the DOWNSIDE only);
      * 'binary' -- scale = 1.0 where forecast > 0 else `floor` (classic TSMOM on/off).
    Warmup (no forecast yet) -> NaN, which the caller treats as neutral (1.0)."""
    fc = combined_forecast(close, lookbacks, vol_window, cap)
    if scheme == "binary":
        scale = fc.where(fc.isna(), (fc > 0).astype(float))
        scale = scale.where(scale != 0.0, floor).where(fc.notna())
        return scale
    if scheme == "linear":
        return (1.0 + fc / cap).clip(lower=floor, upper=1.0)         # cut on downside only
    raise ValueError(f"unknown trend scheme '{scheme}' (use linear | binary)")


# --------------------------------------------------------------------------- #
# LONG/SHORT vol-scaled trend book (the CTA sleeve) — shared so the modelling  #
# trend sleeve AND the post-processing CTA strategy build it from one source.  #
# --------------------------------------------------------------------------- #
def vol_scaled_positions(forecast: pd.DataFrame, close: pd.DataFrame, vol_window: int,
                         per_asset_vol_target: float) -> pd.DataFrame:
    """Turn (signed) forecasts into weights sized so each asset's ex-ante annual vol ~
    forecast * per_asset_vol_target: weight_i = forecast_i * vol_target / ann_vol_i (date x asset)."""
    ann_vol = daily_vol(close, vol_window) * np.sqrt(_ANN)
    ann_vol = ann_vol.where(np.isfinite(ann_vol) & (ann_vol > 0))
    w = forecast * (per_asset_vol_target / ann_vol)
    return w.replace([np.inf, -np.inf], np.nan)


def rebalanced(weights: pd.DataFrame, rebalance_freq: int) -> pd.DataFrame:
    """Update target weights only every `rebalance_freq` rows; hold (ffill) in between."""
    if rebalance_freq <= 1:
        return weights
    mask = np.zeros(len(weights), dtype=bool)
    mask[::rebalance_freq] = True
    held = weights.where(pd.Series(mask, index=weights.index), other=np.nan)
    return held.ffill()


def sleeve_returns(weights: pd.DataFrame, close: pd.DataFrame, fee_bps: float = 1.0,
                   spread_bps: float = 5.0, rebalance_freq: int = 5) -> pd.DataFrame:
    """Daily NET return of a (long/short) trend book from target weights, held between rebalances.
    Point-in-time: weights decided at t-1 earn the t-1->t return; turnover at t costs (fee+spread).
    Returns date-indexed DataFrame [ret, gross, turnover]."""
    cost_rate = (fee_bps + spread_bps) / 1e4
    r = close.pct_change(fill_method=None)
    w = rebalanced(weights, rebalance_freq).reindex(close.index).fillna(0.0)
    w_held = w.shift(1)                                              # held into today's return
    gross_ret = (w_held * r).sum(axis=1, skipna=True)
    turnover = (w - w.shift(1)).abs().sum(axis=1, skipna=True)
    net_ret = gross_ret - turnover * cost_rate
    out = pd.DataFrame({"ret": net_ret, "gross": w.abs().sum(axis=1), "turnover": turnover})
    return out.iloc[1:]                                             # drop the first all-NaN row
