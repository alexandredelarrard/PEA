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
