"""
strategies_blend.py  (src/post_processing/utils/strategies_blend.py)
--------------------------------------------------------------------
Day-by-day blend of several strategy sleeves' daily return streams (e.g. SP500, equity L/S alpha,
multi-asset trend) into one book, weighting by RISK and targeting a fixed portfolio volatility.

Two SEPARATE decisions (keeping them apart is what stops the book collapsing to ~0 exposure):
  1. MIX  — inverse-vol (risk-parity) weights across sleeves: w_i ∝ 1/vol_i, capped, renormalized
            to sum to 1. This decides the RELATIVE allocation only, so it never shrinks the book.
  2. SIZE — one portfolio-level leverage that scales the blended stream so its trailing vol hits
            `portfolio_vol_target` (capped by `max_leverage`). This decides the total risk.
Everything is point-in-time: weights/leverage at t use volatility estimated up to t-1, applied to
the sleeves' day-t returns, so there is no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ANN: float = 252.0


def trailing_vol(rets: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    """Point-in-time annualized trailing vol per sleeve (shifted 1 day → uses info up to t-1)."""
    mp = int(min_periods) if min_periods is not None else max(10, window // 2)
    return rets.rolling(window, min_periods=mp).std().shift(1) * np.sqrt(_ANN)


def inverse_vol_weights(rets: pd.DataFrame, window: int, scheme: str = "inverse_vol",
                        max_weight: float = 1.0) -> pd.DataFrame:
    """date x sleeve MIX weights summing to 1 each day. `inverse_vol` = risk parity (∝1/vol);
    `equal` = equal-weight. Only sleeves with a return that day get weight; cap + renormalize."""
    if scheme == "equal":
        w = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
    elif scheme == "inverse_vol":
        vol = trailing_vol(rets, window).replace(0.0, np.nan)
        w = 1.0 / vol
    else:
        raise ValueError(f"unknown weight scheme '{scheme}' (use inverse_vol | equal)")
    w = w.where(rets.notna())                                   # don't allocate to a missing sleeve
    w = w.div(w.sum(axis=1), axis=0)
    if max_weight < 1.0:                                        # cap concentration, renormalize
        for _ in range(3):
            w = w.clip(upper=max_weight)
            w = w.div(w.sum(axis=1), axis=0)
    return w.fillna(0.0)


def blend_to_vol_target(rets: pd.DataFrame, weights: pd.DataFrame, portfolio_vol_target: float,
                        vol_window: int, max_leverage: float = 2.0) -> pd.DataFrame:
    """Combine sleeves with `weights` (the MIX) then scale the whole book to `portfolio_vol_target`.
    Returns date-indexed [ret, leverage, mix_ret]. `ret` is the final blended daily return."""
    mix = (weights * rets).sum(axis=1, skipna=True)             # weights sum to 1 → the mix stream
    mp = max(10, vol_window // 2)
    pv = mix.rolling(vol_window, min_periods=mp).std().shift(1) * np.sqrt(_ANN)
    lev = (portfolio_vol_target / pv).clip(upper=max_leverage)
    lev = lev.where(np.isfinite(lev)).fillna(1.0)               # warmup → neutral 1.0
    return pd.DataFrame({"ret": lev * mix, "leverage": lev, "mix_ret": mix})


def blend_strategies(rets: pd.DataFrame, portfolio_vol_target: float = 0.10, vol_window: int = 63,
                     scheme: str = "inverse_vol", max_weight: float = 0.7,
                     max_leverage: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end day-by-day blend of sleeve return streams (date x sleeve) -> (blended, weights).
    `blended` has [ret, leverage, mix_ret]; `weights` is the daily sleeve mix."""
    rets = rets.sort_index()
    weights = inverse_vol_weights(rets, vol_window, scheme, max_weight)
    blended = blend_to_vol_target(rets, weights, portfolio_vol_target, vol_window, max_leverage)
    return blended, weights
