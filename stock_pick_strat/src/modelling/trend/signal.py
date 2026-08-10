"""
signal.py  (src/modelling/trend/signal.py)
------------------------------------------
The TREND / CTA "model": a DIRECTIONAL long/short time-series-momentum book on the macro
asset universe (equity / gold / energy / 10Y-bond-TR / FX from `macro_asset_prices`). Unlike
the long book (always long), it goes SHORT an asset trending down — so it PROFITS in sustained
sell-offs (short bonds in 2022, short equities in 2008): the positive-skew / crisis-alpha
diversifier. Self-contained on the shared trend blocks (src/utils/trend); no ML.

`trend_book()` is the "prediction + construction": vol-normalized multi-lookback forecast ->
vol-scaled long/short weights -> daily NET returns (trailing-vol-scaled to a sleeve vol target).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_store.schema import Tables
from src.utils.trend import combined_forecast, vol_scaled_positions, sleeve_returns

# level (price / total-return-index) columns usable as trend "close"; renamed to short labels
TREND_CLOSE_COLS = ["equity_tr", "gold", "energy", "bond_10y_tr", "fx_usdeur"]
_RENAME = {"equity_tr": "equity", "bond_10y_tr": "bond", "fx_usdeur": "fx"}
_ANN: float = 252.0


def load_close(store, include_fx: bool = True) -> pd.DataFrame:
    """Wide close (level) matrix (date x asset) for the trend universe from `macro_asset_prices`."""
    df = store.load(Tables.macro_asset_prices, optional=True)
    if df is None:
        raise RuntimeError(f"Table '{Tables.macro_asset_prices}' is empty — run fetch_macro_assets.")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").set_index("date")
    cols = [c for c in TREND_CLOSE_COLS if c in d.columns]
    if not include_fx:
        cols = [c for c in cols if c != "fx_usdeur"]
    return d[cols].rename(columns=_RENAME).astype(float)


def _vol_target_scale(ret: pd.Series, target: float, window: int = 126, cap: float = 5.0) -> pd.Series:
    """Trailing-vol-target SCALAR series (point-in-time, shift 1): target / trailing-ann-vol, capped.
    A raw multi-asset trend book runs hot (~40% vol); this normalizes it to a comparable ~target.
    Applied to BOTH the return stream and the position panel (so the held book is consistent)."""
    tv = ret.rolling(window, min_periods=max(20, window // 2)).std().shift(1) * np.sqrt(_ANN)
    return (target / tv).clip(lower=1.0 / cap, upper=cap).fillna(1.0)


def trend_book(close: pd.DataFrame, *, lookbacks: tuple[int, ...] = (63, 126, 252),
               vol_window: int = 63, signal_cap: float = 2.0, per_asset_vol_target: float = 0.15,
               sleeve_vol_target: float = 0.10, rebalance_freq: int = 5,
               fee_bps: float = 2.0, spread_bps: float = 8.0) -> dict[str, object]:
    """Long/short trend book -> {ret (daily NET, vol-targeted), positions (date x asset weights),
    gross, turnover}. Point-in-time (signal at t-1 earns t-1->t)."""
    forecast = combined_forecast(close, list(lookbacks), vol_window, signal_cap)   # SIGNED
    weights = vol_scaled_positions(forecast, close, vol_window, per_asset_vol_target)
    sr = sleeve_returns(weights, close, fee_bps, spread_bps, rebalance_freq)
    scale = _vol_target_scale(sr["ret"].astype(float), sleeve_vol_target)
    ret = sr["ret"].astype(float) * scale
    positions = weights.mul(scale.reindex(weights.index), axis=0)     # EFFECTIVE held book
    return {"ret": ret, "positions": positions, "scale": scale,
            "gross": sr["gross"], "turnover": sr["turnover"]}
