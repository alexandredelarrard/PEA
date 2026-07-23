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

from src.constants.constants import MACRO_ASSET_PRICES_TABLE
from src.utils.trend import combined_forecast, vol_scaled_positions, sleeve_returns

# level (price / total-return-index) columns usable as trend "close"; renamed to short labels
TREND_CLOSE_COLS = ["equity_tr", "gold", "energy", "bond_10y_tr", "fx_usdeur"]
_RENAME = {"equity_tr": "equity", "bond_10y_tr": "bond", "fx_usdeur": "fx"}
_ANN: float = 252.0


def load_close(store, include_fx: bool = True) -> pd.DataFrame:
    """Wide close (level) matrix (date x asset) for the trend universe from `macro_asset_prices`."""
    df = store.load(MACRO_ASSET_PRICES_TABLE)
    if df is None or df.empty:
        raise RuntimeError(f"Table '{MACRO_ASSET_PRICES_TABLE}' is empty — run fetch_macro_assets.")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").set_index("date")
    cols = [c for c in TREND_CLOSE_COLS if c in d.columns]
    if not include_fx:
        cols = [c for c in cols if c != "fx_usdeur"]
    return d[cols].rename(columns=_RENAME).astype(float)


def _vol_target(ret: pd.Series, target: float, window: int = 126, cap: float = 5.0) -> pd.Series:
    """Scale to `target` ann-vol using TRAILING realized vol (point-in-time, shift 1) — a raw
    multi-asset trend book runs hot (~40% vol); this normalizes it to a comparable ~target."""
    tv = ret.rolling(window, min_periods=max(20, window // 2)).std().shift(1) * np.sqrt(_ANN)
    scale = (target / tv).clip(lower=1.0 / cap, upper=cap).fillna(1.0)
    return ret * scale


def trend_book(close: pd.DataFrame, *, lookbacks: tuple[int, ...] = (63, 126, 252),
               vol_window: int = 63, signal_cap: float = 2.0, per_asset_vol_target: float = 0.15,
               sleeve_vol_target: float = 0.10, rebalance_freq: int = 5,
               fee_bps: float = 2.0, spread_bps: float = 8.0) -> dict[str, object]:
    """Long/short trend book -> {ret (daily NET, vol-targeted), positions (date x asset weights),
    gross, turnover}. Point-in-time (signal at t-1 earns t-1->t)."""
    forecast = combined_forecast(close, list(lookbacks), vol_window, signal_cap)   # SIGNED
    weights = vol_scaled_positions(forecast, close, vol_window, per_asset_vol_target)
    sr = sleeve_returns(weights, close, fee_bps, spread_bps, rebalance_freq)
    ret = _vol_target(sr["ret"].astype(float), sleeve_vol_target)
    return {"ret": ret, "positions": weights, "gross": sr["gross"], "turnover": sr["turnover"]}
