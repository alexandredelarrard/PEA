"""
strategy_trend.py  (src/post_processing/strategies/strategy_trend.py)
---------------------------------------------------------------------
The TREND / CTA sleeve: a DIRECTIONAL long/short time-series-momentum book on the macro
asset universe (equity / gold / energy / 10Y-bond-TR / FX from `macro_asset_prices`). Unlike
the long book (always long), it goes SHORT an asset trending down — so it PROFITS in sustained
sell-offs (short bonds in 2022, short equities in 2008): the positive-skew / crisis-alpha
diversifier. Self-contained: builds the book from the shared `src/utils/trend` blocks (no
modelling import), on the same long-history data as the long book.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.post_processing.strategies.base import Strategy
from src.constants.constants import MACRO_ASSET_PRICES_TABLE
from src.utils.trend import combined_forecast, vol_scaled_positions, sleeve_returns

# level (price / total-return-index) columns usable as trend "close"; renamed to short labels
_TREND_CLOSE_COLS = ["equity_tr", "gold", "energy", "bond_10y_tr", "fx_usdeur"]
_RENAME = {"equity_tr": "equity", "bond_10y_tr": "bond", "fx_usdeur": "fx"}


class TrendCTAStrategy(Strategy):
    name = "trend_cta"

    def returns(self) -> pd.Series:
        c = dict(self._config.backtest.portfolio_backtest.get("trend", {}) or {})
        df = self._context.store.load(MACRO_ASSET_PRICES_TABLE)
        if df is None or df.empty:
            raise RuntimeError(f"Table '{MACRO_ASSET_PRICES_TABLE}' is empty — run fetch_macro_assets.")
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date").set_index("date")
        cols = [x for x in _TREND_CLOSE_COLS if x in d.columns]
        if not bool(c.get("include_fx", True)):
            cols = [x for x in cols if x != "fx_usdeur"]
        close = d[cols].rename(columns=_RENAME).astype(float)

        lookbacks = [int(x) for x in c.get("lookbacks", [63, 126, 252])]
        vw, cap = int(c.get("vol_window", 63)), float(c.get("signal_cap", 2.0))
        forecast = combined_forecast(close, lookbacks, vw, cap)              # SIGNED long/short
        weights = vol_scaled_positions(forecast, close, vw,
                                       float(c.get("per_asset_vol_target", 0.15)))
        sr = sleeve_returns(weights, close, float(c.get("fee_bps", 2.0)),
                            float(c.get("spread_bps", 8.0)), int(c.get("rebalance_freq", 5)))
        self._positions = weights
        ret = self._vol_target(sr["ret"].astype(float), float(c.get("sleeve_vol_target", 0.10)))
        self._log.info("trend_cta sleeve universe %s: %d days, ann-vol %.1f%% (target %.0f%%)",
                       list(close.columns), len(ret), float(ret.std() * (252 ** 0.5)) * 100,
                       float(c.get("sleeve_vol_target", 0.10)) * 100)
        return ret

    @staticmethod
    def _vol_target(ret: pd.Series, target: float, window: int = 126, cap: float = 5.0) -> pd.Series:
        """Scale the sleeve to `target` ann-vol using TRAILING realized vol (point-in-time, shift 1
        so today's scale uses only past vol) — a raw multi-asset trend book runs hot (~40% vol);
        this normalizes it to a comparable ~target so the sleeve metrics + blend are sensible."""
        tv = ret.rolling(window, min_periods=max(20, window // 2)).std().shift(1) * np.sqrt(252.0)
        scale = (target / tv).clip(lower=1.0 / cap, upper=cap).fillna(1.0)
        return ret * scale

    def positions(self) -> pd.DataFrame | None:
        return getattr(self, "_positions", None)
