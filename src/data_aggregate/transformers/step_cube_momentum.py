"""
step_cube_momentum.py  (src/data_aggregate/transformers/step_cube_momentum.py)
--------------------------------------------------------------------------
Everything derived from PRICE VARIATION -> `cube_part_momentum`: momentum and reversal,
trailing volatility, moving-average trend, 52-week-high proximity, gaps and ranges, the
lottery family (max return, skew, downside/idiosyncratic vol), the liquidity family (dollar
volume, Amihud illiquidity, relative volume, signed volume, volume trend/CV), cross-sectional
seasonality per target horizon, tax-loss-selling pressure, and the lagged MACD / RSI / ATR.

The only feature step whose panel arrives ALREADY long with its own column names, so it
needs no (date, ticker) skeleton and no merge -- `build_feature_panel` emits the finished
frame.

Warm-up 1320 trading days: `seasonal_h*` reaches back `close.shift(252 * seasonal_years=5)`
= 1260, which is the binding look-back in the whole cube.
"""
from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.data_store.schema import Tables
from src.context import Context
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, plan_window, write_part
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import (
    PriceFrames, load_price_frames, load_trading_calendar,
)
from src.data_aggregate.utils.momentum.features import build_feature_panel
from src.utils.step import Step


class StepCubeMomentum(Step):

    # `ret` is read from the part rather than recomputed: build_feature_panel used to derive
    # it internally from close, duplicating what the price step already persisted
    _FIELDS = ("close", "open", "high", "low", "volume", "ret", "sector_ret")

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_momentum)
        self._store = context.store

    def run(self, full: bool = False) -> None:
        window = plan_window(self._store, Tables.cube_part_momentum, full=full,
                             warmup=self._warmup(),
                             trading_index=load_trading_calendar(self._store))
        frames = self._load_frames(window.since)
        panel = self._price_panel(frames)
        del frames
        n = write_part(self._store, Tables.cube_part_momentum, panel, window, drop_empty=True)
        if n == COLUMNS_CHANGED:
            return self.run(full=True)

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    def _load_frames(self, since: pd.Timestamp | None) -> PriceFrames:
        return load_price_frames(
            self._store, peers=load_peers_or_raise(self._context, self._config),
            fields=self._FIELDS, since=since)

    def _price_panel(self, frames: PriceFrames) -> pd.DataFrame:
        frames.require("close", "open", "sector_ret", "ret")
        panel = build_feature_panel(
            frames.close, frames.open, frames.sector_ret,
            method=self._cfg.features.standardize_method,
            high=frames.high, low=frames.low, volume=frames.volume,
            seasonal_horizons=[int(h) for h in self._cfg.targets.horizons],
            returns=frames.ret,
        )
        self._log.info("Price feature panel: %s rows, %s features (volume liquidity: %s)",
                       len(panel), len(panel.columns) - 2,
                       "yes" if frames.volume is not None else "no")
        return panel
