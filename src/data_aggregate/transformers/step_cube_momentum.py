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

from src.context import Context
from src.data_aggregate.utils.common.incremental import plan_window, write_part
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.momentum.features import build_feature_panel
from src.data_aggregate.utils.price.checks import measure_quote_seams, measure_seams
from src.data_aggregate.utils.price.frames import (
    PriceFrames,
    load_price_frames,
    load_trading_calendar,
)
from src.data_store.schema import Table, Tables
from src.utils.price.adjustments import load_bugfix
from src.utils.step import Step
from src.utils.window import WindowPlan


class StepCubeMomentum(Step):
    # `ret` is read from the part rather than recomputed: build_feature_panel used to derive
    # it internally from close, duplicating what the price step already persisted
    # THREE price bases: `close_total` for every return-shaped feature, `close_split` for the
    # three that subtract an `open`/`high`/`low` (all split-adjusted only), and the as-traded
    # `close_pit x volume_pit` for dollar volume. See the basis table at the top of momentum/features.py.
    _FIELDS = ("close_split", "close_total", "open", "high", "low", "volume", "ret", "sector_ret", "close_pit", "volume_pit")

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.data_aggregate
        self._store = context.store
        # the `null_ret` entries reach this step as MASKS, not as repairs: `StepCubePrices`
        # already nulled the one `ret` cell, but it left both price legs as published, so
        # `close_total` is still on two bases here. See level_basis's module docstring.
        self._bugfix = load_bugfix(context.config_dir)

    def run(self) -> dict[Table, WindowPlan]:
        # Plan configured read and write windows on the retained source calendar.
        calendar = load_trading_calendar(self._context)
        window = plan_window(self._context, Tables.cube_part_momentum, calendar)
        if window is None:
            return {}
        frames = load_price_frames(
            self._context, peers=load_peers_or_raise(self._context, self._config), calendar=calendar, fields=self._FIELDS, window=window.read
        )
        panel = self._price_panel(frames)
        del frames
        write_part(self._store, Tables.cube_part_momentum, panel, window, drop_empty=True)
        return {Tables.cube_part_momentum: window}

    def _price_panel(self, frames: PriceFrames) -> pd.DataFrame:
        frames.require("close_split", "close_total", "open", "sector_ret", "ret")
        assert frames.close_total is not None
        assert frames.open is not None
        assert frames.sector_ret is not None
        seams = measure_seams(frames.close_total, self._bugfix)
        assert frames.close_split is not None
        quote_seams = measure_quote_seams(frames.close_split, self._bugfix)
        split_seams = {ticker: sorted(set(seams.get(ticker, [])) | set(quote_seams.get(ticker, []))) for ticker in seams.keys() | quote_seams.keys()}
        listed = sum(len(v) for v in (self._bugfix.get("null_ret") or {}).values())
        # a SKIP here is not cosmetic: it means the register no longer matches this build's
        # `close_total`, so the window that would have been masked is being emitted fabricated.
        self._log.info(
            "Seam masks: %s of %s registered null_ret entries measured on close_total (%s ticker(s))",
            sum(len(v) for v in seams.values()),
            listed,
            len(seams),
        )
        panel = build_feature_panel(
            frames,
            method=self._cfg.features.standardize_method,
            seasonal_horizons=[int(h) for h in self._cfg.targets.horizons],
            seams=seams,
            split_seams=split_seams,
        )
        self._log.info(
            "Price feature panel: %s rows, %s features (volume liquidity: %s)",
            len(panel),
            len(panel.columns) - 2,
            "yes" if frames.volume is not None else "no",
        )
        return panel
