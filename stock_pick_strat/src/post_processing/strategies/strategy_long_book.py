"""
strategy_long_book.py  (src/post_processing/strategies/strategy_long_book.py)
-----------------------------------------------------------------------------
The multi-asset LONG BOOK sleeve: a long-only % allocation across equity / gold / energy /
10Y-bond-TR / FX + cash (risk-parity ERC + trend overlay + VIX regime tilt + responsive
leverage). Wraps the existing `StepAllocationBacktest` engine unchanged — this class just
adapts it to the `Strategy.returns()` interface for the portfolio blend.
"""
from __future__ import annotations

import pandas as pd

from src.post_processing.strategies.base import Strategy
from src.post_processing.step_alloc_backtest import StepAllocationBacktest


class LongBookStrategy(Strategy):
    name = "long_book"

    def returns(self) -> pd.Series:
        step = StepAllocationBacktest(context=self._context, config=self._config)
        step.load_assets()
        step.backtest()
        self._step = step
        self._log.info("long_book sleeve: %d days, ann-vol %.1f%%",
                       len(step.daily), step.metrics["ann_vol"] * 100)
        return step.result["net_ret"].astype(float)

    def positions(self) -> pd.DataFrame | None:
        step = getattr(self, "_step", None)
        return None if step is None else step.result.get("alloc")
