"""
strategy_ls.py  (src/post_processing/strategies/strategy_ls.py)
---------------------------------------------------------------
The market-neutral equity LONG/SHORT (alpha) sleeve: long the model's high-conviction names,
short the low ones, dollar/beta/sector-neutral — orthogonal to the market by construction.
Wraps the existing `StepBacktest` engine (model load → cube projection → signal → optimizer)
and takes the PURE alpha book (`market_weight=0`, no vol overlay) as the sleeve return. Only
out-of-sample: it starts at the trained model's `train_end` (earlier would be look-ahead).
"""
from __future__ import annotations

import pandas as pd

from src.post_processing.strategies.base import Strategy
from src.post_processing.step_backtest import StepBacktest
from src.post_processing.utils.strategies_opt import simulate_portfolio_opt


class LongShortStrategy(Strategy):
    name = "ls_equity"

    def returns(self) -> pd.Series:
        bt = StepBacktest(context=self._context, config=self._config)
        bt.load_models()                        # sets bt.backtest_start = model train_end (OOS start)
        bt.load_cube_and_returns()
        bt.predict_and_blend()                  # sets bt.signal / bt.stock_ret / bt.spy_ret
        sector_map = bt._sector_map() if bt._cfg.get("sector_neutral", False) else None
        alpha = simulate_portfolio_opt(**bt._opt_common(sector_map),
                                       market_weight=0.0, vol_scaling=False)   # pure market-neutral alpha
        self._bt = bt
        if alpha is None or alpha.empty:
            self._log.warning("ls_equity sleeve: empty alpha book (no tradeable signal in window)")
            return pd.Series(dtype=float)
        ret = alpha["net_ret"].astype(float)
        self._log.info("ls_equity sleeve: %d OOS days from %s, ann-vol %.1f%%",
                       len(ret), bt.backtest_start.date(), float(ret.std() * (252 ** 0.5)) * 100)
        return ret
