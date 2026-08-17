"""
step_eq_long_only.py  (src/strategies/step_eq_long_only.py)
-----------------------------------------------------------
Long-only equity sleeve `eq_long_only`: holds the best-ranked names by the SAME trained model
signal as the L/S sleeve, top-N with a hold-band buffer, weighted by inverse-vol / ERC — no
shorts. It's the retail-viable long leg of the L/S: fully invested long (market beta ~1), so it
behaves like a smart-beta long equity book (tilted toward the model's top picks), NOT a
market-neutral alpha. Self-contained; shares the signal builder with `step_ls` (no step coupling).
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy, PortfolioInputs, StrategyResult
from src.strategies.utils.ls_model import build_signal
from src.strategies.utils.long_only import long_only_book
from src.strategies.utils.blotter import trade_blotter
from src.utils.risk_parity import series_metrics


class EqLongOnlyStrategy(Strategy):
    name = "eq_long_only"
    config_key = "strategy_eq_long_only"

    def run(self, inputs: PortfolioInputs) -> StrategyResult:
        c = self.config
        b = build_signal(self._context, self._config, end=inputs.end)
        book = long_only_book(
            b.signal, b.stock_ret, float(inputs.capital),
            top_n=int(c.get("top_n", 50)), buffer_mult=float(c.get("buffer_mult", 2.0)),
            weighting=str(c.get("weighting", "inverse_vol")),
            vol_window=int(c.get("vol_window", 63)), rebalance_freq=int(c.get("rebalance_freq", 1)),
            fee_bps=float(c.get("fee_bps", inputs.fee_bps)),
            spread_bps=float(c.get("spread_bps", inputs.spread_bps)),
            cov_shrink=float(c.get("cov_shrink", 0.5)))

        ret = book["net_ret"].astype(float)
        w = book["weights"]
        if inputs.start is not None:
            ret, w = ret[ret.index >= inputs.start], w[w.index >= inputs.start]
        self._log.info("eq_long_only sleeve: %d days, ann-vol %.1f%%, avg %d holdings (top_n=%d, %s)",
                       len(ret), float(ret.std() * (252 ** 0.5)) * 100,
                       int(book["n_holdings"].mean()), int(c.get("top_n", 50)),
                       str(c.get("weighting", "inverse_vol")))

        extra = {}
        if inputs.analysis:
            from src.strategies.analysis.ls_analysis import analyze_ls
            from src.strategies.analysis.common import load_market_refs
            horizon = int(self._config.build_cube.targets.get("primary_horizon", b.horizons[0]))
            out_dir = self._context.paths["OUTPUT_DIR"] / "eq_long_only" / "analysis"
            # same IC / beta-to-SP / corr-to-energy view (here beta_SP ~1 confirms it's a long book)
            extra["analysis"] = analyze_ls(ret, b.signal, b.stock_ret, b.spy_ret,
                                           load_market_refs(self._context.store).get("energy"),
                                           out_dir, horizon=horizon)
            self._log.info("eq_long_only analysis: IC %.3f (IR %.2f), full beta_SP %.2f -> %s",
                           extra["analysis"]["ic_mean"], extra["analysis"]["ic_ir"],
                           extra["analysis"]["full_beta_sp"], out_dir)

        trades = trade_blotter(w, float(inputs.capital), float(c.get("fee_bps", inputs.fee_bps)),
                               float(c.get("spread_bps", inputs.spread_bps)), self.name, prices=b.close)
        return StrategyResult(name=self.name, returns=ret,
                              metrics=series_metrics(ret, inputs.risk_free_rate),
                              positions=w, trades=trades, extra=extra,
                              book_weights=w, book_prices=b.close)
