"""
step_trend.py  (src/strategies/step_trend.py)
---------------------------------------------
Trend / CTA strategy sleeve: reads `strategy_trend` config + PortfolioInputs, loads the macro
asset closes, and runs the long/short time-series-momentum "model" (trend forecast -> vol-scaled
long/short book -> vol-targeted returns). Self-contained; no dependency on other strategy steps.
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy, PortfolioInputs, StrategyResult
from src.modelling.trend.signal import load_close, trend_book
from src.utils.risk_parity import series_metrics


class TrendCTAStrategy(Strategy):
    name = "trend_cta"

    def run(self, inputs: PortfolioInputs) -> StrategyResult:
        c = self._config.strategy_trend
        close = load_close(self._context.store, include_fx=bool(c.get("include_fx", True)))
        book = trend_book(
            close,
            lookbacks=tuple(int(x) for x in c.get("lookbacks", [63, 126, 252])),
            vol_window=int(c.get("vol_window", 63)), signal_cap=float(c.get("signal_cap", 2.0)),
            per_asset_vol_target=float(c.get("per_asset_vol_target", 0.15)),
            sleeve_vol_target=float(inputs.target_vol),             # sleeve targets the reference vol
            rebalance_freq=int(c.get("rebalance_freq", 5)),
            fee_bps=float(c.get("fee_bps", inputs.fee_bps)),
            spread_bps=float(c.get("spread_bps", inputs.spread_bps)))

        ret = _slice(book["ret"].astype(float), inputs.start, inputs.end)
        positions = _slice(book["positions"], inputs.start, inputs.end)
        self._log.info("trend_cta sleeve universe %s: %d days, ann-vol %.1f%%",
                       list(close.columns), len(ret), float(ret.std() * (252 ** 0.5)) * 100)
        extra = {}
        if inputs.analysis:
            from src.strategies.analysis.trend_analysis import analyze_trend
            from src.strategies.analysis.common import load_market_refs
            sp = load_market_refs(self._context.store).get("sp", pd.Series(dtype=float))
            out_dir = self._context.paths["OUTPUT_DIR"] / "trend_cta" / "analysis"
            extra["analysis"] = analyze_trend(ret, positions, sp, out_dir)
            self._log.info("trend_cta analysis: full beta_SP %.2f -> %s",
                           extra["analysis"]["full_beta_sp"], out_dir)
        return StrategyResult(name=self.name, returns=ret,
                              metrics=series_metrics(ret, inputs.risk_free_rate),
                              positions=positions, extra=extra)


def _slice(obj, start, end):
    if start is not None:
        obj = obj[obj.index >= start]
    if end is not None:
        obj = obj[obj.index <= end]
    return obj
