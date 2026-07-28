"""
step_long_book.py  (src/strategies/step_long_book.py)
-----------------------------------------------------
Long-book strategy sleeve: reads `strategy_long_book` config + PortfolioInputs, loads the macro
asset prices, and runs the long-book allocation "model" (ERC/inverse-vol + trend overlay + VIX
regime tilt + responsive leverage). Self-contained; no dependency on other strategy steps.
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy, PortfolioInputs, StrategyResult
from src.constants.constants import MACRO_ASSET_PRICES_TABLE
from src.modelling.long_book.allocation import asset_returns_from_macro, allocation_backtest
from src.utils.risk_parity import series_metrics


class LongBookStrategy(Strategy):
    name = "long_book"
    config_key = "strategy_long_book"

    def run(self, inputs: PortfolioInputs) -> StrategyResult:
        c = self.config
        df = self._context.store.load(MACRO_ASSET_PRICES_TABLE)
        if df is None or df.empty:
            raise RuntimeError(f"'{MACRO_ASSET_PRICES_TABLE}' empty — run fetch_macro_assets.")
        rets, cash = asset_returns_from_macro(df, include_fx=bool(c.get("include_fx", True)))
        vix = None
        if bool(c.get("use_vix", False)) and "vix" in df.columns:
            d = df.copy(); d["date"] = pd.to_datetime(d["date"])
            vix = d.sort_values("date").set_index("date")["vix"].astype(float).reindex(rets.index)

        res = allocation_backtest(
            rets, cash,
            scheme=str(c.get("scheme", "erc")), vol_window=int(c.get("vol_window", 63)),
            rebalance_freq=int(c.get("rebalance_freq", 21)),
            trend_enabled=bool(c.get("trend_enabled", True)),
            trend_lookbacks=tuple(int(x) for x in c.get("trend_lookbacks", [63, 126, 252])),
            trend_scheme=str(c.get("trend_scheme", "binary")),
            trend_floor=float(c.get("trend_floor", 0.0)),
            trend_vol_window=int(c.get("trend_vol_window", 63)),
            trend_cap=float(c.get("trend_cap", 2.0)),
            portfolio_vol_target=float(inputs.target_vol),          # sleeve targets the reference vol
            max_leverage=float(c.get("max_leverage", 2.0)),
            fee_bps=float(c.get("fee_bps", inputs.fee_bps)),
            spread_bps=float(c.get("spread_bps", inputs.spread_bps)),
            cov_mode=str(c.get("cov_mode", "ewma")), cov_halflife=int(c.get("cov_halflife", 42)),
            vol_mode=str(c.get("vol_mode", "ewma")), lever_on=str(c.get("lever_on", "base")),
            risk_on=bool(c.get("risk_on_tilt", False)), vix=vix,
            offensive=tuple(c.get("offensive", ["equity", "energy"])),
            off_share_range=(float(c.get("off_share_min", 0.15)), float(c.get("off_share_max", 0.85))),
            lev_responsive=bool(c.get("lev_responsive", False)),
            lev_min=float(c.get("lev_min", 1.0)), lev_max=float(c.get("lev_max", 2.0)))

        ret = _slice(res["net_ret"].astype(float), inputs.start, inputs.end)
        alloc = res["alloc"].copy()
        alloc["cash"] = res["alloc_cash"]
        self._log.info("long_book sleeve: %d days, ann-vol %.1f%%",
                       len(ret), float(ret.std() * (252 ** 0.5)) * 100)
        extra = {"leverage": res["leverage"], "cash_weight": res["alloc_cash"]}
        if inputs.analysis:
            from src.strategies.analysis.long_book_analysis import analyze_long_book
            out_dir = self._context.paths["OUTPUT_DIR"] / "long_book" / "analysis"
            extra["analysis"] = analyze_long_book(rets, out_dir)   # FULL-history asset-class corr
            self._log.info("long_book analysis: avg pairwise corr %.2f -> %s",
                           extra["analysis"]["avg_pairwise_corr"], out_dir)
        # trade blotter: levered risky weights + cash residual, share-accurate on the asset LEVELS
        # (equity_tr/gold/energy/bond_10y_tr/fx_usdeur — renamed to the alloc's asset labels; cash
        # has no price -> reported in $ only, no fee).
        from src.strategies.utils.blotter import trade_blotter
        _lvl = {"equity": "equity_tr", "gold": "gold", "energy": "energy",
                "bond": "bond_10y_tr", "fx": "fx_usdeur"}
        dd = df.copy(); dd["date"] = pd.to_datetime(dd["date"]); dd = dd.sort_values("date").set_index("date")
        levels = pd.DataFrame({k: dd[v].astype(float) for k, v in _lvl.items() if v in dd.columns})
        wl = res["weights"].copy(); wl["cash"] = res["alloc_cash"]
        book = _slice(wl, inputs.start, inputs.end)
        trades = trade_blotter(book, inputs.capital,
                               float(c.get("fee_bps", inputs.fee_bps)),
                               float(c.get("spread_bps", inputs.spread_bps)), self.name,
                               prices=levels)
        return StrategyResult(name=self.name, returns=ret,
                              metrics=series_metrics(ret, inputs.risk_free_rate),
                              positions=_slice(alloc, inputs.start, inputs.end),
                              trades=trades, extra=extra,
                              book_weights=book, book_prices=levels)


def _slice(obj, start, end):
    if start is not None:
        obj = obj[obj.index >= start]
    if end is not None:
        obj = obj[obj.index <= end]
    return obj
