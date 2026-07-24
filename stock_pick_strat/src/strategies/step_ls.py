"""
step_ls.py  (src/strategies/step_ls.py)
---------------------------------------
Market-neutral equity LONG/SHORT (alpha) strategy sleeve. Flow:
  1. read `strategy_ls` construction config + PortfolioInputs (capital, target vol, window)
  2. PREDICT: `ls_model.build_signal` loads the trained ensemble + projects the OOS cube and
     blends horizons -> per-name combined z-signal (shared with the long-only sleeve)
  3. CONSTRUCT: dollar/beta/sector-neutral optimizer (strategies_opt, market_weight=0 -> pure
     alpha; diagonal inverse-variance OR correlation-aware shrunk-covariance risk model),
     sized to the portfolio's target vol
  4. metrics + trade blotter -> StrategyResult
Self-contained: depends on no other strategy step (the signal builder is a shared util).
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy, PortfolioInputs, StrategyResult
from src.strategies.utils.ls_model import build_signal
from src.strategies.utils.strategies_opt import simulate_portfolio_opt, simulate_integer_ls
from src.utils.risk_parity import series_metrics


class LongShortStrategy(Strategy):
    name = "ls_equity"

    def run(self, inputs: PortfolioInputs) -> StrategyResult:
        self._inputs = inputs
        self._cfg = self._config.strategy_ls
        self._cube_cfg = self._config.build_cube
        b = build_signal(self._context, self._config, end=inputs.end)
        self.signal, self.stock_ret, self.spy_ret, self.close = b.signal, b.stock_ret, b.spy_ret, b.close
        self.backtest_start, self.horizons = b.backtest_start, b.horizons
        self._log.info("ls_equity: signal %s days x %s tickers; OOS from %s",
                       *self.signal.shape, self.backtest_start.date())

        daily = self.simulate()
        if daily is None or daily.empty:
            self._log.warning("ls_equity: empty alpha book (no tradeable signal in window)")
            return StrategyResult(self.name, pd.Series(dtype=float), series_metrics(pd.Series(dtype=float)))
        ret = daily["net_ret"].astype(float)
        if inputs.start is not None:
            ret = ret[ret.index >= inputs.start]
        self._log.info("ls_equity sleeve: %d OOS days from %s, ann-vol %.1f%%",
                       len(ret), self.backtest_start.date(), float(ret.std() * (252 ** 0.5)) * 100)
        extra = {"signal_shape": tuple(self.signal.shape)}
        if inputs.analysis:
            extra["analysis"] = self._analyze(ret)
        trades = self._blotter(daily, inputs)
        return StrategyResult(name=self.name, returns=ret,
                              metrics=series_metrics(ret, inputs.risk_free_rate),
                              positions=None, trades=trades, extra=extra)

    def _sector_map(self) -> dict:
        tk = self._context.store.load("sp500_tickers")
        if tk.empty:
            return {}
        col = ("industry_group" if "industry_group" in tk.columns
               else "sector" if "sector" in tk.columns else None)
        return dict(zip(tk["ticker"], tk[col])) if col else {}

    def simulate(self):
        c, inp = self._cfg, self._inputs
        sector_neutral = bool(c.get("sector_neutral", False))
        sector_map = self._sector_map() if sector_neutral else None
        isc = dict(c.get("integer_shares", {}) or {})
        if bool(isc.get("enabled", False)):                 # WHOLE-SHARE book (no fractional shorts)
            return simulate_integer_ls(
                signal=self.signal, stock_ret=self.stock_ret, spy_ret=self.spy_ret, close=self.close,
                starting_capital=float(inp.capital), target_ann_vol=float(inp.target_vol),
                beta_neutral=c.get("beta_neutral", True), pos_cap=c.get("pos_cap", 0.05),
                gross_cap=c.get("gross_cap", 3.0), beta_window=c.get("beta_window", 63),
                vol_window=c.get("vol_window", 63),
                fee_bps=float(c.get("fee_bps", inp.fee_bps)), spread_bps=float(c.get("spread_bps", inp.spread_bps)),
                rebalance_freq=c.get("rebalance_freq", 63), sector_map=sector_map,
                sector_neutral=sector_neutral, risk_model=str(c.get("risk_model", "diagonal")),
                cov_shrink=float(c.get("cov_shrink", 0.5)),
                gross_tol=float(isc.get("gross_tol", 0.02)), dollar_tol=float(isc.get("dollar_tol", 0.005)),
                beta_tol=float(isc.get("beta_tol", 0.02)), sector_tol=float(isc.get("sector_tol", 0.03)),
                int_method=str(isc.get("method", "milp")),
                share_cap_mult=float(isc.get("share_cap_mult", 3.0)),
                time_limit=float(isc.get("time_limit", 10.0)),
                long_fractional=bool(isc.get("long_fractional", False)))
        return simulate_portfolio_opt(
            signal=self.signal, stock_ret=self.stock_ret, spy_ret=self.spy_ret,
            starting_capital=float(inp.capital), target_ann_vol=float(inp.target_vol),
            beta_neutral=c.get("beta_neutral", True), pos_cap=c.get("pos_cap", 0.05),
            gross_cap=c.get("gross_cap", 3.0), step=c.get("step", 0.35),
            no_trade_band=c.get("no_trade_band", 0.0), beta_window=c.get("beta_window", 63),
            vol_window=c.get("vol_window", 63),
            fee_bps=float(c.get("fee_bps", inp.fee_bps)), spread_bps=float(c.get("spread_bps", inp.spread_bps)),
            rebalance_freq=c.get("rebalance_freq", 63), sector_map=sector_map,
            sector_neutral=sector_neutral,
            risk_model=str(c.get("risk_model", "diagonal")), cov_shrink=float(c.get("cov_shrink", 0.5)),
            market_weight=0.0, vol_scaling=False, collect_weights=True)

    def _blotter(self, daily: pd.DataFrame, inputs: PortfolioInputs):
        """Per-(day, ticker) SHARE-accurate trade blotter from the optimizer's captured weights."""
        w = daily.attrs.get("weights") if daily is not None else None
        if w is None or w.empty:
            return None
        from src.strategies.utils.blotter import trade_blotter
        if inputs.start is not None:
            w = w[w.index >= inputs.start]
        c = self._cfg
        return trade_blotter(w, inputs.capital, float(c.get("fee_bps", inputs.fee_bps)),
                             float(c.get("spread_bps", inputs.spread_bps)), self.name,
                             prices=getattr(self, "close", None))

    def _analyze(self, ret: pd.Series) -> dict:
        """IC + Sharpe/maxDD + market-neutrality (beta to SP, corr to energy) plots."""
        from src.strategies.analysis.ls_analysis import analyze_ls
        from src.strategies.analysis.common import load_market_refs
        refs = load_market_refs(self._context.store)
        horizon = int(self._cube_cfg.targets.get("primary_horizon", self.horizons[0]))
        out_dir = self._context.paths["OUTPUT_DIR"] / "ls_equity" / "analysis"
        m = analyze_ls(ret, self.signal, self.stock_ret, self.spy_ret,
                       refs.get("energy"), out_dir, horizon=horizon)
        self._log.info("ls_equity analysis: IC %.3f (IR %.2f), full beta_SP %.2f, corr_energy %.2f -> %s",
                       m["ic_mean"], m["ic_ir"], m["full_beta_sp"], m["full_corr_energy"], out_dir)
        return m
