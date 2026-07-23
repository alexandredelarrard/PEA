"""
step_portfolio.py  (src/portfolio/step_portfolio.py)
----------------------------------------------------
StepPortfolio — the UNIFIED backtest. Reads `configs/portfolio.yml`, builds the configured set
of strategy sleeves, hands each a common `PortfolioInputs` (capital, target vol, window, fees),
collects each sleeve's daily P&L via `Strategy.run(inputs)`, then:
  1. MIX  — risk-parity (inverse-vol EWMA) or **ERC** weights across the sleeve RETURN streams,
            point-in-time + NaN-aware (a late-starting sleeve like L/S joins once it has history);
            these weights ARE the dynamic capital allocation per sleeve;
  2. SIZE — one global leverage to hit `portfolio_vol_target` (capped by `max_leverage`).
Reports return / vol / **Sharpe per strategy** vs the **global** portfolio + a simple SP-hold,
the sleeve cross-correlation, and the $-allocation per sleeve; saves an equity curve + weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.constants.constants import MACRO_ASSET_PRICES_TABLE
from src.strategies import STRATEGY_REGISTRY, PortfolioInputs
from src.strategies.utils.metrics import compute_metrics
from src.utils.risk_parity import base_weights, series_metrics, daily_frame
from src.portfolio.utils.blend import blend_to_vol_target

_SLEEVE_COLORS = {"ls_equity": "#1f77b4", "long_book": "#2ca02c", "trend_cta": "#d62728"}


class StepPortfolio(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.portfolio

    def run(self) -> None:
        self.load_sleeves()
        self.blend()
        self.report()
        self.plots()
        self.save_trades()

    # ------------------------------------------------------------------ #
    def save_trades(self) -> None:
        """Aggregate each sleeve's daily trade blotter into ONE workbook (one sheet per sleeve
        + a summary), so the trading needed to run the book is auditable per strategy."""
        if not bool(self._cfg.get("save_trades", True)):
            return
        from src.strategies.utils.blotter import write_trades_excel
        sleeve_trades = {n: self.results[n].trades for n in self.results}
        self.trades_path = self._context.paths["OUTPUT_DIR"] / "portfolio" / "trades.xlsx"
        write_trades_excel(sleeve_trades, self.trades_path)
        n_rows = {n: (0 if t is None else len(t)) for n, t in sleeve_trades.items()}
        self._log.info("Saved per-sleeve trade blotters -> %s (rows/sheet: %s)",
                       self.trades_path, n_rows)

    # ------------------------------------------------------------------ #
    def _inputs(self) -> PortfolioInputs:
        c = self._cfg
        return PortfolioInputs(
            capital=float(c.get("starting_capital", 1_000_000)),
            target_vol=float(c.get("sleeve_target_vol", c.get("portfolio_vol_target", 0.10))),
            start=pd.Timestamp(c.get("start")) if c.get("start") else None,
            end=pd.Timestamp(c.get("end")) if c.get("end") else None,
            fee_bps=float(c.get("fee_bps", 2.0)), spread_bps=float(c.get("spread_bps", 8.0)),
            risk_free_rate=float(c.get("risk_free_rate", 0.02)),
            analysis=bool(c.get("plot_analysis", True)))

    def _benchmark(self, index: pd.DatetimeIndex) -> pd.Series:
        """S&P proxy daily returns (equity_tr from macro_asset_prices) aligned to `index`."""
        df = self._context.store.load(MACRO_ASSET_PRICES_TABLE)
        if df is None or df.empty or "equity_tr" not in df.columns:
            return pd.Series(0.0, index=index)
        d = df.copy(); d["date"] = pd.to_datetime(d["date"])
        eq = d.sort_values("date").set_index("date")["equity_tr"].astype(float)
        return eq.pct_change(fill_method=None).reindex(index).fillna(0.0)

    def load_sleeves(self) -> None:
        inputs = self._inputs()
        names = [str(s) for s in self._cfg.get("sleeves", ["ls_equity", "long_book", "trend_cta"])]
        streams, self.results = {}, {}

        for n in names:
            if n not in STRATEGY_REGISTRY:
                self._log.warning("unknown sleeve '%s' — skipped", n); continue
            try:
                res = STRATEGY_REGISTRY[n](self._context, self._config).run(inputs)
            except Exception as e:                                  # noqa: BLE001
                self._log.warning("sleeve '%s' failed (%s) — skipped", n, e); continue
            if res.returns is not None and not res.returns.dropna().empty:
                streams[n] = res.returns.astype(float)
                self.results[n] = res
            else:
                self._log.warning("sleeve '%s' produced an empty return stream — skipped", n)
        
        self.requested_sleeves = names
        self.dropped_sleeves = [n for n in names if n not in streams]     # requested but no data
        if self.dropped_sleeves:
            self._log.warning("sleeves DROPPED (no data in window): %s", self.dropped_sleeves)
        if len(streams) < 2:
            raise RuntimeError(f"portfolio needs >=2 sleeves with data, got {list(streams)}")

        rets = pd.DataFrame(streams).sort_index()
        if inputs.start is not None:
            rets = rets[rets.index >= inputs.start]
        if inputs.end is not None:
            rets = rets[rets.index <= inputs.end]
        self.sleeve_rets = rets.dropna(how="all")
        self.benchmark = self._benchmark(self.sleeve_rets.index)
        cover = {n: f"{rets[n].dropna().index.min().date()}→{rets[n].dropna().index.max().date()}"
                 for n in rets.columns}
        self._log.info("Portfolio sleeves %s | window %s→%s | coverage %s",
                       list(rets.columns), self.sleeve_rets.index.min().date(),
                       self.sleeve_rets.index.max().date(), cover)

    # ------------------------------------------------------------------ #
    def blend(self) -> None:
        c = self._cfg
        self.weights = base_weights(
            self.sleeve_rets, int(c.get("vol_window", 63)), str(c.get("scheme", "erc")),
            int(c.get("rebalance_freq", 21)), cov_mode=str(c.get("cov_mode", "ewma")),
            cov_halflife=int(c.get("cov_halflife", 63)))
        self.blended = blend_to_vol_target(
            self.sleeve_rets, self.weights, float(c.get("portfolio_vol_target", 0.10)),
            int(c.get("vol_window", 63)), float(c.get("max_leverage", 2.0)))
        rf = float(c.get("risk_free_rate", 0.02))
        self.capital = float(c.get("starting_capital", 1_000_000))
        self.daily = daily_frame(self.blended["ret"], self.benchmark,
                                 turnover=pd.Series(0.0, index=self.blended.index),
                                 cost=pd.Series(0.0, index=self.blended.index), starting_capital=self.capital)
        self.metrics = compute_metrics(self.daily, rf_annual=rf)
        self.sleeve_metrics = {n: self.results[n].metrics for n in self.sleeve_rets.columns}

    # ------------------------------------------------------------------ #
    def report(self) -> None:
        c, m = self._cfg, self.metrics
        self._log.info("=== Portfolio (%s across sleeves, global vol-target %.0f%%) vs SP500 ===",
                       str(c.get("scheme", "erc")).upper(), float(c.get("portfolio_vol_target", 0.10)) * 100)
        rows = []
        for n in self.sleeve_rets.columns:
            sm = self.sleeve_metrics[n]
            w = float(self.weights[n].mean())
            rows.append({"strategy": n, "ann_%": round(sm["ann_return"] * 100, 1),
                         "vol_%": round(sm["ann_vol"] * 100, 1), "sharpe": round(sm["sharpe"], 2),
                         "maxDD_%": round(sm["max_drawdown"] * 100, 1), "avg_weight": round(w, 2),
                         "avg_capital": round(w * self.capital), "days": int(self.sleeve_rets[n].dropna().shape[0])})
        rows.append({"strategy": "PORTFOLIO", "ann_%": round(m["ann_return"] * 100, 1),
                     "vol_%": round(m["ann_vol"] * 100, 1), "sharpe": round(m["sharpe"], 2),
                     "maxDD_%": round(m["max_drawdown"] * 100, 1), "avg_weight": np.nan,
                     "avg_capital": round(self.capital), "days": int(m["days"])})
        rows.append({"strategy": "SP500 (hold)", "ann_%": round(m["spy_ann_return"] * 100, 1),
                     "vol_%": round(m["spy_ann_vol"] * 100, 1), "sharpe": round(m["spy_sharpe"], 2),
                     "maxDD_%": round(m["spy_max_drawdown"] * 100, 1), "avg_weight": np.nan,
                     "avg_capital": np.nan, "days": np.nan})
        self.summary = pd.DataFrame(rows)
        self.sleeve_corr = self.sleeve_rets.corr()
        self._log.info("--- per-strategy Sharpe vs global (+ avg $ allocation) ---\n%s",
                       self.summary.to_string(index=False))
        self._log.info("--- sleeve return correlation ---\n%s", self.sleeve_corr.round(2).to_string())
        self._log.info("avg leverage %.2f", float(self.blended["leverage"].mean()))
        out = self._context.paths["OUTPUT_DIR"] / "portfolio"
        out.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(out / "strategy_summary.csv", index=False)
        self.sleeve_corr.to_csv(out / "sleeve_correlation.csv")
        self.daily.to_parquet(out / "portfolio_daily.parquet")
        self.weights.to_parquet(out / "sleeve_weights.parquet")
        self.sleeve_rets.to_parquet(out / "sleeve_returns.parquet")

    # ------------------------------------------------------------------ #
    def plots(self) -> None:
        out = self._context.paths["OUTPUT_DIR"] / "portfolio"
        out.mkdir(parents=True, exist_ok=True)
        self._plot_equity(out / "portfolio_vs_sp.png")
        self._plot_weights(out / "sleeve_weights.png")
        if bool(self._cfg.get("plot_analysis", True)):
            from src.portfolio.analysis import analyze_portfolio
            a = analyze_portfolio(self.sleeve_rets, out / "analysis")
            self._log.info("Portfolio sleeve-correlation analysis: avg pairwise corr %.2f -> %s",
                           a["avg_pairwise_corr"], out / "analysis")
        self._log.info("Saved portfolio backtest outputs to %s", out)

    def _plot_equity(self, path) -> None:
        d = self.daily
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(d.index, d["portfolio_value"] / d["portfolio_value"].iloc[0],
                 label="Portfolio (blend)", lw=2.0, color="black")
        ax1.plot(d.index, d["spy_value"] / d["spy_value"].iloc[0],
                 label="SP500 (hold)", lw=1.3, color="#ff7f0e", ls="--")
        for n in self.sleeve_rets.columns:
            s = self.sleeve_rets[n].reindex(d.index).fillna(0.0)
            ax1.plot(d.index, (1 + s).cumprod(), lw=1.0, alpha=0.7,
                     color=_SLEEVE_COLORS.get(n), label=f"{n} (standalone)")
        ax1.set_yscale("log"); ax1.set_ylabel("Growth of $1 (log)"); ax1.legend(fontsize=8, ncol=2)
        ax1.set_title("Portfolio (3 strategies) vs SP500"); ax1.grid(True, alpha=0.3)
        eq = d["portfolio_value"].to_numpy(); pk = np.maximum.accumulate(eq)
        sp = d["spy_value"].to_numpy(); spk = np.maximum.accumulate(sp)
        ax2.fill_between(d.index, (eq - pk) / pk * 100, 0, alpha=0.4, color="black", label="Portfolio DD")
        ax2.plot(d.index, (sp - spk) / spk * 100, color="#ff7f0e", lw=1.0, ls="--", label="SP DD")
        ax2.set_ylabel("Drawdown (%)"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

    def _plot_weights(self, path) -> None:
        w = self.weights.clip(lower=0.0).resample("ME").mean()
        cols = list(w.columns)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.stackplot(w.index, *[w[c] for c in cols], labels=cols,
                     colors=[_SLEEVE_COLORS.get(c) for c in cols], alpha=0.85)
        ax.set_ylim(0, 1); ax.margins(x=0); ax.set_ylabel("Sleeve weight (monthly mean)")
        ax.set_title("Dynamic capital allocation across strategies (risk-parity/ERC)")
        ax.legend(loc="upper center", ncol=len(cols), fontsize=9)
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
