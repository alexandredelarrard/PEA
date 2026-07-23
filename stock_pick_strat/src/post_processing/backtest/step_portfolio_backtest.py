"""
step_portfolio_backtest.py  (src/post_processing/backtest/step_portfolio_backtest.py)
-------------------------------------------------------------------------------------
StepPortfolioBacktest — the UNIFIED backtest that blends the three strategy sleeves into one
book. It builds each configured sleeve (`ls_equity`, `long_book`, `trend_cta`), collects each
one's daily NET return via the common `Strategy.returns()` interface, then:
  1. MIX  — weight the sleeves' return streams by risk: inverse-vol (EWMA) or **ERC**
            (correlation-aware) across sleeves, point-in-time, NaN-aware (L/S only joins once
            it has enough out-of-sample history);
  2. SIZE — one global leverage to hit `portfolio_vol_target` (capped by `max_leverage`).
Reports return / vol / **Sharpe per strategy** vs the **global** portfolio, the sleeve
cross-correlation, and the blend weights; saves an equity curve, per-sleeve curves and a
sleeve-weight stacked area. Config: `backtest.portfolio_backtest`.
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
from src.post_processing.strategies import STRATEGY_REGISTRY
from src.post_processing.utils.metrics import compute_metrics
from src.post_processing.utils.strategies_alloc import base_weights, series_metrics, daily_frame
from src.post_processing.utils.strategies_blend import blend_to_vol_target

_SLEEVE_COLORS = {"ls_equity": "#1f77b4", "long_book": "#2ca02c", "trend_cta": "#d62728"}


class StepPortfolioBacktest(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.backtest.portfolio_backtest

    def run(self) -> None:
        self.load_sleeves()
        self.blend()
        self.report()
        self.plots()

    # ------------------------------------------------------------------ #
    def _benchmark(self, index: pd.DatetimeIndex) -> pd.Series:
        """S&P proxy daily returns (equity_tr from macro_asset_prices) aligned to `index`."""
        df = self._context.store.load(MACRO_ASSET_PRICES_TABLE)
        if df is None or df.empty or "equity_tr" not in df.columns:
            return pd.Series(0.0, index=index)
        d = df.copy(); d["date"] = pd.to_datetime(d["date"])
        eq = d.sort_values("date").set_index("date")["equity_tr"].astype(float)
        return eq.pct_change(fill_method=None).reindex(index).fillna(0.0)

    def load_sleeves(self) -> None:
        names = [str(s) for s in self._cfg.get("sleeves", ["ls_equity", "long_book", "trend_cta"])]
        streams: dict[str, pd.Series] = {}
        self._strats = {}
        for n in names:
            if n not in STRATEGY_REGISTRY:
                self._log.warning("unknown sleeve '%s' — skipped", n); continue
            try:
                strat = STRATEGY_REGISTRY[n](self._context, self._config)
                s = strat.returns()
            except Exception as e:                                  # noqa: BLE001
                self._log.warning("sleeve '%s' failed to produce returns (%s) — skipped", n, e)
                continue
            if s is not None and not s.dropna().empty:
                streams[n] = s.astype(float)
                self._strats[n] = strat
            else:
                self._log.warning("sleeve '%s' produced an empty return stream — skipped", n)
        if len(streams) < 2:
            raise RuntimeError(f"portfolio backtest needs >=2 sleeves, got {list(streams)}")

        rets = pd.DataFrame(streams).sort_index()
        start = pd.Timestamp(self._cfg.get("start")) if self._cfg.get("start") else None
        end = pd.Timestamp(self._cfg.get("end")) if self._cfg.get("end") else None
        if start is not None:
            rets = rets[rets.index >= start]
        if end is not None:
            rets = rets[rets.index <= end]
        rets = rets.dropna(how="all")
        self.sleeve_rets = rets
        self.benchmark = self._benchmark(rets.index)
        cover = {n: f"{rets[n].dropna().index.min().date()}→{rets[n].dropna().index.max().date()}"
                 for n in rets.columns}
        self._log.info("Portfolio sleeves %s | window %s→%s | coverage %s",
                       list(rets.columns), rets.index.min().date(), rets.index.max().date(), cover)

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
        cap = float(c.get("starting_capital", 1_000_000))
        self.daily = daily_frame(self.blended["ret"], self.benchmark,
                                 turnover=pd.Series(0.0, index=self.blended.index),
                                 cost=pd.Series(0.0, index=self.blended.index), starting_capital=cap)
        self.metrics = compute_metrics(self.daily, rf_annual=rf)
        self.sleeve_metrics = {n: series_metrics(self.sleeve_rets[n], rf) for n in self.sleeve_rets.columns}

    # ------------------------------------------------------------------ #
    def report(self) -> None:
        c = self._cfg
        m = self.metrics
        self._log.info("=== Portfolio (3-strategy blend: %s across sleeves, vol-target %.0f%%) vs SP500 ===",
                       str(c.get("scheme", "erc")).upper(), float(c.get("portfolio_vol_target", 0.10)) * 100)
        # per-strategy standalone metrics + avg blend weight
        rows = []
        for n in self.sleeve_rets.columns:
            sm = self.sleeve_metrics[n]
            rows.append({"strategy": n, "ann_%": round(sm["ann_return"] * 100, 1),
                         "vol_%": round(sm["ann_vol"] * 100, 1), "sharpe": round(sm["sharpe"], 2),
                         "maxDD_%": round(sm["max_drawdown"] * 100, 1),
                         "avg_weight": round(float(self.weights[n].mean()), 2),
                         "days": int(self.sleeve_rets[n].dropna().shape[0])})
        rows.append({"strategy": "PORTFOLIO (blend)", "ann_%": round(m["ann_return"] * 100, 1),
                     "vol_%": round(m["ann_vol"] * 100, 1), "sharpe": round(m["sharpe"], 2),
                     "maxDD_%": round(m["max_drawdown"] * 100, 1), "avg_weight": np.nan,
                     "days": int(m["days"])})
        rows.append({"strategy": "SP500", "ann_%": round(m["spy_ann_return"] * 100, 1),
                     "vol_%": round(m["spy_ann_vol"] * 100, 1), "sharpe": round(m["spy_sharpe"], 2),
                     "maxDD_%": round(m["spy_max_drawdown"] * 100, 1), "avg_weight": np.nan, "days": np.nan})
        self.summary = pd.DataFrame(rows)
        self._log.info("--- per-strategy Sharpe vs global ---\n%s", self.summary.to_string(index=False))
        self.sleeve_corr = self.sleeve_rets.corr()
        self._log.info("--- sleeve return correlation (independence check) ---\n%s",
                       self.sleeve_corr.round(2).to_string())
        self._log.info("Avg sleeve blend weights: %s  | avg leverage %.2f",
                       {n: round(float(self.weights[n].mean()), 2) for n in self.weights.columns},
                       float(self.blended["leverage"].mean()))
        out = self._context.paths["OUTPUT_DIR"] / "portfolio_backtest"
        out.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(out / "strategy_summary.csv", index=False)
        self.sleeve_corr.to_csv(out / "sleeve_correlation.csv")
        self.daily.to_parquet(out / "portfolio_daily.parquet")
        self.weights.to_parquet(out / "sleeve_weights.parquet")

    # ------------------------------------------------------------------ #
    def plots(self) -> None:
        out = self._context.paths["OUTPUT_DIR"] / "portfolio_backtest"
        out.mkdir(parents=True, exist_ok=True)
        self._plot_equity(out / "portfolio_vs_sp.png")
        self._plot_weights(out / "sleeve_weights.png")
        self._log.info("Saved portfolio backtest outputs to %s", out)

    def _plot_equity(self, path) -> None:
        d = self.daily
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(d.index, d["portfolio_value"] / d["portfolio_value"].iloc[0],
                 label="Portfolio (blend)", lw=2.0, color="black")
        ax1.plot(d.index, d["spy_value"] / d["spy_value"].iloc[0],
                 label="SP500", lw=1.3, color="#ff7f0e", ls="--")
        # each sleeve rebased (growth of $1 of the raw sleeve stream over the window)
        for n in self.sleeve_rets.columns:
            s = self.sleeve_rets[n].reindex(d.index).fillna(0.0)
            ax1.plot(d.index, (1 + s).cumprod(), lw=1.0, alpha=0.7,
                     color=_SLEEVE_COLORS.get(n), label=f"{n} (standalone)")
        ax1.set_yscale("log"); ax1.set_ylabel("Growth of $1 (log)"); ax1.legend(fontsize=8, ncol=2)
        ax1.set_title("3-strategy portfolio vs SP500 (+ standalone sleeves)"); ax1.grid(True, alpha=0.3)
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
        ax.set_ylim(0, 1); ax.margins(x=0); ax.set_ylabel("Sleeve blend weight (monthly mean)")
        ax.set_title("Strategy blend weights over time (risk-parity/ERC across sleeves)")
        ax.legend(loc="upper center", ncol=len(cols), fontsize=9)
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
