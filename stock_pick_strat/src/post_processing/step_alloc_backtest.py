"""
step_alloc_backtest.py  (src/post_processing/step_alloc_backtest.py)
--------------------------------------------------------------------
Multi-ASSET ALLOCATION backtest (SEPARATE from the equity L/S StepBacktest).

Allocates a long-only % book across the risky assets in the long-history
`macro_asset_prices` table (equity / gold / 10Y bond total-return / FX) with CASH as
the residual/funding leg, using the layered engine in
`src/post_processing/utils/strategies_alloc.py`:
  1. risk-parity MIX  (erc | inverse_vol),
  2. long-only TREND overlay (scales an asset toward cash when it rolls over),
  3. one global leverage to hit `portfolio_vol_target`.
Fees are charged on weight turnover (conservative default ~10 bps one-way).

Reports return / vol / Sharpe / maxDD PER ASSET and for the global portfolio (+ per
market-regime), and saves two plots: portfolio-vs-SP equity curve and the per-asset
weight-evolution stacked area. Config: `backtest.asset_allocation` in configs/backtest.yml.
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
from src.post_processing.utils.metrics import compute_metrics
from src.post_processing.utils.strategies_alloc import (
    asset_returns_from_macro,
    allocation_backtest,
    per_asset_metrics,
    series_metrics,
    daily_frame,
    sweep_trend_params,
    CASH,
)

# labeled market regimes for per-period reporting (weight/return behaviour by era)
REGIMES: dict[str, tuple[str, str]] = {
    "dotcom_bust": ("2000-01-01", "2002-12-31"),
    "pre_gfc_bull": ("2003-01-01", "2007-06-30"),
    "gfc": ("2007-07-01", "2009-06-30"),
    "euro_crisis": ("2010-01-01", "2012-12-31"),
    "qe_bull": ("2013-01-01", "2019-12-31"),
    "covid": ("2020-01-01", "2020-06-30"),
    "inflation_2022": ("2022-01-01", "2022-12-31"),
    "recent": ("2023-01-01", "2100-01-01"),
}
_ASSET_COLORS = {"equity": "#1f77b4", "gold": "#d4af37", "bond": "#2ca02c",
                 "energy": "#d62728", "fx": "#9467bd", CASH: "#b0b0b0"}
_ASSET_ORDER = ["equity", "gold", "energy", "bond", "fx", CASH]


class StepAllocationBacktest(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.backtest.asset_allocation

    def run(self) -> None:
        self.load_assets()
        self.backtest()
        self.report()
        self.plots()

    # ------------------------------------------------------------------ #
    def load_assets(self) -> None:
        df = self._context.store.load(MACRO_ASSET_PRICES_TABLE)
        if df is None or df.empty:
            raise RuntimeError(
                f"Table '{MACRO_ASSET_PRICES_TABLE}' is empty — run extraction "
                "(fetch_macro_assets) first to populate the long-history FRED series.")
        rets, cash = asset_returns_from_macro(df, include_fx=bool(self._cfg.get("include_fx", True)))
        # VIX is a regime SIGNAL column (not an asset) — carry it for the risk-on score
        vix = None
        if "vix" in df.columns:
            d = df.copy(); d["date"] = pd.to_datetime(d["date"])
            vix = d.sort_values("date").set_index("date")["vix"].astype(float)

        start = pd.Timestamp(self._cfg.get("start")) if self._cfg.get("start") else None
        end = pd.Timestamp(self._cfg.get("end")) if self._cfg.get("end") else None
        if start is not None:
            rets, cash = rets[rets.index >= start], cash[cash.index >= start]
        if end is not None:
            rets, cash = rets[rets.index <= end], cash[cash.index <= end]

        if bool(self._cfg.get("include_ls", False)):
            self._maybe_add_ls(rets)

        self.rets, self.cash = rets, cash
        self.vix = vix.reindex(rets.index) if vix is not None else None
        self.benchmark = rets["equity"] if "equity" in rets.columns else cash * 0.0
        self._log.info("Allocation universe: %s + cash | %s -> %s (%d days)",
                       list(rets.columns), rets.index.min().date(), rets.index.max().date(), len(rets))

    def _maybe_add_ls(self, rets: pd.DataFrame) -> None:
        """Add the equity L/S alpha daily return stream as a 5th risky sleeve (from ~2012),
        read from the saved L/S backtest artifact if present. Best-effort — a missing file
        just means the allocation runs on the market assets only (the since-2000 deliverable)."""
        try:
            p = self._context.paths["OUTPUT_DIR"] / "backtest" / "backtest_daily.parquet"
            if p.exists():
                d = pd.read_parquet(p)
                d.index = pd.to_datetime(d.index)
                rets["ls"] = d["net_ret"].reindex(rets.index)
                self._log.info("Included L/S alpha sleeve from %s (%d overlapping days)",
                               p, int(rets["ls"].notna().sum()))
            else:
                self._log.warning("include_ls=true but %s not found — running market assets only.", p)
        except Exception as e:                                          # noqa: BLE001
            self._log.warning("could not load L/S sleeve (%s) — market assets only.", e)

    # ------------------------------------------------------------------ #
    def _bt_kwargs(self) -> dict:
        c = self._cfg
        return dict(
            scheme=str(c.get("scheme", "erc")), vol_window=int(c.get("vol_window", 63)),
            rebalance_freq=int(c.get("rebalance_freq", 21)),
            trend_enabled=bool(c.get("trend_enabled", True)),
            trend_lookbacks=tuple(int(x) for x in c.get("trend_lookbacks", [63, 126, 252])),
            trend_scheme=str(c.get("trend_scheme", "linear")),
            trend_floor=float(c.get("trend_floor", 0.0)),
            trend_vol_window=int(c.get("trend_vol_window", 63)),
            trend_cap=float(c.get("trend_cap", 2.0)),
            portfolio_vol_target=float(c.get("portfolio_vol_target", 0.10)),
            max_leverage=float(c.get("max_leverage", 2.0)),
            fee_bps=float(c.get("fee_bps", 2.0)), spread_bps=float(c.get("spread_bps", 8.0)),
            cov_mode=str(c.get("cov_mode", "std")), cov_halflife=int(c.get("cov_halflife", 42)),
            vol_mode=str(c.get("vol_mode", "std")), lever_on=str(c.get("lever_on", "scaled")),
            risk_on=bool(c.get("risk_on_tilt", False)),
            vix=(self.vix if bool(c.get("use_vix", False)) else None),
            offensive=tuple(c.get("offensive", ["equity", "energy"])),
            off_share_range=(float(c.get("off_share_min", 0.15)), float(c.get("off_share_max", 0.85))),
            lev_responsive=bool(c.get("lev_responsive", False)),
            lev_min=float(c.get("lev_min", 1.0)), lev_max=float(c.get("lev_max", 2.0)))

    def backtest(self) -> None:
        c = self._cfg
        rf = float(c.get("risk_free_rate", 0.02))
        self.result = allocation_backtest(self.rets, self.cash, **self._bt_kwargs())
        self.daily = daily_frame(self.result["net_ret"], self.benchmark,
                                 self.result["turnover"], self.result["cost"],
                                 float(c.get("starting_capital", 1_000_000)))
        self.metrics = compute_metrics(self.daily, rf_annual=rf)
        self.asset_metrics = per_asset_metrics(self.rets, self.cash, rf_annual=rf)

    # ------------------------------------------------------------------ #
    def report(self) -> None:
        c = self._cfg
        rf = float(c.get("risk_free_rate", 0.02))
        m = self.metrics
        self._log.info("=== Multi-asset allocation (%s + trend=%s, vol-target %.0f%%) vs SP500 ===",
                       str(c.get("scheme", "erc")).upper(), bool(c.get("trend_enabled", True)),
                       float(c.get("portfolio_vol_target", 0.10)) * 100)
        self._log.info("Portfolio: total %.1f%%  ann %.1f%%  vol %.1f%%  Sharpe %.2f  maxDD %.1f%%  "
                       "avg lev %.2f  avg cash %.0f%%",
                       m["total_return"]*100, m["ann_return"]*100, m["ann_vol"]*100, m["sharpe"],
                       m["max_drawdown"]*100, float(self.result["leverage"].mean()),
                       float(self.result["alloc_cash"].mean())*100)
        self._log.info("SP500 (SPY, total return): total %.1f%%  ann %.1f%%  vol %.1f%%  Sharpe %.2f  maxDD %.1f%%",
                       m["spy_total_return"]*100, m["spy_ann_return"]*100, m["spy_ann_vol"]*100,
                       m["spy_sharpe"], m["spy_max_drawdown"]*100)

        pa = self.asset_metrics.copy()
        pa_fmt = pa.assign(ann_return=(pa["ann_return"]*100).round(1),
                           ann_vol=(pa["ann_vol"]*100).round(1),
                           sharpe=pa["sharpe"].round(2),
                           max_drawdown=(pa["max_drawdown"]*100).round(1))
        self._log.info("--- Per-asset (standalone buy&hold) return%%/vol%%/Sharpe/maxDD%% ---\n%s",
                       pa_fmt.to_string())

        # per-regime portfolio vs SP
        rows = []
        for name, (s, e) in REGIMES.items():
            mask = (self.daily.index >= pd.Timestamp(s)) & (self.daily.index <= pd.Timestamp(e))
            if mask.sum() < 20:
                continue
            pm = series_metrics(self.daily.loc[mask, "net_ret"], rf)
            bm = series_metrics(self.daily.loc[mask, "spy_value"].pct_change().fillna(0.0), rf)
            rows.append({"regime": name, "days": int(mask.sum()),
                         "port_ann_%": round(pm["ann_return"]*100, 1),
                         "port_vol_%": round(pm["ann_vol"]*100, 1),
                         "port_sharpe": round(pm["sharpe"], 2),
                         "port_maxDD_%": round(pm["max_drawdown"]*100, 1),
                         "sp_ann_%": round(bm["ann_return"]*100, 1),
                         "sp_maxDD_%": round(bm["max_drawdown"]*100, 1)})
        self.regime_table = pd.DataFrame(rows)
        if not self.regime_table.empty:
            self._log.info("--- Portfolio vs SP by regime ---\n%s",
                           self.regime_table.to_string(index=False))

        out = self._context.paths["OUTPUT_DIR"] / "allocation"
        out.mkdir(parents=True, exist_ok=True)
        pa.to_csv(out / "per_asset_metrics.csv")
        pd.DataFrame([m]).to_csv(out / "portfolio_metrics.csv", index=False)
        self.regime_table.to_csv(out / "regime_metrics.csv", index=False)
        self.daily.to_parquet(out / "allocation_daily.parquet")
        self.result["alloc"].assign(**{CASH: self.result["alloc_cash"]}).to_parquet(
            out / "allocation_weights.parquet")

    # ------------------------------------------------------------------ #
    def plots(self) -> None:
        out = self._context.paths["OUTPUT_DIR"] / "allocation"
        out.mkdir(parents=True, exist_ok=True)
        self._plot_equity(out / "portfolio_vs_sp.png")
        self._plot_weights(out / "allocation_weights.png")
        self._log.info("Saved allocation plots + artifacts to %s", out)

    def _plot_equity(self, path) -> None:
        d = self.daily
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(d.index, d["portfolio_value"] / d["portfolio_value"].iloc[0],
                 label=f"Allocation ({str(self._cfg.get('scheme','erc')).upper()}+trend)",
                 lw=1.8, color="#1f77b4")
        ax1.plot(d.index, d["spy_value"] / d["spy_value"].iloc[0],
                 label="SP500 (SPY, total return)", lw=1.5, color="#ff7f0e", ls="--")
        ax1.set_yscale("log"); ax1.set_ylabel("Growth of $1 (log)")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax1.set_title("Multi-asset allocation vs SP500")
        eq = d["portfolio_value"].to_numpy(); pk = np.maximum.accumulate(eq)
        sp = d["spy_value"].to_numpy(); spk = np.maximum.accumulate(sp)
        ax2.fill_between(d.index, (eq-pk)/pk*100, 0, alpha=0.4, color="#1f77b4", label="Allocation DD")
        ax2.plot(d.index, (sp-spk)/spk*100, color="#ff7f0e", lw=1.0, ls="--", label="SP DD")
        ax2.set_ylabel("Drawdown (%)"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

    def _plot_weights(self, path) -> None:
        alloc = self.result["alloc"].copy()
        alloc[CASH] = self.result["alloc_cash"]
        alloc = alloc.clip(lower=0.0)                       # display: hide tiny negatives
        # MONTHLY mean smooths the binary-trend daily on/off flips into a readable mix
        m = alloc.resample("ME").mean()
        cols = [c for c in _ASSET_ORDER if c in m.columns]
        colors = [_ASSET_COLORS.get(c, None) for c in cols]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # top: stacked-area composition (sums to 1)
        ax1.stackplot(m.index, *[m[c] for c in cols], labels=cols, colors=colors, alpha=0.85)
        ax1.set_ylim(0, 1); ax1.margins(x=0)
        ax1.set_ylabel("Weight (stacked, pre-leverage)")
        ax1.set_title("Allocation weight by asset over time — monthly mean "
                      "(trend moves risk into cash in downtrends)")
        ax1.legend(loc="upper center", ncol=len(cols), fontsize=8)
        # bottom: per-asset lines so each class is individually legible
        for c, col in zip(cols, colors):
            ax2.plot(m.index, m[c], label=c, color=col, lw=1.4)
        ax2.set_ylim(0, 1); ax2.margins(x=0)
        ax2.set_ylabel("Weight per asset"); ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", ncol=len(cols), fontsize=8)
        for (s, e) in REGIMES.values():                     # light regime dividers on both
            xs = pd.Timestamp(s)
            if m.index.min() <= xs <= m.index.max():
                ax1.axvline(xs, color="k", lw=0.5, alpha=0.2)
                ax2.axvline(xs, color="k", lw=0.5, alpha=0.2)
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

    # ------------------------------------------------------------------ #
    def sweep(self, grid: list[dict]) -> pd.DataFrame:
        """Trend-param sweep on the loaded assets (call after load_assets) — used to CHOOSE
        the trend defaults from evidence. Returns the ann/vol/Sharpe/maxDD comparison table."""
        base = self._bt_kwargs()
        rf = float(self._cfg.get("risk_free_rate", 0.02))
        return sweep_trend_params(self.rets, self.cash, grid, base, rf_annual=rf)
