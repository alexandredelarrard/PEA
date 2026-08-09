"""
step_super_investors.py  (src/strategies/step_super_investors.py)
-------------------------------------------------------------------
"Smart-money" REPLICATION sleeve: rebuild the elite 13F cohort's (Dataroma superinvestors)
equity book from `sec13f_hr`, seed a fixed pot of capital across it, and then mirror every
buy/sell they disclose -- long-only, unlevered, net of fees. Self-contained; no dependency on
other strategy steps.

The sleeve is deliberately thin: `utils/superinvestors.py` turns raw 13F filings into the daily
(ticker, as_of) holdings + flow panel, `utils/replication.py` runs the mirror, and
`analysis/super_investors_analysis.py` draws it against SPY. This file only does IO and wiring.
"""

from __future__ import annotations
import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.strategies.base import Strategy, PortfolioInputs, StrategyResult
from src.strategies.utils.superinvestors import (
    _load_superinvestor_roster,
    _aggregate_superinvestors,
)
from src.strategies.utils.replication import replicate_superinvestors
from src.utils.risk_parity import series_metrics
from src.strategies.analysis.super_investors_analysis import (
    analyze_super_investors, analyze_super_investors_by_cik)

from src.constants.constants import SEC13F_TABLE

# A name the cohort has exited must be gone, not merely small. The only legitimate residual is
# a position that briefly cannot be sold (no price that day), so the bar is float noise, not a
# risk budget -- the bug this guards against ran to 94% of equity.
_ORPHAN_TOLERANCE = 1e-6

class SuperInvestorsStrategy(Strategy):
    name = "super_investors"
    config_key = "strategy_super_investors"

    def __init__(self, context: Context, config: DictConfig):
            super().__init__(context=context, config=config)

    def run(self) -> StrategyResult:

        inputs= PortfolioInputs(analysis=True)
        c = self.config if self.config_key in self._config else {}
        raw_funds, prices, end = self.load_raw()
        panel = _aggregate_superinvestors(raw_funds, end=end)

        res = replicate_superinvestors(
            panel, prices, capital=float(inputs.capital),
            fee_bps=float(c.get("fee_bps", inputs.fee_bps)),
            spread_bps=float(c.get("spread_bps", inputs.spread_bps)),
            start=inputs.start, end=inputs.end,
            execution_lag=int(c.get("execution_lag", 1)))

        ret, diag = res["returns"], res["diagnostics"]
        self._log.info("super_investors: seeded %s across %d names on %s; %d trading days, "
                       "max leverage %.4fx, min cash %s, fees %s, buys capped on %d days",
                       f"EUR {inputs.capital:,.0f}", diag["seed_names"],
                       diag["seed_date"].date(), len(ret), diag["max_leverage"],
                       f"EUR {diag['min_cash']:,.0f}", f"EUR {diag['total_cost_usd']:,.0f}",
                       diag["n_days_buy_capped"])
        # the sleeve's defining constraints -- assert them rather than trust them
        if diag["max_leverage"] > 1.0 + 1e-9 or diag["min_cash"] < -1e-6:
            raise RuntimeError(f"super_investors: leverage/cash invariant broken "
                               f"(max_leverage={diag['max_leverage']:.6f}, "
                               f"min_cash={diag['min_cash']:.2f})")
        # a replication may only hold what the cohort holds. Leverage/cash/short checks all
        # passed while the book sat 94% in a name the cohort had already exited, so this is the
        # one that actually catches a stale residual compounding into the result.
        if diag["max_orphan_weight"] > _ORPHAN_TOLERANCE:
            raise RuntimeError(
                f"super_investors: {diag['max_orphan_weight']:.2%} of equity held in names the "
                f"cohort does not hold (worst on {diag['orphan_date']}) -- stale residual.")

        extra = {"cash_weight": res["cash_weight"], **diag}
        if inputs.analysis:
            out_dir = self._context.paths["OUTPUT_DIR"] / self.name / "analysis"
            extra["analysis"] = analyze_super_investors(
                ret, self._benchmark_returns(prices, ret.index), res["weights"],
                out_dir, capital=float(inputs.capital))
            self._log.info("super_investors analysis -> %s", out_dir)

            if bool(c.get("per_cik_analysis", False)):
                # the pooled book averages away the fact that its managers disagree -- replay
                # each one as its own portfolio to see who actually carries the sleeve
                roster = _load_superinvestor_roster(self._context)
                per_cik = _aggregate_superinvestors(raw_funds, end=end, by_cik=True)
                spy = self._benchmark_returns(
                    prices, pd.DatetimeIndex(per_cik["as_of"].unique()).sort_values())
                summary = analyze_super_investors_by_cik(
                    per_cik, prices, spy, roster, out_dir / "per_cik",
                    capital=float(inputs.capital),
                    fee_bps=float(c.get("fee_bps", inputs.fee_bps)),
                    spread_bps=float(c.get("spread_bps", inputs.spread_bps)),
                    execution_lag=int(c.get("execution_lag", 1)),
                    start=inputs.start, end=inputs.end)
                extra["per_cik_summary"] = summary
                if not summary.empty:
                    self._log.info("super_investors per-cik: %d managers backtested, %d beat "
                                   "SPY on their own window -> %s", len(summary),
                                   int((summary["excess_ann"] > 0).sum()), out_dir / "per_cik")

        return StrategyResult(name=self.name, returns=ret,
                              metrics=series_metrics(ret, inputs.risk_free_rate),
                              positions=res["weights"], trades=res["trades"], extra=extra,
                              book_weights=res["weights"])

    def _benchmark_returns(self, prices: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
        """SPY daily returns on the sleeve's own calendar. SPY is fetched into `prices` as a
        market series (never into the equity universe), so it comes from the same read."""
        spy = prices[prices["ticker"] == self._config.build_cube.market_ticker].copy()
        spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()
        s = spy.sort_values("date").set_index("date")["close"].astype(float)
        return s.reindex(s.index.union(index)).ffill().reindex(index).pct_change(
            fill_method=None).fillna(0.0)

    def load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp | None]:
        """The roster managers' raw 13F rows + close prices (equity universe AND the SPY
        benchmark row), plus the date to carry holdings forward to.

        Returns the RAW filings rather than an aggregated panel because the caller aggregates
        the same rows twice -- once pooled, once `by_cik` -- and `sec13f_hr` is a 21.7M-row
        table, so reading it (and the whole `prices` table) a second time is the expensive part."""
        store = self._context.store
        _FUNDS_COLS = ["cik", "ticker", "filing_date", "period", "shares", "value_usd"]

        roster_ciks = _load_superinvestor_roster(self._context)
        if not roster_ciks:
            raise RuntimeError("super_investors: superinvestors roster resolved to no manager "
                               "-- check data/superinvestors/superinvestors.json.")
        df_funds = store.load(SEC13F_TABLE, columns=_FUNDS_COLS,
                              where={"cik": set(roster_ciks.keys())})
        prices = store.load("prices", columns=["date", "ticker", "close"])

        # carry the holdings forward to the last price date, so the book is marked to market
        # right up to today rather than stopping at the most recent 13F filing
        end = pd.to_datetime(prices["date"]).max() if not prices.empty else None
        return df_funds, prices, end
