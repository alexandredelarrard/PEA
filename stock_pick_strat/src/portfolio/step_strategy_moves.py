"""
step_strategy_moves.py  (src/portfolio/step_strategy_moves.py)
--------------------------------------------------------------
StepStrategyMoves — the LIVE TRADING LEDGER. Answers "if I actually ran this portfolio,
what would I have traded, and what did each position earn?" and persists it to the
`strategy` table.

Flow (daily, after the prediction step):
  1. run the configured sleeves and blend them exactly as `StepPortfolio` does -> per-sleeve
     dynamic ERC/risk-parity weights + the one global leverage that hits the vol target;
  2. RE-SIZE each sleeve's traded weight panel by its own `erc_weight(t) * leverage(t)`, so
     the notional traded is the real dollars the portfolio would deploy in that sleeve rather
     than the full `starting_capital` each sleeve is backtested on standalone;
  3. rebuild the share-accurate blotter on the re-sized panel (`trade_blotter`);
  4. FIFO-match every move into round trips (`positions.round_trip_ledger`) so each row
     carries its entry price, its exit price and its realized P&L;
  5. upsert the whole ledger to `strategy` on (trading_day, sleeve, ticker).

Why re-size the PANEL rather than scale the blotter's dollars: share counts are
`weight * capital / price`, and the capital multiplier varies day by day, so the number of
shares TRADED (a difference of share counts) is not proportional to the multiplier. Scaling
the resulting $ figures would silently misreport both the shares and the fees.

Why the full history every run rather than only today's moves: the ledger is a book of
positions, not a log of events. A BUY row written weeks ago only learns its exit price and
P&L on the day the position closes, so past rows must be rewritten -- which also makes the
step self-healing: a missed day, or a retrained model, corrects itself on the next run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import STRATEGY_TABLE
from src.context import Context
from src.portfolio.step_portfolio import StepPortfolio
from src.strategies import STRATEGY_REGISTRY
from src.strategies.utils.blotter import trade_blotter
from src.strategies.utils.positions import LEDGER_COLUMNS, round_trip_ledger
from src.utils.step import Step


class StepStrategyMoves(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.portfolio

    def run(self) -> pd.DataFrame:
        self.run_time = pd.Timestamp.now().floor("s")
        self._blend()
        ledger = self._ledger()
        self._save(ledger)
        self._report(ledger)
        return ledger

    # ------------------------------------------------------------------ #
    def _blend(self) -> None:
        """Sleeve returns + the ERC weights and global leverage, via StepPortfolio.

        Reuses the portfolio step rather than re-deriving the blend: the $ allocation in this
        ledger MUST be the same allocation the backtest reports, or the two disagree about what
        the strategy actually does."""
        pf = StepPortfolio(context=self._context, config=self._config)
        pf.load_sleeves()
        pf.blend()
        self._portfolio = pf
        self.capital = float(self._cfg.get("starting_capital", 1_000_000))
        self._log.info("Sleeves %s blended (%s, vol target %.0f%%); avg leverage %.2f",
                       list(pf.sleeve_rets.columns), str(self._cfg.get("scheme", "erc")).upper(),
                       float(self._cfg.get("portfolio_vol_target", 0.10)) * 100,
                       float(pf.blended["leverage"].mean()))

    def _capital_factor(self, sleeve: str, index: pd.DatetimeIndex) -> pd.Series:
        """`erc_weight(t) * leverage(t)` for one sleeve, aligned to `index`.

        Forward-filled then zero-filled: before a sleeve joins the blend it gets no capital
        (weight 0 -> no trades), which is the honest answer for a sleeve whose history has
        not started yet (the L/S sleeve only exists out-of-sample)."""
        pf = self._portfolio
        w = pf.weights[sleeve] if sleeve in pf.weights.columns else pd.Series(dtype=float)
        lev = pf.blended["leverage"]
        f = (w.reindex(index).ffill() * lev.reindex(index).ffill()).fillna(0.0)
        return f.clip(lower=0.0)

    def _sleeve_ledger(self, sleeve: str) -> pd.DataFrame | None:
        """One sleeve's ledger: re-size its traded panel to the portfolio's $ allocation,
        rebuild the blotter, then FIFO-match it into round trips."""
        res = self._portfolio.results.get(sleeve)
        if res is None or res.book_weights is None or res.book_weights.empty:
            self._log.warning("sleeve '%s': no traded weight panel -> no ledger rows", sleeve)
            return None
        book = res.book_weights.sort_index()
        factor = self._capital_factor(sleeve, book.index)
        scaled = book.mul(factor, axis=0)
        if not np.isfinite(scaled.to_numpy(dtype="float64")).any() or (scaled.abs().sum().sum() == 0):
            self._log.warning("sleeve '%s': allocation is zero over the whole window -> skipped", sleeve)
            return None

        cfg = self._sleeve_cfg(sleeve)
        trades = trade_blotter(
            scaled, self.capital,
            float(cfg.get("fee_bps", self._cfg.get("fee_bps", 2.0))),
            float(cfg.get("spread_bps", self._cfg.get("spread_bps", 8.0))),
            sleeve, prices=res.book_prices,
            floor_usd=float(self._cfg.get("trade_floor_usd", 0.0)))
        led = round_trip_ledger(trades, run_time=self.run_time)
        self._log.info("sleeve '%s': %d move(s) over %d day(s); avg allocation $%.0f",
                       sleeve, len(led), led["trading_day"].nunique() if not led.empty else 0,
                       float(factor.mean()) * self.capital)
        return led

    def _sleeve_cfg(self, sleeve: str):
        """That sleeve's own config block, so the ledger charges the SAME fee/spread the sleeve
        traded at. Resolved via the strategy class's `config_key` -- the sleeve name does not map
        mechanically to the key ('ls_equity' -> 'strategy_ls')."""
        cls = STRATEGY_REGISTRY.get(sleeve)
        return (self._config.get(cls.config_key) or {}) if cls is not None else {}

    def _ledger(self) -> pd.DataFrame:
        parts = [led for s in self._portfolio.sleeve_rets.columns
                 if (led := self._sleeve_ledger(str(s))) is not None and not led.empty]
        if not parts:
            raise RuntimeError("No sleeve produced any trading move -> nothing to write to "
                               f"'{STRATEGY_TABLE}'. Check the portfolio window / sleeve data.")
        led = pd.concat(parts, ignore_index=True)
        return led.sort_values(["trading_day", "sleeve", "ticker"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def _save(self, ledger: pd.DataFrame) -> None:
        """Upsert on (trading_day, sleeve, ticker) so a re-run refreshes the exit price and P&L
        of positions that have closed since, without duplicating the moves."""
        if not self._context.save:
            self._log.info("save=False -> ledger not persisted (%d rows computed)", len(ledger))
            return
        saved = self._context.store.save(STRATEGY_TABLE, ledger[LEDGER_COLUMNS])
        self._log.info("Upserted %d row(s) to '%s'", saved, STRATEGY_TABLE)

    def _report(self, ledger: pd.DataFrame) -> None:
        closed = ledger[ledger["pnl"].notna()]
        open_rows = ledger[ledger["pnl"].isna() & ledger["price_sold"].isna()]
        last_day = ledger["trading_day"].max()
        today = ledger[ledger["trading_day"] == last_day]
        by_sleeve = (ledger.groupby("sleeve")
                     .agg(moves=("ticker", "size"),
                          traded_usd=("amount_invested", "sum"),
                          fees=("fee", "sum"),
                          realized_pnl=("pnl", "sum"))
                     .round(0))
        self._log.info("--- trading ledger: %d move(s), %s -> %s ---", len(ledger),
                       pd.Timestamp(ledger["trading_day"].min()).date(),
                       pd.Timestamp(last_day).date())
        self._log.info("\n%s", by_sleeve.to_string())
        self._log.info("closed round trips %d (realized P&L $%.0f net of $%.0f fees) | still open %d",
                       len(closed), float(ledger["pnl"].sum(min_count=1) or 0.0),
                       float(ledger["fee"].sum()), len(open_rows))
        self._log.info("LATEST trading day %s -> %d move(s) to place:\n%s",
                       pd.Timestamp(last_day).date(), len(today),
                       today[["sleeve", "ticker", "side", "shares", "price",
                              "amount_invested", "fee"]].to_string(index=False))
