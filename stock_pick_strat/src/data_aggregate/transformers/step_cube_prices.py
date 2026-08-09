"""
step_cube_prices.py  (src/data_aggregate/transformers/step_cube_prices.py)
-----------------------------------------------------------------------
The ONLY step that reads the raw `prices` table, the ONLY one that pivots it, and the ONLY
one that computes peer sector returns. Everything downstream reads the two part tables it
writes (`cube_part_prices`, `cube_part_market`) through
`utils/common/price_frames.load_price_frames`, projected to the fields it needs.

That prologue -- load ~1.9M rows, pivot to five wide frames, filter to the trading
calendar, restrict to the universe, then a 490-iteration Python loop for the peer sector
returns -- used to run once per DAG task, fourteen times per build, at ~400 MB peak each.

MEMORY DISCIPLINE (the invariant that makes seven sequential sub-steps safe): every heavy
frame is LOCAL to `run()`. Nothing is stashed on `self`, so when `run()` returns its
locals are collected and the next sub-step starts from a clean slate.
"""

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common import data_utils as du
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, plan_window, write_part
from src.data_aggregate.utils.common.part_io import PartStore
from src.data_aggregate.utils.common.parts import PART_BY_NAME
from src.data_aggregate.utils.common.peers_io import load_peers
from src.data_aggregate.utils.common.price_frames import frames_to_long, universe_columns
from src.data_peers.utils.sector_peers import compute_sector_returns
from src.data_aggregate.utils.common.price_frames import load_trading_calendar

from src.utils.step import Step
from src.utils.universe import load_universe_tickers
from src.constants.constants import (PRICES_TABLE, UNIVERSE_TABLE,
CUBE_PART_MARKET, CUBE_PART_PRICES)


class StepCubePrices(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube
        self._part = PART_BY_NAME[CUBE_PART_PRICES]
        self._market_ticker = str(self._cfg.market_ticker)
        self._other_tickers = tuple(config.data_extract.get("other_tickers", ()) or ())
        self._parts = PartStore(context.store, self._log)
        self._tickers = load_universe_tickers(context)
        self._log.info(f"Ticker universe: {len(self._tickers)} tickers from {UNIVERSE_TABLE}")

    def run(self, full: bool = False) -> None:
        window = self._plan_window(full)
        raw = self._load_prices(window.since)
        wide = self._pivot_fields(raw)

        # filter wide on trading days with close value 
        days_index = self._trading_calendar(wide["close"])
        wide = self._on_calendar(wide, days_index)

        # get deltas
        returns =  self._daily_returns(wide["close"])
        market = self._market_frames(wide["close"], returns)
        peers = self._peers()
        universe = self._universe_frames(wide, returns, peers)
        del raw, wide, returns

        # save it all
        n = self._persist(universe, market, window)
        if n == COLUMNS_CHANGED:                      # schema drift -> one clean full rebuild
            return self.run(full=True)

    # ---- window ---- #
    def _plan_window(self, full: bool):
        """Warm-up 260 trading days. `ret` needs only one prior day (and is persisted, so
        the trailing recompute is exact), but `get_trading_days`'s interior-calendar-hole
        warning is a diagnostic over history and wants a year of context."""
        idx = None
        if self._parts.exists(CUBE_PART_MARKET):
            idx = load_trading_calendar(self._parts)

        return plan_window(self._parts, CUBE_PART_PRICES, full=full,
                           warmup=self._part.warmup_trading_days, 
                           trading_index=idx)

    def _load_prices(self, since: pd.Timestamp | None) -> pd.DataFrame:
        """The one read of the raw ~1.9M-row `prices` table. `since` is pushed into SQL
        so an incremental run transfers a few hundred trading days instead of fifteen years 
        and then discarding 90% of them"""
        raw = self._parts.read(PRICES_TABLE, since=since)
        self._log.info(f"Loading {PRICES_TABLE} since={since if since else "full"}")
        return raw

    @staticmethod
    def _pivot_fields(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
        pivot = du.prices_long_to_multiindex(raw)
        return {"close": du.extract_field(pivot, "Close"),
                "open": du.extract_field(pivot, "Open"),
                "high": du.extract_field(pivot, "High"),
                "low": du.extract_field(pivot, "Low"),
                "volume": du.extract_field(pivot, "Volume")}

    def _trading_calendar(self, close: pd.DataFrame) -> pd.DatetimeIndex:
        """The market ticker's own calendar. `du.get_trading_days` also WARNS about interior
        holes -- dates where a quorum of stocks trade but the market ticker is missing --
        because those dates get dropped for the entire universe."""
        mask = du.get_trading_days(close, self._market_ticker)
        idx = pd.DatetimeIndex(close.index[mask.to_numpy()], name="date")
        self._log.info("Trading calendar: %d dates (%s .. %s)", len(idx),
                       idx.min().date(), idx.max().date())
        return idx

    @staticmethod
    def _on_calendar(wide: dict[str, pd.DataFrame],
                     idx: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
        """close/open/high/low are sliced to the calendar; volume is REINDEXED (it may be
        missing on a date the price exists), matching the original behaviour."""
        out = {k: v.loc[idx] for k, v in wide.items() if k != "volume"}
        out["volume"] = wide["volume"].reindex(idx)
        return out

    @staticmethod
    def _daily_returns(close: pd.DataFrame) -> pd.DataFrame:
        return du.daily_returns(close)

    def _market_frames(self, close: pd.DataFrame,
                       returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The market ticker + the configured `other_tickers` (commodities / FX) that the
        factor panel needs. Kept in their OWN table so they can never leak into a
        cross-sectional rank over the universe."""
        # `other_tickers` already contains the market ticker in the shipped config, so
        # dict.fromkeys dedupes while keeping the market ticker first (a duplicate column
        # makes the wide->long `stack` raise).
        want = list(dict.fromkeys([self._market_ticker, *self._other_tickers]))
        cols = [c for c in want if c in close.columns]
        missing = sorted(set(want) - set(cols))
        if missing:
            self._log.warning("market/other tickers absent from prices: %s", missing)
        if self._market_ticker not in cols:
            raise RuntimeError(f"market_ticker {self._market_ticker} is not in {PRICES_TABLE}"
                               " -> the trading calendar cannot be reconstructed downstream")
        return close[cols], returns[cols]

    def _peers(self) -> dict:
        peers = load_peers(self._context, self._config)
        n = sum(1 for p in peers.values() if p)
        self._log.info("Peer baskets ready for %s / %s tickers", n, len(peers))
        return peers

    def _universe_frames(self, wide: dict[str, pd.DataFrame], returns: pd.DataFrame,
                         peers: dict) -> dict[str, pd.DataFrame]:
        """Restrict every frame to the analysis universe (SORTED -- see
        `universe_columns`), then add the persisted return and peer-basket return."""
        
        universe = universe_columns(self._tickers, wide["close"])
        out = {k: v.reindex(columns=universe) for k, v in wide.items()}
        out["ret"] = returns.reindex(columns=universe)
        out["sector_ret"] = self._sector_returns(out["ret"], peers)

        self._log.info("Normalized prices: %d dates x %d universe tickers",
                       len(wide["close"]), len(universe))
        return out

    def _sector_returns(self, stock_ret: pd.DataFrame, peers: dict) -> pd.DataFrame:
        """The peer-basket ("neighbor sector") return per ticker. This is the single most
        expensive thing the old per-task prologue did, and it ran fourteen times per build;
        persisting it means it runs once."""
        return compute_sector_returns(stock_ret, peers)

    def _persist(self, universe: dict[str, pd.DataFrame],
                 market: tuple[pd.DataFrame, pd.DataFrame], window) -> int:
        
        prices_long, market_long = frames_to_long(universe, market[0], market[1])
        n = write_part(self._parts, CUBE_PART_PRICES, prices_long, window)
        if n == COLUMNS_CHANGED:
            return n
        m = write_part(self._parts, CUBE_PART_MARKET, market_long, window)
        return COLUMNS_CHANGED if m == COLUMNS_CHANGED else n
