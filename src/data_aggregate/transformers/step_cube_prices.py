"""
step_cube_prices.py  (src/data_aggregate/transformers/step_cube_prices.py)
-----------------------------------------------------------------------
The ONLY step that reads the raw `prices` table, the ONLY one that pivots it, and the ONLY
one that computes peer sector returns. Everything downstream reads the part table it writes
(`cube_part_prices`) through `utils/common/price_frames.load_price_frames`, projected to the
fields it needs.

`prices` is the EQUITY universe and nothing else, so there is no market/commodity/FX split to
make here any more: the second part table (`cube_part_market`) and its `_market_frames`
subsetting are gone, and `StepCubeTarget` reads those series from `prices_macro` directly.
The trading calendar still comes from the market series -- read from `prices_macro`, which is
the table that owns it.

That prologue -- load ~1.9M rows, pivot to five wide frames, filter to the trading
calendar, restrict to the universe, then a 490-iteration Python loop for the peer sector
returns -- used to run once per DAG task, fourteen times per build, at ~400 MB peak each.

MEMORY DISCIPLINE (the invariant that makes seven sequential sub-steps safe): every heavy
frame is LOCAL to `run()`. Nothing is stashed on `self`, so when `run()` returns its
locals are collected and the next sub-step starts from a clean slate.
"""

import pandas as pd
from omegaconf import DictConfig

from src.data_store.schema import Tables
from src.constants.constants_price import MACRO_MARKET_SERIES
from src.context import Context
from src.data_aggregate.utils.common import data_utils as du
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, plan_window, write_part
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.peers_io import load_peers
from src.data_aggregate.utils.common.price_frames import frames_to_long, universe_columns
from src.data_aggregate.utils.common.price_frames import load_trading_calendar
from src.data_peers.utils.sector_peers import compute_sector_returns

from src.utils.macro import load_macro_series
from src.utils.step import Step
from src.utils.universe import load_universe_tickers

PRICE_COLS = ['date', 'close', 'volume', 'ticker']

class StepCubePrices(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_prices)
        self._store = context.store
        self._tickers = load_universe_tickers(context)
        self._log.info(f"Ticker universe: {len(self._tickers)} tickers from {Tables.sp500_tickers}")

    def run(self, full: bool = False) -> None:

        window = self._plan_window(full)
        since = window.since
        raw = self._store.load(Tables.prices, 
                            since=since, 
                            columns = PRICE_COLS)
        self._log.info(f"Loading {Tables.prices} since={since if since else "full"}")

        # long to wide format 
        wide = self._pivot_fields(raw)

        # filter wide on trading days with close value
        days_index = self._trading_calendar(wide["close"])
        wide = self._on_calendar(wide, days_index)

        # get deltas
        returns =  self._daily_returns(wide["close"])
        peers = self._peers()
        universe = self._universe_frames(wide, returns, peers)
        del raw, wide, returns

        # save it all
        n = write_part(self._store, Tables.cube_part_prices,
                          frames_to_long(universe), window)
        
        if n == COLUMNS_CHANGED:
            return self.run(full=True)

    # ---- window ---- #
    def _plan_window(self, full: bool):
        """Warm-up 260 trading days. `ret` needs only one prior day (and is persisted, so
        the trailing recompute is exact), but `get_trading_days`'s interior-calendar-hole
        warning is a diagnostic over history and wants a year of context."""
        idx = None
        if self._store.exists(Tables.cube_part_prices):
            idx = load_trading_calendar(self._store)

        return plan_window(self._store, Tables.cube_part_prices, full=full,
                           warmup=self._part.warmup_trading_days,
                           trading_index=idx)

    @staticmethod
    def _pivot_fields(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
        pivot = du.prices_long_to_multiindex(raw)
        return {"close": du.extract_field(pivot, "Close"),
                "volume": du.extract_field(pivot, "Volume")}

    def _trading_calendar(self, close: pd.DataFrame) -> pd.DatetimeIndex:
        """The market series' own calendar, read from `prices_macro` (the table that owns it).
        `du.get_trading_days` also WARNS about interior holes -- dates where a quorum of stocks
        trade but the market series is missing -- because those dates get dropped for the
        entire universe."""
        market = load_macro_series(self._store, MACRO_MARKET_SERIES)
        if market is None:
            raise RuntimeError(
                f"'{Tables.prices_macro}' has no '{MACRO_MARKET_SERIES}' rows -> the cube "
                "trading calendar is undefined. Run `data_extract macro` first.")
        mask = du.get_trading_days(close, market, MACRO_MARKET_SERIES)
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
