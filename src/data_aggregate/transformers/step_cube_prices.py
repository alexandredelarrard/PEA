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
from src.constants.constants import SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT
from src.constants.constants_price import MACRO_MARKET_SERIES
from src.context import Context
from src.data_aggregate.utils.common import data_utils as du
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, plan_window, write_part
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.peers_io import load_peers
from src.data_aggregate.utils.common.level_basis import (
    apply_level_bugfix, apply_return_seams, apply_split_vintage, describe, genuine_splits,
    level_factor, load_bugfix)
from src.data_aggregate.utils.common.price_frames import frames_to_long, universe_columns
from src.data_aggregate.utils.common.price_frames import load_trading_calendar
from src.data_peers.utils.sector_peers import compute_sector_returns

from src.utils.macro import load_macro_series
from src.utils.step import Step
from src.utils.universe import load_universe_tickers

PRICE_COLS = ['date', 'close_split', 'close_total', 'volume', 'ticker']
#: The two `sharadar_actions` kinds `split_events` reads. Market-wide table, so the read is
#: filtered to these or it drags back every action of every ticker Sharadar covers.
ACTION_COLS = ['ticker', 'date', 'action', 'value']
#: Sharadar's as-reported quarterly dimension -- the only one whose `price` is a single dated
#: observation rather than a period aggregate, so the only one a bar can be compared against.
VENDOR_DIMENSION = 'ARQ' 

class StepCubePrices(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_prices)
        self._store = context.store
        self._tickers = load_universe_tickers(context)
        self._log.info(f"Ticker universe: {len(self._tickers)} tickers from {Tables.sp500_tickers}")
        # Read at construction so a malformed or unapproved register fails the step before it
        # has loaded 1.9M price rows, not after.
        self._bugfix = load_bugfix(context.config_dir)

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
        # The calendar and the universe key on `close_split`: it is never null when
        # `close_total` is (the latter is derived from it), so it defines the widest
        # legitimate grid.
        days_index = self._trading_calendar(wide["close_split"])
        wide = self._on_calendar(wide, days_index)

        # get deltas
        # ⚠ `close_total`, NOT `close_split`. `ret` is the persisted return that momentum,
        # vol, betas and every LABEL are built from, so it must be the buy-and-hold path.
        # Feeding it `close_split` would make every one of them a PRICE return -- MO reads
        # 1.24x over this sample where the truth is 20.2x.
        # ⚠ BEFORE `_daily_returns`. Two of the three repairs move `close_total`, so running
        # them afterwards would leave `ret` computed from the series we just disowned.
        vendor = self._vendor_price()
        self._repair(wide, vendor)

        returns =  self._daily_returns(wide["close_total"])
        peers = self._peers()
        universe = self._universe_frames(wide, returns, peers)
        universe["level_factor"] = self._level_factor(days_index, universe["close_split"],
                                                      vendor)
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
        return {"close_split": du.extract_field(pivot, "CloseSplit"),
                "close_total": du.extract_field(pivot, "CloseTotal"),
                "volume": du.extract_field(pivot, "Volume")}

    def _trading_calendar(self, close_split: pd.DataFrame) -> pd.DatetimeIndex:
        """The market series' own calendar, read from `prices_macro` (the table that owns it).
        `du.get_trading_days` also WARNS about interior holes -- dates where a quorum of stocks
        trade but the market series is missing -- because those dates get dropped for the
        entire universe."""
        market = load_macro_series(self._store, MACRO_MARKET_SERIES)
        if market is None:
            raise RuntimeError(
                f"'{Tables.prices_macro}' has no '{MACRO_MARKET_SERIES}' rows -> the cube "
                "trading calendar is undefined. Run `data_extract macro` first.")
        mask = du.get_trading_days(close_split, market, MACRO_MARKET_SERIES)
        idx = pd.DatetimeIndex(close_split.index[mask.to_numpy()], name="date")
        self._log.info("Trading calendar: %d dates (%s .. %s)", len(idx),
                       idx.min().date(), idx.max().date())
        return idx

    @staticmethod
    def _on_calendar(wide: dict[str, pd.DataFrame],
                     idx: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
        """The price frames are sliced to the calendar; volume is REINDEXED (it may be
        missing on a date the price exists), matching the original behaviour."""
        out = {k: v.loc[idx] for k, v in wide.items() if k != "volume"}
        out["volume"] = wide["volume"].reindex(idx)
        return out

    @staticmethod
    def _daily_returns(close_total: pd.DataFrame) -> pd.DataFrame:
        """TOTAL-return series in, `ret` out. The parameter is named for the basis it
        requires, so the one call site states its contract."""
        return du.daily_returns(close_total)

    def _peers(self) -> dict:
        peers = load_peers(self._context, self._config)
        n = sum(1 for p in peers.values() if p)
        self._log.info("Peer baskets ready for %s / %s tickers", n, len(peers))
        return peers

    def _universe_frames(self, wide: dict[str, pd.DataFrame], returns: pd.DataFrame,
                         peers: dict) -> dict[str, pd.DataFrame]:
        """Restrict every frame to the analysis universe (SORTED -- see
        `universe_columns`), then add the persisted return and peer-basket return."""
        
        universe = universe_columns(self._tickers, wide["close_split"])
        out = {k: v.reindex(columns=universe) for k, v in wide.items()}
        out["ret"] = returns.reindex(columns=universe)
        out["sector_ret"] = self._sector_returns(out["ret"], peers)

        self._log.info("Normalized prices: %d dates x %d universe tickers",
                       len(wide["close_split"]), len(universe))
        return out

    def _vendor_price(self) -> pd.DataFrame | None:
        """Sharadar's `price` for the REGISTERED tickers only -- the independent statement of
        what a stock actually traded at that every bugfix entry is re-verified against.

        Projected to the register's own tickers because this is a market-wide table and the
        entries name nine of them: the unfiltered read is ~1.4M rows to answer a question
        about a handful."""
        named = sorted({*(self._bugfix.get("level_factor") or {}),
                        *(self._bugfix.get("split_vintage") or {})})
        if not named:
            return None
        frame = self._store.load(Tables.sharadar_fundamentals,
                                 columns=["ticker", "date", "price"],
                                 where={"dimension": VENDOR_DIMENSION, "ticker": named},
                                 optional=True)
        if frame is None or frame.empty:
            self._log.warning("price bugfix: %s has no price for any of the %d registered "
                              "tickers -- every entry will be SKIPPED unverified",
                              Tables.sharadar_fundamentals, len(named))
            return None
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
        return frame

    def _repair(self, wide: dict[str, pd.DataFrame],
                vendor: pd.DataFrame | None) -> None:
        """Defects in YAHOO's own price data that no event feed can express, from
        `configs/prices/yf_price_bugfix.json`.

        ⚠ Every entry there states the value it EXPECTS TO OBSERVE, and each is re-measured
        against this build's own frames before anything is multiplied. An entry whose defect
        has been fixed upstream no longer matches, so it is SKIPPED and logged rather than
        applied a second time on top of Yahoo's own correction. Nothing is applied on faith.

        Only the two RETURN-MOVING repairs run here, because they have to happen before `ret`
        is computed. The level wedges are folded into `S` in `_level_factor`, where they
        belong: they change a market cap and must leave every return exactly as it was."""
        if not self._bugfix:
            return
        applied = apply_split_vintage(wide, self._bugfix, vendor, self._log.info)
        applied += apply_return_seams(wide, self._bugfix, self._log.info)
        listed = (sum(len(v) for v in (self._bugfix.get("split_vintage") or {}).values())
                  + sum(len(v) for v in (self._bugfix.get("return_seams") or {}).values()))
        self._log.info("price bugfix: %d of %d registered price repair(s) applied",
                       applied, listed)

    def _level_factor(self, idx: pd.DatetimeIndex, close_split: pd.DataFrame,
                      vendor: pd.DataFrame | None = None) -> pd.DataFrame:
        """`S(d)` -- the SPINOFF price adjustment `close_split` carries and a share count does
        not. Computed ONCE here and persisted, rather than three times in three sub-steps.

        The denominator comes from `field_map.split_events`, THE SAME function the extract
        layer de-adjusts `sharesOutstandingPit` with. Sharing it is what stops the numerator
        and denominator drifting apart -- a row that counts as a genuine split there has
        already cancelled against `sharesbas`, so it must not be counted twice here.

        ⚠ MASKED to `close_split`'s non-null pattern. `S` is 1.0 everywhere by construction,
        never NaN, so an unmasked column would make `frames_to_long`'s "drop rows where every
        value is NULL" a no-op and materialise a row for every (date, ticker) pair in the
        grid -- including years before a ticker listed.
        """
        yf_splits = self._store.load(Tables.prices_splits,
                                     columns=["ticker", "date", "ratio"], optional=True)
        actions = self._store.load(Tables.sharadar_actions, columns=ACTION_COLS,
                                   where={"action": [SHARADAR_ACTION_SPLIT,
                                                     SHARADAR_ACTION_SPINOFF]},
                                   optional=True)
        genuine = genuine_splits(actions, yf_splits)
        factor = level_factor(idx, list(close_split.columns), yf_splits, genuine)
        # The registered wedges are cases `S` is structurally BLIND to -- Yahoo adjusted the
        # price and its splits feed never said so -- so they multiply `S` rather than
        # replacing it, and land in the same stored column.
        factor = apply_level_bugfix(factor, self._bugfix, vendor, close_split,
                                    self._log.info)
        # Logged AFTER the mask, so the line describes what is actually STORED. Unmasked it
        # reports pre-listing cells and their factors dominate the ranking -- LVS x0.0038 and
        # VRSK x0.0200 both date from before those tickers had a single bar.
        masked = factor.reindex_like(close_split).where(close_split.notna())
        self._log.info(describe(masked))
        return masked

    def _sector_returns(self, stock_ret: pd.DataFrame, peers: dict) -> pd.DataFrame:
        """The peer-basket ("neighbor sector") return per ticker. This is the single most
        expensive thing the old per-task prologue did, and it ran fourteen times per build;
        persisting it means it runs once."""
        return compute_sector_returns(stock_ret, peers)
