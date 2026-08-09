"""
price_frames.py  (src/data_aggregate/utils/common/price_frames.py)
---------------------------------------------------------------
THE price contract between the cube sub-steps: `StepCubePrices` normalizes the raw
`prices` table ONCE into two part tables, and every later step reads them back through
here, projected to the fields it actually needs.

WHY. In the exploded DAG each of the fourteen tasks re-ran the same prologue: load the
whole ~1.9M-row `prices` table, pivot it to five wide frames, filter it to the trading
calendar, restrict it to the universe, and recompute peer sector returns with a
490-iteration Python loop. That is ~400 MB peak and the most expensive thing in the task,
paid FOURTEEN times per run. Now it is paid once.

TWO TABLES, NOT AN `is_universe` FLAG. Consumers pivot to wide and then rank
cross-sectionally (`xs_rank_pct`, `panel.peer_relative`), so a single SPY or CL=F column
leaking into that pivot would silently shift every percentile in the panel. `du._sub(...,
universe)` guarantees that structurally today; a flag column would demote the guarantee to
a `where=` that six call sites must remember. The market table is ~4 tickers x ~3.8k days
= 15k rows, so the second round-trip is free.

`ret` AND `sector_ret` ARE PERSISTED, not recomputed on read. Beyond the obvious saving,
this removes an incremental-correctness wart: a trimmed window recomputing
`close.pct_change()` returns NaN on the window's FIRST row where the full build had a
value, so the trailing tail would not reproduce the full build. Persisting them makes the
recompute exact by construction.

FLOAT64, not float32. `_atr` takes `high - low` and `high - close.shift(1)`; `_macd` takes
`ema_fast - ema_slow` -- differences of nearly-equal large numbers, then divided by close.
The repo convention is already *features float32, raw inputs float64* (see
`panel.build_peer_relative_panel` and `part_io.downcast_float32` for the feature side, and
the `prices` table itself for the input side). This is a raw input.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.constants.constants import CUBE_PART_MARKET, CUBE_PART_PRICES
from src.data_aggregate.utils.common import data_utils as du
from src.data_aggregate.utils.common.part_io import PartStore

logger = logging.getLogger(__name__)

ALL_FIELDS = ("close", "open", "high", "low", "volume", "ret", "sector_ret")

@dataclass(frozen=True, slots=True)
class PriceFrames:
    """Wide (date x ticker) price frames for the ANALYSIS UNIVERSE, plus the market /
    commodity / FX series the factor panel needs, plus the peer dict.

    Every wide field is `None` unless it was requested via `load_price_frames(fields=...)`,
    so a step that only needs `close` never materialises open/high/low/volume.

    FROZEN: a sub-step may not mutate what the next one reads. Combined with the rule that
    sub-steps keep their frames LOCAL to `run()`, this is what stops seven sequential
    sub-steps accumulating seven sub-steps' worth of memory.
    """
    trading_index: pd.DatetimeIndex
    universe: tuple[str, ...]                 
    peers: dict[str, dict[str, float]]
    close: pd.DataFrame | None = None
    open: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None
    ret: pd.DataFrame | None = None
    sector_ret: pd.DataFrame | None = None
    market_close: pd.Series | None = None
    mkt_ret: pd.Series | None = None
    other_close: pd.DataFrame | None = None   # market + other_tickers, wide close

    def require(self, *fields: str) -> None:
        """Fail at the top of a builder rather than with a None-arithmetic TypeError two
        hundred lines deep."""
        missing = [f for f in fields if getattr(self, f, None) is None]
        if missing:
            raise ValueError(
                f"PriceFrames was loaded without {missing}; pass fields={tuple(fields)} "
                "to load_price_frames")

    def skeleton(self) -> pd.DataFrame:
        """The (date, ticker) grid of cells that HAVE a close -- what the merge-based panel
        builders left-join onto."""
        self.require("close")
        s = self.close.reset_index()
        idx_col = s.columns[0]
        m = (s.melt(id_vars=idx_col, var_name="ticker", value_name="_v")
             .dropna(subset=["_v"]).rename(columns={idx_col: "date"}))
        return m[["date", "ticker"]].reset_index(drop=True)


def frames_to_long(universe_fields: dict[str, pd.DataFrame],
                   market_close: pd.DataFrame,
                   market_ret: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide -> the two long part frames. WRITER side; used only by `StepCubePrices`.

    Rows where every value column is NULL are dropped (a ticker that had not listed yet
    contributes nothing)."""

    frames = []
    for field, wide in universe_fields.items():
        if wide is None or wide.empty:
            continue
        s = wide.stack(future_stack=True)
        s.index = s.index.set_names(["date", "ticker"])
        frames.append(s.rename(field))

    prices = pd.concat(frames, axis=1)
    prices = prices.dropna(how="all").reset_index()

    mkt = []
    for name, wide in (("close", market_close), ("ret", market_ret)):
        if wide is None or wide.empty:
            continue
        s = wide.stack(future_stack=True)
        s.index = s.index.set_names(["date", "ticker"])
        mkt.append(s.rename(name))

    market = (pd.concat(mkt, axis=1).dropna(how="all").reset_index() if mkt
              else pd.DataFrame(columns=["date", "ticker", "close", "ret"]))
    return prices, market


def load_trading_calendar(parts: PartStore) -> pd.DatetimeIndex:
    """The trading calendar, read from the MARKET part's own dates (~15k rows, ~1 MB).

    `du.get_trading_days` defines the calendar as the dates the market ticker traded, and
    `StepCubePrices` stores the market ticker on exactly those dates -- so this is the
    definition, not an inference. Read first and cheaply, so the incremental window can be
    decided BEFORE the wide read (the old code loaded 15 years and then trimmed)."""
    if not parts.exists(CUBE_PART_MARKET):
        raise RuntimeError(
            f"{CUBE_PART_MARKET} is missing -> run `data_aggregate build-prices` first")
    rows = parts.read(CUBE_PART_MARKET, columns=["date", "ticker"])
    if rows.empty:
        raise RuntimeError(f"{CUBE_PART_MARKET} is empty -> re-run `build-prices`")
    dates = pd.to_datetime(rows["date"]).dt.normalize().unique()
    return pd.DatetimeIndex(sorted(dates))


def load_price_frames(
    parts: PartStore,
    *,
    peers: dict[str, dict[str, float]],
    market_ticker: str,
    fields: Sequence[str] = ALL_FIELDS,
    with_market: bool = False,
    other_tickers: Sequence[str] = (),
    since: pd.Timestamp | None = None,
) -> PriceFrames:
    """Read `cube_part_prices` (+ `cube_part_market` when `with_market`) and pivot to wide.

    ONE read projected to the requested fields, then a pivot per field: the long frame's
    `object` ticker column costs ~100 MB on its own, so reading once per field would pay
    that N times. The long frame is dropped as soon as the pivots exist.
    """

    if not parts.exists(CUBE_PART_PRICES):
        raise RuntimeError(
            f"{CUBE_PART_PRICES} is missing -> run `data_aggregate build-prices` first")

    cols = ["date", "ticker"] + list(fields)
    long = parts.read(CUBE_PART_PRICES, columns=cols, since=since)
    if long.empty:
        raise RuntimeError(f"{CUBE_PART_PRICES} returned no rows"
                           f"{f' since {pd.Timestamp(since).date()}' if since else ''}")
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()

    wide: dict[str, pd.DataFrame] = {}
    for f in fields:
        piv = long.pivot(index="date", columns="ticker", values=f).sort_index()
        piv.index.name, piv.columns.name = "date", "ticker"
        wide[f] = piv
    idx = pd.DatetimeIndex(sorted(long["date"].unique()), name="date")
    universe = tuple(sorted(long["ticker"].astype(str).unique()))
    del long

    market_close = mkt_ret = other_close = None
    if with_market:
        want = [str(market_ticker)] + [str(t) for t in other_tickers]
        mrows = parts.read(CUBE_PART_MARKET, columns=["date", "ticker", "close", "ret"],
                           since=since)
        mrows["date"] = pd.to_datetime(mrows["date"]).dt.normalize()
        mrows = mrows[mrows["ticker"].astype(str).isin(want)]
        oc = mrows.pivot(index="date", columns="ticker", values="close").sort_index()
        orr = mrows.pivot(index="date", columns="ticker", values="ret").sort_index()
        # reindexed onto the equity calendar: the factor functions derive their own index
        # from this frame, so `price_column_returns(other_close, ...)` is numerically
        # identical to the old call against the full untrimmed close panel.
        other_close = oc.reindex(idx)
        if market_ticker in other_close.columns:
            market_close = other_close[market_ticker]
            mkt_ret = orr.reindex(idx)[market_ticker]
        else:
            logger.warning("market_ticker %s absent from %s", market_ticker, CUBE_PART_MARKET)

    return PriceFrames(
        trading_index=idx, universe=universe, peers=peers,
        close=wide.get("close"), open=wide.get("open"), high=wide.get("high"),
        low=wide.get("low"), volume=wide.get("volume"), ret=wide.get("ret"),
        sector_ret=wide.get("sector_ret"),
        market_close=market_close, mkt_ret=mkt_ret, other_close=other_close,
    )


def universe_columns(tickers: Sequence[str], close: pd.DataFrame) -> list[str]:
    """The analysis universe as a SORTED, deterministic column list.

    The old code built this as a `set` and let `du._sub` iterate it, so every `stock_*`
    frame's column order varied with PYTHONHASHSEED across processes. The effect on values
    is ~1e-16 (float summation order in `ret.mean(axis=1)`,
    `characteristic_to_factor_return`'s `.sum(axis=1)`), but it makes any bit-identity
    claim between two processes unprovable -- and the CLI runs each sub-step in its own
    process."""
    universe = sorted(set(map(str, tickers)) & set(map(str, close.columns)))
    if not universe:
        raise RuntimeError("sp500_tickers empty/unseeded -> cube universe is empty")
    return universe
