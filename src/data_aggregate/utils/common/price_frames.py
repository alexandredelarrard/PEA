"""
price_frames.py  (src/data_aggregate/utils/common/price_frames.py)
---------------------------------------------------------------
THE price contract between the cube sub-steps: `StepCubePrices` normalizes the raw
`prices` table ONCE into `cube_part_prices`, and every later step reads it back through
here, projected to the fields it actually needs.

WHY. In the exploded DAG each of the fourteen tasks re-ran the same prologue: load the
whole ~1.9M-row `prices` table, pivot it to five wide frames, filter it to the trading
calendar, restrict it to the universe, and recompute peer sector returns with a
490-iteration Python loop. That is ~400 MB peak and the most expensive thing in the task,
paid FOURTEEN times per run. Now it is paid once.

ONE PART TABLE, NOT TWO. There used to be a second, `cube_part_market`, holding the market /
commodity / FX series: consumers pivot to wide and rank cross-sectionally (`xs_rank_pct`,
`panel.peer_relative`), so a single SPY or CL=F column leaking into that pivot would silently
shift every percentile in the panel, and a separate table guaranteed it structurally. That
firewall is now unnecessary at the source -- those series live in `prices_macro` and are
NEVER in `prices` -- so the part, and the `with_market` / `market_ticker` / `other_tickers`
plumbing that fed it, are gone. `StepCubeTarget` reads them straight from `prices_macro` via
`src/utils/macro.load_macro_wide`, the same helper the strategy sleeves use.

`ret` AND `sector_ret` ARE PERSISTED, not recomputed on read. Beyond the obvious saving,
this removes an incremental-correctness wart: a trimmed window recomputing
`close.pct_change()` returns NaN on the window's FIRST row where the full build had a
value, so the trailing tail would not reproduce the full build. Persisting them makes the
recompute exact by construction.

FLOAT64, not float32. `_atr` takes `high - low` and `high - close.shift(1)`; `_macd` takes
`ema_fast - ema_slow` -- differences of nearly-equal large numbers, then divided by close.
The repo convention is already *features float32, raw inputs float64* (see
`panel.build_peer_relative_panel` and `frames.downcast_float32` for the feature side, and
the `prices` table itself for the input side). This is a raw input.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.data_store.schema import Tables
from src.data_aggregate.utils.common import data_utils as du
from src.data_store.store import DataStore

logger = logging.getLogger(__name__)

ALL_FIELDS = ("close", "volume", "ret", "sector_ret")

@dataclass(frozen=True, slots=True)
class PriceFrames:
    """Wide (date x ticker) price frames for the ANALYSIS UNIVERSE, plus the peer dict.

    EQUITY ONLY. The market / commodity / FX series are not here -- they are read from
    `prices_macro` by the one step that needs them (`StepCubeTarget`), so nothing non-equity
    can reach a cross-sectional rank through this object.

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


def frames_to_long(universe_fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide -> the long part frame. WRITER side; used only by `StepCubePrices`.

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
    return prices.dropna(how="all").reset_index()


def load_trading_calendar(store: DataStore) -> pd.DatetimeIndex:
    """The trading calendar, read as `cube_part_prices`' own distinct dates.

    Still the definition rather than an inference: `du.get_trading_days` defines the calendar
    as the dates the market series traded, `StepCubePrices` slices every frame to exactly
    those dates before writing, and a row survives the write only if some field is non-null.
    It used to read `cube_part_market` -- the same dates, one table earlier. Cheap
    (`SELECT DISTINCT date`), so the incremental window can be decided BEFORE the wide read."""
    dates = store.distinct(Tables.cube_part_prices, "date")
    if not dates:
        raise RuntimeError(f"{Tables.cube_part_prices} is missing or empty -> run "
                           "`data_aggregate build-prices` first")
    return pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(dates)).dt.normalize().unique()),
                            name="date")


def load_price_frames(
    store: DataStore,
    *,
    peers: dict[str, dict[str, float]],
    fields: Sequence[str] = ALL_FIELDS,
    since: pd.Timestamp | None = None,
) -> PriceFrames:
    """Read `cube_part_prices` and pivot to wide.

    ONE read projected to the requested fields, then a pivot per field: the long frame's
    `object` ticker column costs ~100 MB on its own, so reading once per field would pay
    that N times. The long frame is dropped as soon as the pivots exist.
    """

    cols = ["date", "ticker"] + list(fields)
    long = store.load(Tables.cube_part_prices, columns=cols, since=since, optional=True)
    if long is None:
        raise RuntimeError(f"{Tables.cube_part_prices} is missing or returned no rows"
                           f"{f' since {pd.Timestamp(since).date()}' if since else ''}"
                           " -> run `data_aggregate build-prices` first")
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()

    wide: dict[str, pd.DataFrame] = {}
    for f in fields:
        piv = long.pivot(index="date", columns="ticker", values=f).sort_index()
        piv.index.name, piv.columns.name = "date", "ticker"
        wide[f] = piv

    idx = pd.DatetimeIndex(sorted(long["date"].unique()), name="date")
    universe = tuple(sorted(long["ticker"].astype(str).unique()))
    del long

    return PriceFrames(
        trading_index=idx, universe=universe, peers=peers,
        close=wide.get("close"), 
        volume=wide.get("volume"), 
        ret=wide.get("ret"),
        sector_ret=wide.get("sector_ret"),
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
