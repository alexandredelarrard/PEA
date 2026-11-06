"""
Projected price-part fields shared by the cube builders.
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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.price.adjustments import ShareBasis
from src.utils.price.frames import extract_field, prices_long_to_multiindex
from src.utils.price.levels import rebase_security_quantities
from src.utils.price.per_share import vendor_ordinary_per_share

logger = logging.getLogger(__name__)

# Adjusted close_total drives returns; close_split preserves the existing adjusted levels.
# close_pit is the historical security quote. Counts preserve source class coverage;
# mcap_pit is issuer equity and requires its separately supported class coefficient.
# level_factor is dimensionless and must never enter return calculations.
# security_basis is the forward quantity basis F (q_t = q_s * F_t / F_s); volume_pit is as-traded volume.
ALL_FIELDS = (
    "close_split",
    "close_total",
    "open",
    "high",
    "low",
    "volume",
    "ret",
    "sector_ret",
    "level_factor",
    "close_pit",
    "shares_outstanding_pit",
    "mcap_pit",
    "security_basis",
    "volume_pit",
)


@dataclass(frozen=True, slots=True)
class PriceFrames:
    """Wide (date x ticker) price frames for the ANALYSIS UNIVERSE, plus the peer dict.

    EQUITY ONLY. The market / commodity / FX series are not here -- they are read from
    `prices_macro` by the one step that needs them (`StepCubeTarget`), so nothing non-equity
    can reach a cross-sectional rank through this object.

    Every wide field is `None` unless it was requested via `load_price_frames(fields=...)`,
    so a step that only needs `close_split` never materialises open/high/low/volume.

    FROZEN: a sub-step may not mutate what the next one reads. Combined with the rule that
    sub-steps keep their frames LOCAL to `run()`, this is what stops seven sequential
    sub-steps accumulating seven sub-steps' worth of memory.
    """

    trading_index: pd.DatetimeIndex
    universe: tuple[str, ...]
    peers: dict[str, dict[str, float]]
    close_split: pd.DataFrame | None = None
    close_total: pd.DataFrame | None = None
    open: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None
    ret: pd.DataFrame | None = None
    sector_ret: pd.DataFrame | None = None
    #: Persisted compatibility factor S = Y * H / (F * U); never a return input.
    level_factor: pd.DataFrame | None = None
    close_pit: pd.DataFrame | None = None
    shares_outstanding_pit: pd.DataFrame | None = None
    mcap_pit: pd.DataFrame | None = None
    security_basis: pd.DataFrame | None = None
    volume_pit: pd.DataFrame | None = None

    def require(self, *fields: str) -> None:
        """Fail at the top of a builder rather than with a None-arithmetic TypeError two
        hundred lines deep."""
        missing = [f for f in fields if getattr(self, f, None) is None]
        if missing:
            raise ValueError(f"PriceFrames was loaded without {missing}; pass fields={tuple(fields)} to load_price_frames")

    @property
    def availability(self) -> pd.DataFrame | None:
        """True where split-adjusted close is observed; None for non-price test frames."""
        return None if self.close_split is None else self.close_split.notna()

    def skeleton(self) -> pd.DataFrame:
        """The (date, ticker) grid of cells that HAVE a price -- what the merge-based panel
        builders left-join onto.

        Keyed on `close_split` rather than `close_total`: the two have the same non-null
        pattern by construction (one response, one upsert), but `close_split` is the one that
        is never null when the other is -- `close_total` is derived from it."""
        self.require("close_split")
        assert self.close_split is not None
        s = self.close_split.where(self.close_split.notna()).reset_index()
        idx_col = s.columns[0]
        m = s.melt(id_vars=idx_col, var_name="ticker", value_name="_v").dropna(subset=["_v"]).rename(columns={idx_col: "date"})
        return m[["date", "ticker"]].reset_index(drop=True)


def rebase_vendor_quantities(
    basis: ShareBasis, quantities: pd.Series, tickers: pd.Series, source_dates: pd.Series, target_dates: pd.Series
) -> pd.Series:
    """Undo the source's F restatement, then carry genuine quantities in target security units."""
    factor = basis.factor(tickers, source_dates, "vendor")
    original = pd.to_numeric(quantities, errors="coerce") / factor.where(np.isfinite(factor) & factor.gt(0))
    result = rebase_security_quantities(basis, original, tickers, source_dates, target_dates, pd.Series(1.0, index=quantities.index))
    return result if basis.actual_available else result.where(quantities.eq(0))


def comparison_share_history(
    df_history: pd.DataFrame,
    basis: ShareBasis | None,
    target_date: pd.Timestamp,
    count_fields: Sequence[str] = ("sharesOutstanding",),
    per_share_fields: Sequence[str] = (),
) -> pd.DataFrame:
    """Copy declared source fields into common comparison units, preserving raw definitions."""
    df_result = df_history.copy(deep=False)
    if basis is None:
        for field in (*count_fields, *per_share_fields):
            if field in df_result:
                df_result[field] = np.nan
        return df_result
    target = pd.Series(target_date, index=df_result.index)
    for field in count_fields:
        if field in df_result:
            values = rebase_vendor_quantities(basis, df_result[field], df_result["ticker"], df_result["as_of"], target)
            df_result[field] = values.where(values.gt(0))
    for field in per_share_fields:
        if field in df_result:
            df_result[field] = vendor_ordinary_per_share(basis, df_result[field], df_result["ticker"], df_result["as_of"], target)
    return df_result


def frames_to_long(universe_fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide -> the long part frame. WRITER side; used only by `StepCubePrices`.

    Rows where every value column is NULL are dropped (a ticker that had not listed yet
    contributes nothing)."""

    close = universe_fields.get("close_split")
    if close is None:
        raise ValueError("frames_to_long requires close_split to define ticker availability")
    active = close.notna()
    frames = []
    for field, wide in universe_fields.items():
        if wide is None or wide.empty:
            continue
        wide = wide.where(active.reindex(index=wide.index, columns=wide.columns, fill_value=False))
        s = cast(pd.Series, wide.stack(future_stack=True))
        s.index = s.index.set_names(["date", "ticker"])
        s.name = field
        frames.append(s)

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
        raise RuntimeError(f"{Tables.cube_part_prices} is missing or empty -> run `data_aggregate build-prices` first")
    return pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(dates)).dt.normalize().unique()), name="date")


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
    chunks = list(store.iter_load(Tables.cube_part_prices, columns=cols, since=since))
    long = pd.concat(chunks, ignore_index=True) if chunks else None
    if long is None:
        raise RuntimeError(
            f"{Tables.cube_part_prices} is missing or returned no rows"
            f"{f' since {pd.Timestamp(since).date()}' if since else ''}"
            " -> run `data_aggregate build-prices` first"
        )
    long["date"] = pd.to_datetime(long["date"], format="%Y-%m-%d")

    wide: dict[str, pd.DataFrame] = {}
    for f in fields:
        piv = long.pivot(index="date", columns="ticker", values=f).sort_index().astype("float64")
        piv.index.name, piv.columns.name = "date", "ticker"
        wide[f] = piv

    idx = pd.DatetimeIndex(sorted(long["date"].unique()), name="date")
    universe = tuple(sorted(long["ticker"].astype(str).unique()))
    del long

    return PriceFrames(
        trading_index=idx,
        universe=universe,
        peers=peers,
        close_split=wide.get("close_split"),
        close_total=wide.get("close_total"),
        open=wide.get("open"),
        high=wide.get("high"),
        low=wide.get("low"),
        volume=wide.get("volume"),
        ret=wide.get("ret"),
        sector_ret=wide.get("sector_ret"),
        level_factor=wide.get("level_factor"),
        close_pit=wide.get("close_pit"),
        shares_outstanding_pit=wide.get("shares_outstanding_pit"),
        mcap_pit=wide.get("mcap_pit"),
        security_basis=wide.get("security_basis"),
        volume_pit=wide.get("volume_pit"),
    )


def universe_columns(tickers: Sequence[str], close: pd.DataFrame) -> list[str]:
    """The analysis universe as a SORTED, deterministic column list.

    The old code built this as a `set` and let `du._sub` iterate it, so every `stock_*`
    frame's column order varied with PYTHONHASHSEED across processes. The effect on values
    is ~1e-16 (float summation order in `ret.mean(axis=1)`,
    `characteristic_to_factor_return`'s `.sum(axis=1)`), but it makes any bit-identity
    claim between two processes unprovable -- and the CLI runs each sub-step in its own
    process."""
    universe = sorted(set(tickers) & set(close.columns))
    if not universe:
        raise RuntimeError("no universe ticker has prices -> cube universe is empty")
    return universe


def get_trading_days(close: pd.DataFrame, market: pd.Series, market_name: str = "market") -> pd.Series:
    """The trading calendar: a boolean over `close.index`, True where the MARKET traded.

    Takes the market series, not a column name. It used to be `close[market_ticker]`, back
    when SPY was stored inside `prices` alongside the equities; the market series now lives in
    `prices_macro`, so the caller reads it and hands it in. Same definition, so the calendar
    is unchanged -- and the quorum denominator is now right, since `close` no longer carries
    four non-equity columns it was counting as stocks."""
    quorum = 0.5 * close.shape[1]
    trading_days = market.reindex(close.index).notna()
    stock_cov = close.notna().sum(axis=1)
    holes = close.index[(~trading_days) & (stock_cov >= quorum)]
    if len(holes):
        logger.warning(
            "%s (market series) missing on %d date(s) where >=50%% of stocks "
            "trade (%s .. %s) -> these dates are dropped for the ENTIRE universe. "
            "Re-run `data_extract macro` to backfill %s.",
            market_name,
            len(holes),
            holes.min().date(),
            holes.max().date(),
            market_name,
        )

    return trading_days


def pivot_fields(df_prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivot Yahoo bars into float64 adjusted price and volume frames."""
    df_pivot = prices_long_to_multiindex(df_prices)
    fields = {"open": "Open", "high": "High", "low": "Low", "close_split": "CloseSplit", "close_total": "CloseTotal", "volume": "Volume"}
    return {field: extract_field(df_pivot, source) for field, source in fields.items()}
