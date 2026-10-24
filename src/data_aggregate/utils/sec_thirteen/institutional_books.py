"""Bounded raw preparation and compact public-book history for the broad 13F panel."""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import closing
from itertools import groupby
from typing import Any, cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.sec_thirteen.holdings_clean import attach_tickers, clean_public_holdings
from src.data_aggregate.utils.sec_thirteen.institutional_vintages import _history_floor
from src.data_aggregate.utils.sec_thirteen.value_basis import prepare_public_value_quotes
from src.utils.institutional_vintages import prepare_book_vintages, resolve_prepared_books

logger = logging.getLogger(__name__)


def compact_institutional_books(df_books: pd.DataFrame) -> pd.DataFrame:
    """Keep mapped amounts, unknown/zero rows and positive-share security identities."""
    numeric = ["shares", "value_usd", "call_value", "put_value"]
    keys = ["period", "cik", "ticker"]
    df_mapped = df_books.loc[df_books.ticker.notna()]
    grouped = df_mapped.groupby(keys)[numeric]
    df_totals = grouped.sum(min_count=1).where(grouped.count().eq(df_mapped.groupby(keys).size(), axis=0))
    identities = df_mapped.loc[df_mapped.shares.gt(0)].groupby(keys).cusip.agg(frozenset)
    df_totals["_cusips"] = identities.astype("object")
    return df_totals


def _compact_vintage_events(
    df_manifest: pd.DataFrame, df_holdings: pd.DataFrame, identities: dict[frozenset, frozenset]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce each public replacement of one raw manager-quarter before releasing it."""
    frames: list[pd.DataFrame] = []
    states: list[pd.DataFrame] = []
    offset = 0
    days = pd.DatetimeIndex(df_manifest.public_at.dropna().unique()).sort_values()

    for day in days:
        df_book, df_state = resolve_prepared_books(df_manifest, df_holdings, day)
        df_compact = compact_institutional_books(df_book)
        identities.update((value, value) for value in df_compact["_cusips"].dropna() if value not in identities)
        df_compact["_cusips"] = df_compact["_cusips"].map(identities).astype("object")
        frames.append(df_compact)
        states.append(df_state.assign(_day=day, _start=offset, _stop=offset + len(df_compact)))
        offset += len(df_compact)
    return (pd.concat(frames), pd.concat(states, ignore_index=True)) if states else (pd.DataFrame(), pd.DataFrame())


def _packet_cik(packet: tuple[pd.DataFrame, pd.DataFrame]) -> str:
    """Identify a filer's ordered quarter, including positions without completion evidence."""
    df_filings, df_holdings = packet
    if not df_filings.empty:
        return str(df_filings.cik.iat[0])
    return str(df_holdings.cik.iat[0]) if not df_holdings.empty else ""


def _prepare_manager_book_history(
    packets: Iterator[tuple[pd.DataFrame, pd.DataFrame]],
    trading_index: pd.DatetimeIndex,
    cusip_map: pd.DataFrame | None,
    universe: Sequence[str],
    valuation_close: pd.DataFrame | None,
    value_basis: Mapping[str, Any] | None,
    *,
    period_quotes: pd.Series | None,
    identities: dict[frozenset, frozenset],
    settle_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Clean raw quarters separately and join only one filer's compact event fragments."""
    frames: list[pd.DataFrame] = []
    states: list[pd.DataFrame] = []
    offset = 0
    filings_count = 0
    raw_rows = 0

    for df_filings, df_raw in packets:
        filings_count += len(df_filings)
        raw_rows += len(df_raw)
        df_holdings = attach_tickers(df_raw, cusip_map, universe)
        df_holdings = clean_public_holdings(df_holdings, valuation_close, value_basis, period_quotes=period_quotes)
        df_manifest = prepare_book_vintages(df_filings, df_holdings, trading_index, settle_days)
        df_compact, df_states = _compact_vintage_events(df_manifest, df_holdings, identities)
        del df_filings, df_raw, df_holdings, df_manifest
        if df_states.empty:
            continue
        frames.append(df_compact)
        states.append(df_states.assign(_start=df_states["_start"] + offset, _stop=df_states["_stop"] + offset))
        offset += len(df_compact)

    df_compact = pd.concat(frames) if frames else pd.DataFrame()
    df_states = (
        pd.concat(states, ignore_index=True) if states else pd.DataFrame({"_start": pd.Series(dtype="int64"), "_stop": pd.Series(dtype="int64")})
    )
    return df_compact, df_states, filings_count, raw_rows


def prepare_institutional_book_history(
    packets: Generator[tuple[pd.DataFrame, pd.DataFrame]],
    trading_index: pd.DatetimeIndex,
    cusip_map: pd.DataFrame | None,
    universe: Sequence[str],
    valuation_close: pd.DataFrame | None,
    value_basis: Mapping[str, Any] | None,
    *,
    settle_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Read raw quarters once, retaining one compact event pair per filer."""
    period_quotes = prepare_public_value_quotes(valuation_close, value_basis)
    frames: list[pd.DataFrame] = []
    states: list[pd.DataFrame] = []
    identities: dict[frozenset, frozenset] = {}
    offset = 0
    filings_count = 0
    raw_rows = 0

    with closing(packets):
        for _, manager_packets in groupby(packets, key=_packet_cik):
            df_compact, df_states, filings, rows = _prepare_manager_book_history(
                manager_packets,
                trading_index,
                cusip_map,
                universe,
                valuation_close,
                value_basis,
                period_quotes=period_quotes,
                identities=identities,
                settle_days=settle_days,
            )
            frames.append(df_compact)
            states.append(df_states.assign(_start=df_states["_start"] + offset, _stop=df_states["_stop"] + offset))
            offset += len(df_compact)
            filings_count += filings
            raw_rows += rows

    if not filings_count:
        logger.warning("No complete immutable 13F filings in scope; recover originals/amendments. Latest-only rows cannot supply PIT books.")
        return None
    frames = [frame for frame, state in zip(frames, states, strict=True) if not state.empty]
    states = [state for state in states if not state.empty]
    df_compact = pd.concat(frames) if frames else pd.DataFrame()
    df_states = pd.concat(states, ignore_index=True) if states else pd.DataFrame()
    logger.info(
        "Immutable broad 13F: %s accessions / %s raw rows reduced to %s mapped rows / %s dated book states.",
        filings_count,
        raw_rows,
        len(df_compact),
        len(df_states),
    )
    return df_compact, df_states


def iter_institutional_books(df_compact: pd.DataFrame, df_states: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    """Replay compact replacements against the existing global active-quarter floor."""
    cache: dict[tuple[str, pd.Timestamp], int] = {}
    active: Any = pd.NaT
    if df_states.empty:
        return

    publications = df_states.groupby("_day", sort=True).indices
    days = pd.DatetimeIndex(df_states["_day"].unique()).sort_values()
    for day in days:
        df_events = df_states.iloc[publications[day]]
        floor = _history_floor(active, 1)
        for event in df_events.loc[df_events.period.ge(floor)].itertuples():
            cache[(cast(str, event.cik), cast(pd.Timestamp, event.period))] = cast(int, event.Index)
        df_current = df_states.iloc[list(cache.values())]
        known = df_current.loc[df_current.book_known, "period"]
        if not known.empty:
            active = known.max()
        floor = _history_floor(active, 1)
        cache = {key: index for key, index in cache.items() if key[1] >= floor}
        if not cache:
            continue
        df_current = df_states.iloc[list(cache.values())]
        bounds = df_current[["_start", "_stop"]].to_numpy(dtype=np.intp)
        indices = np.concatenate([np.arange(start, stop, dtype=np.intp) for start, stop in bounds])
        yield pd.Timestamp(day), df_compact.iloc[indices], df_current.drop(columns=["_day", "_start", "_stop"]).reset_index(drop=True)
