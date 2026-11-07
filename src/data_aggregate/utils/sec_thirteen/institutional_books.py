"""Bounded raw preparation and compact public-book history for the broad 13F panel."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.data_aggregate.utils.sec_thirteen.institutional_vintages import _history_floor
from src.utils.institutional_vintages import prepare_book_vintages, resolve_prepared_books

logger = logging.getLogger(__name__)


def compact_institutional_books(df_books: pd.DataFrame) -> pd.DataFrame:
    """Keep mapped amounts, unknown/zero rows and positive-share security identities, one row per (period, manager,
    company) as stored: `sec13f_hr` holds one row per manager x company x accession in canonical-class shares (U-67)."""
    numeric = ["shares", "value_usd", "call_value", "put_value"]
    keys = ["period", "cik", "ticker"]
    df_mapped = df_books.loc[df_books.ticker.notna()]
    duplicated = df_mapped.duplicated(keys)
    if duplicated.any():
        sample = df_mapped.loc[duplicated, keys].head(3).to_dict("records")
        raise ValueError(f"13F books hold {int(duplicated.sum())} duplicate manager x company row(s), e.g. {sample}: run `identity --only propagate`")
    df_totals = df_mapped.set_index(keys)[numeric]
    identities = [frozenset({cusip}) if held else np.nan for cusip, held in zip(df_mapped["cusip"], df_mapped["shares"].gt(0), strict=True)]
    df_totals["_cusips"] = pd.Series(identities, index=df_totals.index, dtype="object")
    return df_totals


def _book_codes(df_manifest: pd.DataFrame, df_holdings: pd.DataFrame) -> tuple[pd.MultiIndex, NDArray[np.intp], NDArray[np.intp]]:
    """Number manager-quarters in first-appearance order; a position takes its quarter's code only under its own accession."""
    books = pd.MultiIndex.from_frame(df_manifest[["cik", "period"]]).unique()
    codes = books.get_indexer(pd.MultiIndex.from_frame(df_manifest[["cik", "period"]]))
    holding_codes = books.get_indexer(pd.MultiIndex.from_frame(df_holdings[["cik", "period"]]))
    accession_codes = df_holdings["accession_number"].map(pd.Series(codes, index=df_manifest["accession_number"].to_numpy()))
    holding_codes = np.where(accession_codes.to_numpy(dtype=float) == holding_codes, holding_codes, -1)
    return books, codes, holding_codes


def _resolve_public_days(
    df_manifest: pd.DataFrame, df_holdings: pd.DataFrame, identities: dict[frozenset, frozenset]
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.MultiIndex]:
    """Resolve and compact, once per public day, every manager-quarter published that day."""
    books, codes, holding_codes = _book_codes(df_manifest, df_holdings)
    df_events = pd.DataFrame({"_book": codes, "_day": df_manifest["public_at"].to_numpy()}).dropna().drop_duplicates()
    df_positions = df_holdings.drop(columns=[column for column in df_holdings if column.startswith("value_basis_") or column == "filing_date"])
    frames: list[pd.DataFrame] = []
    states: list[pd.DataFrame] = []

    # A quarter's book at a day depends only on its own versions, so one resolver call serves all quarters of that day;
    # provenance columns never reach the compact amounts, so they stay out of the per-accession resolver copies.
    for day, published in df_events.groupby("_day", sort=True)["_book"]:
        member = np.zeros(len(books) + 1, dtype=bool)  # the last slot is the unowned code -1
        member[published.to_numpy()] = True
        df_book, df_state = resolve_prepared_books(df_manifest.loc[member[codes]], df_positions.loc[member[holding_codes]], day)
        df_compact = compact_institutional_books(df_book)
        identities.update((value, value) for value in df_compact["_cusips"].dropna() if value not in identities)
        df_compact["_cusips"] = df_compact["_cusips"].map(identities).astype("object")
        frames.append(df_compact)
        states.append(df_state.assign(_day=day))
    return frames, states, books


def _compact_batch_events(
    df_manifest: pd.DataFrame, df_holdings: pd.DataFrame, identities: dict[frozenset, frozenset]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Order compact public replacements by manager-quarter, then public day, with each state's row span."""
    frames, states, books = _resolve_public_days(df_manifest, df_holdings, identities)
    if not states or all(state.empty for state in states):
        return pd.DataFrame(), pd.DataFrame({"_start": pd.Series(dtype="int64"), "_stop": pd.Series(dtype="int64")})

    # Rows keep their within-day order; the stable sort groups them by quarter first-appearance, then day.
    df_compact = pd.concat(frames)
    df_states = pd.concat(states, ignore_index=True)
    day_numbers = np.arange(len(frames))
    compact_books = books.get_indexer(
        pd.MultiIndex.from_arrays([df_compact.index.get_level_values("cik"), df_compact.index.get_level_values("period")])
    )
    compact_keys = compact_books * len(frames) + np.repeat(day_numbers, [len(frame) for frame in frames])
    state_books = books.get_indexer(pd.MultiIndex.from_frame(df_states[["cik", "period"]]))
    state_keys = state_books * len(frames) + np.repeat(day_numbers, [len(state) for state in states])
    order = np.argsort(compact_keys, kind="stable")
    df_compact, compact_keys = df_compact.iloc[order], compact_keys[order]
    df_states = df_states.iloc[np.argsort(state_keys, kind="stable")].reset_index(drop=True)
    state_keys = np.sort(state_keys, kind="stable")

    # Each state spans its own rows in the ordered compact frame.
    df_states["_start"] = np.searchsorted(compact_keys, state_keys, side="left").astype("int64")
    df_states["_stop"] = np.searchsorted(compact_keys, state_keys, side="right").astype("int64")
    return df_compact, df_states


def prepare_institutional_book_history(
    batches: Iterator[tuple[pd.DataFrame, pd.DataFrame]], trading_index: pd.DatetimeIndex, *, settle_days: int
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Retain only compact public events of the cleaned complete-book batches."""
    frames: list[pd.DataFrame] = []
    states: list[pd.DataFrame] = []
    identities: dict[frozenset, frozenset] = {}
    offset = 0
    filings_count = 0
    raw_rows = 0

    for df_filings, df_holdings in batches:
        filings_count += len(df_filings)
        raw_rows += len(df_holdings)
        df_manifest = prepare_book_vintages(df_filings, df_holdings, trading_index, settle_days)
        df_compact, df_states = _compact_batch_events(df_manifest, df_holdings, identities)
        del df_filings, df_holdings, df_manifest
        frames.append(df_compact)
        states.append(df_states.assign(_start=df_states["_start"] + offset, _stop=df_states["_stop"] + offset))
        offset += len(df_compact)

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
