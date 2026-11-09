"""Reads of immutable `sec13f_hr` manifests and whole-book positions."""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator, Sequence
from contextlib import closing
from typing import Any, cast

import pandas as pd

from src.data_aggregate.utils.sec_thirteen.holdings_clean import clean_filing_dates
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)


def _filing_layout(
    store: DataStore, until: object, ciks: Sequence[str] | None, universe: Sequence[str] | None
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]] | None:
    """Validate the source layout and build its projected public-prefix query: whole books, or with `universe`
    every completion marker OR'd with the positions stored under a universe ticker."""
    actual = store.columns(Tables.sec13f_hr)
    manifests = (
        "accession_number",
        "cik",
        "period",
        "filing_date",
        "accepted_at",
        "amendment_type",
        "amendment_number",
        "report_type",
        "restricted",
        "parsed_row_count",
        "parse_status",
    )
    numeric = ("shares", "value_usd", "call_shares", "call_value", "put_shares", "put_value", "debt_prn", "debt_value", "other_value")
    positions = (
        "cik",
        "accession_number",
        "cusip",
        "period",
        "filing_date",
        "position_type",
        "class_evidence",
        *numeric,
        *("raw_" + column for column in numeric if "value" in column),
        "ticker",
    )
    if not set(manifests) <= set(actual):
        logger.warning("%s has no completed accession schema; replay original sources before building PIT features.", Tables.sec13f_hr.name)
        return None
    scope: dict[str, Any] = {} if ciks is None else {"cik": sorted({pad_cik(cik) for cik in ciks})}
    where: dict[str, Any] | list[dict[str, Any]] = {"parse_status": ["complete", "position"], **scope}
    if universe is not None:
        markers = {"parse_status": "complete", "cusip": cast(tuple[str, str], Tables.sec13f_hr.empty_marker)[1], **scope}
        where = [markers, {"parse_status": "position", "ticker": sorted(set(universe)), **scope}]
    query = {
        "columns": list(dict.fromkeys((*manifests, *positions))),
        "where": where,
        "until": until,
        "date_col": "filing_date",
        "markers": True,
    }
    return manifests, positions, query


def _partition_packet(
    df_payload: pd.DataFrame, manifest_columns: tuple[str, ...], position_columns: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate verified completion markers from economic position rows."""
    complete = df_payload["parse_status"].eq("complete") & df_payload["cusip"].eq(cast(tuple[str, str], Tables.sec13f_hr.empty_marker)[1])
    df_filings = df_payload.loc[complete, list(manifest_columns)].reset_index(drop=True)
    df_holdings = df_payload.loc[df_payload["parse_status"].eq("position"), list(position_columns)].reset_index(drop=True)
    return df_filings, df_holdings


def iter_filing_vintages(
    store: DataStore,
    until: object,
    ciks: Sequence[str] | None = None,
    *,
    universe: Sequence[str] | None = None,
    chunksize: int = 500_000,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]] | None:
    """Open one projected source stream of many complete CIK-quarter groups. With `universe`, a single-version book
    carries only its universe positions (the extraction writes positions and marker atomically, so it is complete)
    and an amended book carries all its positions, so the resolver's stored-row count check still holds."""
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize <= 0:
        raise ValueError("13F chunksize must be a positive integer")

    layout = _filing_layout(store, until, ciks, universe)
    if layout is None:
        return None
    manifest_columns, position_columns, query = layout
    return _iter_filing_packets(store, manifest_columns, position_columns, query, chunksize)


def _iter_filing_packets(
    store: DataStore, manifest_columns: tuple[str, ...], position_columns: tuple[str, ...], query: dict[str, Any], chunksize: int
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Partition verified markers and positions while owning the source cursor."""
    with closing(store.iter_load(Tables.sec13f_hr, chunksize=chunksize, order_by=("cik", "period", "accession_number"), **query)) as stream:
        for df_packet in _iter_whole_quarters(stream):
            df_packet = _complete_amended_books(store, query, df_packet)
            yield clean_filing_dates(*_partition_packet(df_packet, manifest_columns, position_columns))
            del df_packet


def _complete_amended_books(store: DataStore, query: dict[str, Any], df_packet: pd.DataFrame) -> pd.DataFrame:
    """On a universe-scoped (OR) read, replace the positions of every amended book (any book that is not exactly one
    complete original/restatement) by all its stored positions: one keyed read per batch, index-backed on `cik`."""
    if not isinstance(query["where"], list):
        return df_packet

    # Amended books from the batch's completion markers (the read's only complete rows); whole quarters never span batches.
    df_markers = df_packet.loc[df_packet["parse_status"].eq("complete")]
    versions = df_markers.groupby(["cik", "period"])["accession_number"].transform("size")
    single = versions.eq(1) & df_markers["amendment_type"].isin(["original", "restatement"])
    df_amended = df_markers.loc[~single]
    if df_amended.empty:
        return df_packet

    # Every stored position of those accessions replaces their universe slice.
    where = {
        "cik": sorted(set(df_amended["cik"])),
        "period": sorted(set(df_amended["period"])),
        "accession_number": sorted(df_amended["accession_number"]),
        "parse_status": "position",
    }
    df_full = store.load(Tables.sec13f_hr, columns=query["columns"], where=where, optional=True, markers=True)
    amended_positions = df_packet["accession_number"].isin(set(df_amended["accession_number"])) & df_packet["parse_status"].eq("position")
    return pd.concat([df_packet.loc[~amended_positions], df_full], ignore_index=True)


def _iter_whole_quarters(stream: Iterator[pd.DataFrame]) -> Generator[pd.DataFrame]:
    """Yield complete leading groups together, carrying only the unfinished final group."""
    pieces: list[pd.DataFrame] = []
    previous_key: object = None
    for df_chunk in stream:
        if df_chunk.empty:
            continue
        key = (df_chunk.cik.iat[-1], df_chunk.period.iat[-1])
        trailing = df_chunk.cik.eq(key[0]) & df_chunk.period.eq(key[1])
        boundary = len(df_chunk) - int(trailing.sum())
        if key == previous_key:
            pieces.append(df_chunk.copy())
            del df_chunk
            continue
        if boundary:
            pieces.append(df_chunk.iloc[:boundary].copy())
        df_pending = df_chunk.iloc[boundary:].copy()
        del df_chunk
        df_batch = pd.concat(pieces, ignore_index=True) if pieces else None
        pieces = [df_pending]
        previous_key = key
        del df_pending
        if df_batch is not None:
            yield df_batch
            del df_batch
    if pieces:
        yield pd.concat(pieces, ignore_index=True)
