"""Source-specific reads of immutable 13F manifests and whole-book positions."""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator, Sequence
from contextlib import closing
from typing import Any

import pandas as pd

from src.data_aggregate.utils.sec_thirteen.holdings_clean import clean_filing_dates
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)


def _filing_layout(
    store: DataStore, table: Table, until: object, ciks: Sequence[str] | None
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]] | None:
    """Validate the source layout and build its projected public-prefix query."""
    actual = store.columns(table)
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
    )
    if not set(manifests) <= set(actual):
        logger.warning("%s has no completed accession schema; replay original sources before building PIT features.", table.name)
        return None
    where: dict[str, Any] = {"parse_status": ["complete", "position"]}
    if ciks is not None:
        where["cik"] = sorted({pad_cik(cik) for cik in ciks})
    query = {
        "columns": list(dict.fromkeys((*manifests, *positions))),
        "where": where,
        "until": until,
        "date_col": "filing_date",
        "markers": True,
    }
    return manifests, positions, query


def _partition_packet(
    df_payload: pd.DataFrame, table: Table, manifest_columns: tuple[str, ...], position_columns: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate verified completion markers from economic position rows."""
    complete = df_payload["parse_status"].eq("complete") & df_payload["cusip"].eq(table.empty_marker[1])
    df_filings = df_payload.loc[complete, list(manifest_columns)].reset_index(drop=True)
    df_holdings = df_payload.loc[df_payload["parse_status"].eq("position"), list(position_columns)].reset_index(drop=True)
    return df_filings, df_holdings


def iter_filing_vintages(
    store: DataStore,
    until: object,
    ciks: Sequence[str] | None = None,
    *,
    table: Table = Tables.sec13f_hr,
    chunksize: int = 500_000,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]] | None:
    """Open one projected source stream of many complete CIK-quarter groups."""
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize <= 0:
        raise ValueError("13F chunksize must be a positive integer")
    
    layout = _filing_layout(store, table, until, ciks)
    if layout is None:
        return None
    manifest_columns, position_columns, query = layout
    return _iter_filing_packets(store, table, manifest_columns, position_columns, query, chunksize)


def _iter_filing_packets(
    store: DataStore, table: Table, manifest_columns: tuple[str, ...], position_columns: tuple[str, ...], query: dict[str, Any], chunksize: int
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Partition verified markers and positions while owning the source cursor."""
    with closing(store.iter_load(table, chunksize=chunksize, order_by=("cik", "period", "accession_number"), **query)) as stream:
        for df_packet in _iter_whole_quarters(stream):
            yield clean_filing_dates(*_partition_packet(df_packet, table, manifest_columns, position_columns))
            del df_packet


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
