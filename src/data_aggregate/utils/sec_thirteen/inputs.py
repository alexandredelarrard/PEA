"""Source-specific reads of immutable 13F manifests and whole-book positions."""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator, Sequence
from contextlib import closing
from itertools import chain
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


def _finish_packet(df_filings: pd.DataFrame, df_holdings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Keep completion evidence, normalize dates and report the loaded source scope."""
    if df_filings.empty:
        logger.warning("No complete immutable 13F filings in scope; recover originals/amendments. Latest-only rows cannot supply PIT books.")
        return None
    df_filings, df_holdings = clean_filing_dates(df_filings, df_holdings)
    logger.info(
        "Immutable 13F: %s complete accessions / %s whole-book rows / %s managers; unverified legacy rows excluded.",
        len(df_filings),
        len(df_holdings),
        df_filings.cik.nunique(),
    )
    return df_filings, df_holdings


def iter_filing_vintages(store: DataStore, until: object) -> Generator[tuple[pd.DataFrame, pd.DataFrame]] | None:
    """Open one projected broad stream, yielding complete raw CIK-quarter packets."""
    layout = _filing_layout(store, Tables.sec13f_hr, until, None)
    if layout is None:
        return None
    manifest_columns, position_columns, query = layout
    return _iter_filing_packets(store, manifest_columns, position_columns, query)


def _iter_filing_packets(
    store: DataStore, manifest_columns: tuple[str, ...], position_columns: tuple[str, ...], query: dict[str, Any]
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Partition verified markers and positions while owning the source cursor."""
    with closing(store.iter_load(Tables.sec13f_hr, order_by=("cik", "period", "accession_number"), **query)) as stream:
        for df_packet in _iter_whole_quarters(stream):
            yield clean_filing_dates(*_partition_packet(df_packet, Tables.sec13f_hr, manifest_columns, position_columns))
            del df_packet


def _iter_whole_quarters(stream: Iterator[pd.DataFrame]) -> Generator[pd.DataFrame]:
    """Carry only the trailing CIK-quarter, concatenating its pieces once at completion."""
    pieces: list[pd.DataFrame] = []
    previous_key: object = None
    groups = chain.from_iterable(df_chunk.groupby(["cik", "period"], sort=False) for df_chunk in stream)

    for key, df_piece in groups:
        if pieces and key != previous_key:
            yield pd.concat(pieces, ignore_index=True)
            pieces.clear()
        # A group view must not retain the other raw books in its source chunk.
        pieces.append(df_piece.copy())
        previous_key = key
        del df_piece
    if pieces:
        yield pd.concat(pieces, ignore_index=True)


def load_filing_vintages(
    store: DataStore, until: object, ciks: Sequence[str] | None = None, *, table: Table = Tables.sec13f_hr
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load a bounded compatibility scope; broad production uses iter_filing_vintages."""
    if table == Tables.sec13f_manager_holdings:
        return load_manager_filing_vintages(store, until, ciks)
    layout = _filing_layout(store, table, until, ciks)
    if layout is None:
        return None
    manifest_columns, position_columns, query = layout
    manifests, positions = [], []
    for df_chunk in store.iter_load(table, **query):
        df_filings, df_holdings = _partition_packet(df_chunk, table, manifest_columns, position_columns)
        manifests.append(df_filings)
        positions.append(df_holdings)
    df_filings = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame(columns=manifest_columns)
    df_holdings = pd.concat(positions, ignore_index=True) if positions else pd.DataFrame(columns=position_columns)
    return _finish_packet(df_filings, df_holdings)


def load_manager_filing_vintages(store: DataStore, until: object, ciks: Sequence[str] | None) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Read the bounded manager table once with ordinary projected, scoped loading."""
    table = Tables.sec13f_manager_holdings
    layout = _filing_layout(store, table, until, ciks)
    if layout is None:
        return None
    manifest_columns, position_columns, query = layout
    df_payload = store.load(table, optional=True, **query)
    if df_payload is None:
        logger.warning("No complete immutable 13F filings in scope; recover originals/amendments. Latest-only rows cannot supply PIT books.")
        return None
    df_filings, df_holdings = _partition_packet(df_payload, table, manifest_columns, position_columns)
    return _finish_packet(df_filings, df_holdings)
