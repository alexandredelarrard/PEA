"""Reads of immutable `sec13f_hr` manifests and whole-book positions."""

from __future__ import annotations

import logging
from collections.abc import Collection, Generator, Iterator, Mapping, Sequence
from contextlib import closing
from typing import Any, cast

import pandas as pd

from src.context import Context
from src.data_aggregate.utils.sec_thirteen.holdings_clean import clean_filings
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.data_policy import source_read_window
from src.utils.institutional_vintages import single_version_books
from src.utils.string import pad_cik
from src.utils.window import Window

logger = logging.getLogger(__name__)

_CHUNKSIZE = 500_000
_MANIFEST_COLUMNS = (
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
#: Read when the table carries them: the filer's own cover total, the check that a cleaned book kept its value.
_OPTIONAL_MANIFEST_COLUMNS = ("declared_value_total",)
_NUMERIC = ("shares", "value_usd", "call_shares", "call_value", "put_shares", "put_value", "debt_prn", "debt_value", "other_value")
_POSITION_COLUMNS = (
    "cik",
    "accession_number",
    "cusip",
    "period",
    "filing_date",
    "position_type",
    "class_evidence",
    *_NUMERIC,
    *("raw_" + column for column in _NUMERIC if "value" in column),
    "ticker",
)
#: Read when the table carries them: the text the bond-line mapping reads, dropped once it has run.
_TEXT_COLUMNS = ("issuer_name", "title_of_class")


def _filing_layout(
    store: DataStore, window: Window, ciks: Sequence[str] | None, universe: Sequence[str] | None, cusips: Collection[str]
) -> dict[str, Any] | None:
    """Validate the source layout and build its projected public-prefix query: whole books, or with `universe`
    every completion marker OR'd with the positions stored under a universe ticker or on one of `cusips` (the universe
    companies' `universe_cusip` CUSIPs, so a class filed under its own market ticker is read too)."""
    stored = set(store.columns(Tables.sec13f_hr))
    if not set(_MANIFEST_COLUMNS) <= stored:
        logger.warning("%s has no completed accession schema; replay original sources before building PIT features.", Tables.sec13f_hr.name)
        return None
    scope: dict[str, Any] = {} if ciks is None else {"cik": sorted({pad_cik(cik) for cik in ciks})}
    where: dict[str, Any] | list[dict[str, Any]] = {"parse_status": ["complete", "position"], **scope}
    if universe is not None:
        markers = {"parse_status": "complete", "cusip": cast(tuple[str, str], Tables.sec13f_hr.empty_marker)[1], **scope}
        positions = [
            {"parse_status": "position", "ticker": sorted(set(universe)), **scope},
            {"parse_status": "position", "cusip": sorted(set(cusips)), **scope},
        ]
        where = [markers, *positions]
    optional = (c for c in (*_OPTIONAL_MANIFEST_COLUMNS, *_TEXT_COLUMNS) if c in stored)
    columns = list(dict.fromkeys((*_MANIFEST_COLUMNS, *optional, *_POSITION_COLUMNS)))
    return {"columns": columns, "where": where, "since": window.start, "until": window.end, "date_col": "filing_date", "markers": True}


def _partition_packet(df_payload: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate verified completion markers from economic position rows."""
    complete = df_payload["parse_status"].eq("complete") & df_payload["cusip"].eq(cast(tuple[str, str], Tables.sec13f_hr.empty_marker)[1])
    df_filings = df_payload.loc[complete, [*_MANIFEST_COLUMNS, *(c for c in _OPTIONAL_MANIFEST_COLUMNS if c in df_payload)]].reset_index(drop=True)
    positions = [*_POSITION_COLUMNS, *(c for c in _TEXT_COLUMNS if c in df_payload)]
    df_holdings = df_payload.loc[df_payload["parse_status"].eq("position"), positions].reset_index(drop=True)
    return df_filings, df_holdings


def iter_filing_vintages(
    context: Context,
    until: object,
    ciks: Sequence[str] | None = None,
    *,
    bond_lines: Mapping[str, Any],
    universe: Sequence[str] | None = None,
    cusips: Collection[str] = frozenset(),
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]] | None:
    """Open one projected source stream of many complete CIK-quarter groups. With `universe`, a single-version book
    carries only its positions under a universe ticker or on one of `cusips` (the extraction writes positions and
    marker atomically, so it is complete) and an amended book carries all its positions, so the resolver's stored-row
    count check still holds. Bond lines (`bond_lines`, `data_aggregate.bond_lines`) leave every packet as the debt rows do."""
    retained = source_read_window(context.config, Tables.sec13f_hr, min(context.run.as_of, pd.Timestamp(until)))
    if retained is None:
        return None
    store = context.store
    query = _filing_layout(store, retained, ciks, universe, cusips)
    if query is None:
        return None
    return _iter_filing_packets(context, query, bond_lines, scoped=universe is not None)


def _iter_filing_packets(
    context: Context, query: dict[str, Any], bond_lines: Mapping[str, Any], *, scoped: bool
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Partition verified markers and positions while owning the source cursor."""
    store = context.store
    with closing(store.iter_load(Tables.sec13f_hr, chunksize=_CHUNKSIZE, order_by=("cik", "period", "accession_number"), **query)) as stream:
        for df_packet in _iter_whole_quarters(stream):
            df_packet = _complete_amended_books(store, query, df_packet) if scoped else df_packet
            yield clean_filings(*_partition_packet(df_packet), bond_lines, context.config)
            del df_packet


def _complete_amended_books(store: DataStore, query: dict[str, Any], df_packet: pd.DataFrame) -> pd.DataFrame:
    """On a universe-scoped (OR) read, replace the positions of every amended book (any book that is not exactly one
    complete original/restatement) by all its stored positions: one keyed read per batch, index-backed on `cik`."""
    # Amended books from the batch's completion markers (the read's only complete rows); whole quarters never span batches.
    df_markers = df_packet.loc[df_packet["parse_status"].eq("complete")]
    df_amended = df_markers.loc[~single_version_books(df_markers)]
    if df_amended.empty:
        return df_packet

    # Every stored position of those accessions replaces their universe slice.
    where = {
        "cik": sorted(set(df_amended["cik"])),
        "period": sorted(set(df_amended["period"])),
        "accession_number": sorted(df_amended["accession_number"]),
        "parse_status": "position",
    }
    df_full = store.load(
        Tables.sec13f_hr,
        columns=query["columns"],
        where=where,
        since=query["since"],
        until=query["until"],
        date_col="filing_date",
        optional=True,
        markers=True,
    )
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
