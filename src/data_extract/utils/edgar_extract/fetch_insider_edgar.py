"""Daily EDGAR ownership filings (Forms 3/4/5) into `insider_transactions`.

EDGAR is authoritative. Each run reads, one filing at a time, every Form 3/4/5 in the local EDGAR
index inside its configured filing Window that the key has not stored from either complete source (markers included). The index lists a filing under its issuer and under every reporting owner, so a filing
whose XML issuer is not the key's company (the key is only an owner) becomes an empty-filing
marker, as does a holdings-only filing. A zip-sourced accession EDGAR re-reads is replaced whole
(its zip `quarter` is kept), so no accession holds rows from both sources. Rejected rows are never
stored, only summarised once per run.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from typing import Any, cast
from xml.etree import ElementTree

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.identity import Identity
from src.data_extract.utils.common.resume import DONE_PER_KEY
from src.data_extract.utils.common.sec_io import ParseFailureError, TransientReadError, filing_header, filing_xml
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarFetch, FilingStamp
from src.data_extract.utils.edgar_extract.insider_common import (
    INSIDER_COLUMNS,
    XML_DATE_FORMATS,
    build_insider_frame,
    empty_footnotes,
    exclusion_rows,
    filter_footnotes,
    log_exclusions,
    screen_insider_rows,
)
from src.data_extract.utils.edgar_extract.insider_edgar_parser import extract_xml_strings
from src.data_extract.utils.edgar_extract.insider_store import save_insider_source
from src.data_store.schema import Table, Tables
from src.utils.string import pad_cik

#: EDGAR rows carry the whole contract except `quarter`, so the merge-upsert keeps a stored zip quarter.
_EDGAR_COLUMNS = tuple(column for column in INSIDER_COLUMNS if column != "quarter")


def _acceptance_datetime(filing: object) -> pd.Timestamp:
    """The header's acceptance time, timezone-naive (NaT when the header has none); a transient read raises."""
    try:
        raw = getattr(filing_header(filing), "acceptance_datetime", None)
    except TransientReadError:
        raise
    except Exception:  # noqa: BLE001 -- optional EDGAR header metadata
        raw = None
    value = pd.to_datetime(cast(Any, raw), errors="coerce")
    if pd.notna(value) and value.tz is not None:
        value = value.tz_localize(None)
    return value


def _owner_role(df_str: pd.DataFrame, ticker: str, identity: Identity) -> bool:
    """True when the XML names an issuer CIK outside the key's lineage (the key is only a reporting owner)."""
    issuer = pad_cik(df_str["issuer_cik"].dropna().iloc[0]) if "issuer_cik" in df_str and df_str["issuer_cik"].notna().any() else ""
    lineage = frozenset(identity.filing_scope(ticker).event_ciks)
    return bool(issuer and lineage and issuer not in lineage)


def parse_insider(
    ticker: str, cik: str, stamp: FilingStamp, identity: Identity, *, universe: Sequence[str], excluded: list[pd.DataFrame]
) -> dict[Table, pd.DataFrame]:
    """One ownership filing -> its kept `source='edgar'` rows and their footnotes; nothing for a
    holdings-only filing or one where the key is only an owner. Rejected in-scope rows go to `excluded`."""
    xml = filing_xml(stamp.filing)
    if not xml:
        raise ParseFailureError(f"{stamp.accession_number}: no ownership XML")
    try:
        df_str, df_owners, df_notes = extract_xml_strings(xml, str(stamp.accession_number))
    except (ElementTree.ParseError, ValueError) as exc:
        raise ParseFailureError(f"{stamp.accession_number}: invalid ownership XML ({exc})") from exc
    if df_str.empty:
        return {Tables.insider_footnotes: df_notes}
    if _owner_role(df_str, ticker, identity):
        return {}
    df_built = build_insider_frame(df_str, df_owners, date_formats=XML_DATE_FORMATS)
    df_built = df_built.assign(filing_date=stamp.filed.normalize(), acceptance_datetime=_acceptance_datetime(stamp.filing))
    df_kept, df_rejected = screen_insider_rows(df_built.assign(source="edgar"), universe, identity)
    if not df_rejected.empty:
        excluded.append(exclusion_rows(df_rejected))
    df_rows = pd.DataFrame(columns=list(_EDGAR_COLUMNS))
    df_kept_notes = empty_footnotes()
    if not df_kept.empty:
        df_stamped = df_kept.assign(fetched_at=pd.Timestamp.now(tz="UTC").tz_localize(None))
        df_rows = df_stamped[[c for c in _EDGAR_COLUMNS if c in df_stamped.columns]]
        df_kept_notes = filter_footnotes(df_notes, set(df_kept["accession_number"]))
    return {Tables.insider_footnotes: df_kept_notes, Tables.insider_transactions: df_rows}


def save_insider_key(context: Context, fetch: EdgarFetch, ticker: str, frames: dict[Table, list[pd.DataFrame]]) -> tuple[dict[Table, int], bool]:
    """Save one key's transactions and footnotes as one insider source pair: `(rows per table, failed_save)`.
    An empty-filing marker is stamped `fetched_at` here, as the parsed rows are when read."""
    transaction_frames = frames.get(Tables.insider_transactions, [])
    if not transaction_frames:
        return {}, False
    transactions = pd.concat(transaction_frames, ignore_index=True).drop_duplicates(list(Tables.insider_transactions.pk), keep="last")
    fetched = transactions["fetched_at"] if "fetched_at" in transactions else pd.Series(pd.NaT, index=transactions.index)
    transactions["fetched_at"] = pd.to_datetime(fetched).fillna(pd.Timestamp.now(tz="UTC").tz_localize(None))
    note_frames = frames.get(Tables.insider_footnotes, [])
    notes = pd.concat(note_frames, ignore_index=True) if note_frames else pd.DataFrame(columns=list(Tables.insider_footnotes.retrieval_columns))
    try:
        return save_insider_source(context, transactions, notes), False
    except Exception as exc:
        context.log.warning("Insider source pair not saved; all accessions remain retryable (%s)", exc)
        return {}, True


@contextmanager
def _insider_run_scope(context: Context, tickers: Sequence[str]) -> Iterator[EdgarFetch]:
    """Bind only this run's scope and exclusion collector; exact accession writes own reconciliation."""
    excluded: list[pd.DataFrame] = []
    try:
        yield replace(INSIDER_EDGAR_FETCH, parse=partial(parse_insider, universe=tuple(tickers), excluded=excluded))
    finally:
        log_exclusions("EDGAR run", excluded)


INSIDER_EDGAR_FETCH = EdgarFetch(
    tables=(Tables.insider_footnotes, Tables.insider_transactions),
    done_table=Tables.insider_transactions,
    done_scope=DONE_PER_KEY,
    run_scope=_insider_run_scope,
    save=save_insider_key,
)
