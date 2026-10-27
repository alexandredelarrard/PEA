"""Daily EDGAR ownership filings (Forms 3/4/5) into `insider_transactions`.

EDGAR is authoritative. Each run reads, one filing at a time, every Form 3/4/5 in the local EDGAR
index filed since the earlier of the day after the last stored zip quarter and 7 days before the
run date, that the key has not yet stored from EDGAR (markers included). The index lists a filing under its issuer and under every reporting owner, so a filing
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
from src.data_extract.utils.common.edgar_driver import EdgarFetch, EdgarScope, FilingStamp
from src.data_extract.utils.common.resume import DONE_PER_KEY
from src.data_extract.utils.common.sec_io import ParseFailureError, TransientReadError, filing_header, filing_xml
from src.data_extract.utils.institutionals.insider_common import (
    INSIDER_COLUMNS,
    INSIDER_KEY,
    XML_DATE_FORMATS,
    build_insider_frame,
    empty_footnotes,
    exclusion_rows,
    filter_footnotes,
    log_exclusions,
    screen_insider_rows,
)
from src.data_extract.utils.institutionals.insider_edgar_parser import extract_xml_strings
from src.data_extract.utils.institutionals.insider_store import insider_source_run, replace_insider_source
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


def _owner_role(df_str: pd.DataFrame, ticker: str, cik: str, scope: EdgarScope) -> bool:
    """True when the XML names an issuer CIK outside the key's lineage (the key is only a reporting owner)."""
    issuer = pad_cik(df_str["issuer_cik"].dropna().iloc[0]) if "issuer_cik" in df_str and df_str["issuer_cik"].notna().any() else ""
    lineage = frozenset(scope.filing_scope(ticker, cik).event_ciks)
    return bool(issuer and lineage and issuer not in lineage)


def parse_insider(
    ticker: str, cik: str, stamp: FilingStamp, scope: EdgarScope, *, universe: Sequence[str], excluded: list[pd.DataFrame]
) -> dict[Table, pd.DataFrame]:
    """One ownership filing -> its kept `source='edgar'` rows and their footnotes; nothing for a
    holdings-only filing or one where the key is only an owner. Rejected in-scope rows go to `excluded`."""
    if scope.identity is None:
        raise ValueError("insider EDGAR fetch needs an identity-aware scope")
    xml = filing_xml(stamp.filing)
    if not xml:
        raise ParseFailureError(f"{stamp.accession_number}: no ownership XML")
    try:
        df_str, df_owners, df_notes = extract_xml_strings(xml, str(stamp.accession_number))
    except (ElementTree.ParseError, ValueError) as exc:
        raise ParseFailureError(f"{stamp.accession_number}: invalid ownership XML ({exc})") from exc
    if df_str.empty:
        return {Tables.insider_footnotes: df_notes}
    if _owner_role(df_str, ticker, cik, scope):
        return {}
    df_built = build_insider_frame(df_str, df_owners, date_formats=XML_DATE_FORMATS)
    df_built = df_built.assign(filing_date=stamp.filed.normalize(), acceptance_datetime=_acceptance_datetime(stamp.filing))
    df_kept, df_rejected = screen_insider_rows(df_built.assign(source="edgar"), universe, scope.identity)
    if not df_rejected.empty:
        excluded.append(exclusion_rows(df_rejected))
    df_rows = pd.DataFrame(columns=list(_EDGAR_COLUMNS))
    df_kept_notes = empty_footnotes()
    if not df_kept.empty:
        df_stamped = df_kept.assign(fetched_at=pd.Timestamp.now(tz="UTC").tz_localize(None))
        df_rows = df_stamped[[c for c in _EDGAR_COLUMNS if c in df_stamped.columns]]
        df_kept_notes = filter_footnotes(df_notes, set(df_kept["accession_number"]))
    return {Tables.insider_footnotes: df_kept_notes, Tables.insider_transactions: df_rows}


def edgar_window_start(context: Context, as_of: pd.Timestamp) -> pd.Timestamp | None:
    """Earlier of the day after the last zip quarter and the run date minus the registry overlap;
    None without a valid stored quarter, so the ordinary history floor applies."""
    if "quarter" not in context.store.columns(Tables.insider_transactions):
        return None
    _, latest_quarter = context.store.bounds(Tables.insider_transactions, "quarter")
    if latest_quarter is None:
        return None
    try:
        floor = pd.Period(str(latest_quarter).upper(), freq="Q").end_time.normalize() + pd.Timedelta(days=1)
    except (TypeError, ValueError):
        return None
    overlap = Tables.insider_transactions.resume.overlap_days if Tables.insider_transactions.resume is not None else 0
    return min(floor, pd.Timestamp(as_of).normalize() - pd.Timedelta(days=overlap))


def replace_zip_accessions(context: Context, since: pd.Timestamp) -> int:
    """Finish EDGAR's replacement of zip-sourced accessions filed since `since`: stamp each
    accession's stored zip `quarter` on its EDGAR rows that lack one (rows EDGAR added beyond the
    zip's keys), then delete the zip rows of every accession EDGAR now holds. Returns the rows deleted.
    A table without a `quarter` column holds no zip row, so there is nothing to replace."""
    if "quarter" not in context.store.columns(Tables.insider_transactions):
        context.log.info("insider EDGAR: no zip row stored (no `quarter` column) -> nothing to replace")
        return 0
    df_rows = context.store.load(
        Tables.insider_transactions,
        columns=context.store.columns(Tables.insider_transactions),
        since=since,
        date_col="filing_date",
        optional=True,
        markers=True,
    )
    if df_rows is None:
        return 0
    notes = context.store.load(
        Tables.insider_footnotes,
        columns=list(Tables.insider_footnotes.retrieval_columns),
        where={"accession_number": df_rows["accession_number"].dropna().unique().tolist()},
        optional=True,
    )
    quarter_of = df_rows.dropna(subset=["quarter"]).groupby("accession_number")["quarter"].first()
    df_stamp = df_rows.loc[df_rows["source"].eq("edgar") & df_rows["quarter"].isna(), INSIDER_KEY]
    df_stamp = df_stamp.assign(quarter=df_stamp["accession_number"].map(quarter_of)).dropna(subset=["quarter"])
    edgar_accessions = set(df_rows.loc[df_rows["source"].eq("edgar"), "accession_number"])
    mixed = sorted(edgar_accessions & set(df_rows.loc[df_rows["source"].eq("zip"), "accession_number"]))
    affected = sorted(set(mixed) | set(df_stamp["accession_number"]))
    deleted = int((df_rows["accession_number"].isin(mixed) & df_rows["source"].eq("zip")).sum())
    if affected:
        notes = notes.loc[notes["accession_number"].isin(affected)] if notes is not None else None
        payload = df_rows.loc[df_rows["accession_number"].isin(affected) & df_rows["source"].eq("edgar")].copy()
        payload["quarter"] = payload["quarter"].fillna(payload["accession_number"].map(quarter_of))
        replace_insider_source(
            context.store,
            {Tables.insider_transactions: payload, Tables.insider_footnotes: notes if notes is not None else empty_footnotes()},
            {Tables.insider_transactions: {"accession_number": affected}, Tables.insider_footnotes: {"accession_number": affected}},
            expected_source={
                Tables.insider_transactions: df_rows.loc[df_rows["accession_number"].isin(affected)],
                Tables.insider_footnotes: notes if notes is not None else empty_footnotes(),
            },
        )
    context.log.info(
        "insider EDGAR: stamped the zip quarter on %d EDGAR row(s); replaced %d zip-sourced filing(s) whole (%d zip row(s) deleted)",
        len(df_stamp),
        len(mixed),
        deleted,
    )
    return deleted


@contextmanager
def _insider_run_scope(context: Context, tickers: Sequence[str], as_of: pd.Timestamp) -> Iterator[EdgarFetch]:
    """Bind the run's requested tickers, exclusions and one ZIP floor; reconcile on every exit."""
    excluded: list[pd.DataFrame] = []
    since = edgar_window_start(context, as_of)
    overlap = Tables.insider_transactions.resume.overlap_days if Tables.insider_transactions.resume is not None else 0
    recent = since if since is not None else pd.Timestamp(as_of).normalize() - pd.Timedelta(days=overlap)
    failed = True
    try:
        with insider_source_run(context):
            yield replace(
                INSIDER_EDGAR_FETCH,
                parse=partial(parse_insider, universe=tuple(tickers), excluded=excluded),
                runtime_floor=lambda _context, _as_of: recent,
            )
        failed = False
    finally:
        _finish_run(context, since, excluded, after_failure=failed)


def _finish_run(context: Context, since: pd.Timestamp | None, excluded: list[pd.DataFrame], *, after_failure: bool) -> None:
    """Log the run's exclusions and replace the zip rows it re-read (none without a stored zip
    quarter). After a failed run a reconcile error is logged, never raised, so the run's own error propagates."""
    log_exclusions(context.log, "EDGAR run", excluded)
    if since is None:
        return
    if not after_failure:
        replace_zip_accessions(context, since)
        return
    try:
        replace_zip_accessions(context, since)
    except Exception:
        context.log.exception("insider EDGAR: the zip reconcile after a failed run raised; the run's own error follows")


INSIDER_EDGAR_FETCH = EdgarFetch(
    tables=(Tables.insider_footnotes, Tables.insider_transactions),
    done_table=Tables.insider_transactions,
    done_scope=DONE_PER_KEY,
    done_where={"source": "edgar"},
    runtime_floor=edgar_window_start,
    run_scope=_insider_run_scope,
)
