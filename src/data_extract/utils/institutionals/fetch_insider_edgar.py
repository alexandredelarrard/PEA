"""Daily EDGAR ownership filings (Forms 3/4/5) into `insider_transactions`.

EDGAR is authoritative. Each run reads, one filing at a time, every Form 3/4/5 in the local EDGAR
index filed after the last stored zip quarter that the key has not yet stored from EDGAR (markers
included). The index lists a filing under its issuer and under every reporting owner, so a filing
whose XML issuer is not the key's company (the key is only an owner) becomes an empty-filing
marker, as does a holdings-only filing. A zip-sourced accession EDGAR re-reads is replaced whole
(its zip `quarter` is kept), so no accession holds rows from both sources. Rejected rows are never
stored, only summarised once per run.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any, cast

import pandas as pd

from src.constants.constants import SEC_INSIDER_FORMS
from src.context import Context
from src.data_extract.utils.common import edgar_index
from src.data_extract.utils.common.edgar_driver import EdgarFetch, EdgarScope, FilingStamp, run_edgar_fetch
from src.data_extract.utils.common.registrant import issuer_ciks, listing_ciks, resolve_registrant_entries
from src.data_extract.utils.common.resume import DONE_PER_KEY
from src.data_extract.utils.common.sec_io import ParseFailureError, TransientReadError, filing_header, filing_xml
from src.data_extract.utils.institutionals.insider_common import (
    INSIDER_COLUMNS,
    INSIDER_KEY,
    XML_DATE_FORMATS,
    accession_batches,
    build_insider_frame,
    empty_footnotes,
    exclusion_rows,
    filter_footnotes,
    log_exclusions,
    screen_insider_rows,
)
from src.data_extract.utils.institutionals.insider_edgar_parser import extract_xml_strings
from src.data_store.schema import Table, Tables
from src.utils.string import pad_cik

#: EDGAR rows carry the whole contract except `quarter`, so the merge-upsert keeps a stored zip quarter.
_EDGAR_COLUMNS = tuple(column for column in INSIDER_COLUMNS if column != "quarter")
#: The stored rows whose accessions count as read by EDGAR (markers carry `source='edgar'` too).
_EDGAR_DONE: dict[str, object] = {"source": "edgar"}
_LOG = logging.getLogger(__name__)


def insider_filings(
    context: Context,
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None,
    through: pd.Timestamp,
    scope: EdgarScope,
) -> list[Any]:
    """`ticker`'s Forms 3/4/5 in the local EDGAR index filed in `[since, through]`, oldest first (issuer and owner roles)."""
    df_index = edgar_index.entries(context, set(listing_ciks(ticker, cik, scope.registrants, scope.identity)), SEC_INSIDER_FORMS, since=since)
    df_rows = resolve_registrant_entries(ticker, cik, df_index, SEC_INSIDER_FORMS, registrants=scope.registrants, identity=scope.identity)
    df_rows = df_rows[df_rows["filed"] <= pd.Timestamp(through).normalize()]
    return [edgar_index.index_filing(r.cik, r.company, r.form, r.filed, r.accession) for r in df_rows.itertuples(index=False)]


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


def _filing_strings(filing: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One ownership filing -> (transaction strings, owner strings, footnotes), keyed on its accession.

    A filing without ownership XML raises `ParseFailureError`."""
    xml = filing_xml(filing)
    if not xml:
        raise ParseFailureError(f"{getattr(filing, 'accession_number', '?')}: no ownership XML")
    return extract_xml_strings(xml, str(filing.accession_number))


def _filing_metadata(filing: Any) -> dict[str, object]:
    return {
        "accession_number": str(filing.accession_number),
        "filing_date": pd.Timestamp(filing.filing_date).normalize(),
        "acceptance_datetime": _acceptance_datetime(filing),
    }


def _owner_role(df_str: pd.DataFrame, ticker: str, cik: str, scope: EdgarScope) -> bool:
    """True when the XML names an issuer CIK outside the key's lineage (the key is only a reporting owner)."""
    issuer = pad_cik(df_str["issuer_cik"].dropna().iloc[0]) if "issuer_cik" in df_str and df_str["issuer_cik"].notna().any() else ""
    lineage = issuer_ciks(ticker, cik, scope.registrants, scope.identity)
    return bool(issuer and lineage and issuer not in lineage)


def parse_insider(
    ticker: str, cik: str, stamp: FilingStamp, scope: EdgarScope, *, universe: Sequence[str], excluded: list[pd.DataFrame]
) -> dict[Table, pd.DataFrame]:
    """One ownership filing -> its kept `source='edgar'` rows and their footnotes; nothing for a
    holdings-only filing or one where the key is only an owner. Rejected in-scope rows go to `excluded`."""
    if scope.identity is None:
        raise ValueError("insider EDGAR fetch needs an identity-aware scope")
    df_str, df_owners, df_notes = _filing_strings(stamp.filing)
    if df_str.empty or _owner_role(df_str, ticker, cik, scope):
        return {}
    df_built = build_insider_frame(df_str, df_owners, date_formats=XML_DATE_FORMATS)
    df_built = df_built.merge(pd.DataFrame([_filing_metadata(stamp.filing)]), on="accession_number", how="left")
    df_kept, df_rejected = screen_insider_rows(df_built.assign(source="edgar"), universe, scope.identity)
    if not df_rejected.empty:
        excluded.append(exclusion_rows(df_rejected))
    df_rows = pd.DataFrame(columns=list(_EDGAR_COLUMNS))
    df_kept_notes = empty_footnotes()
    if not df_kept.empty:
        df_stamped = df_kept.assign(fetched_at=pd.Timestamp.now(tz="UTC").tz_localize(None))
        df_rows = df_stamped[[c for c in _EDGAR_COLUMNS if c in df_stamped.columns]].drop_duplicates(subset=INSIDER_KEY, keep="last")
        df_kept_notes = filter_footnotes(df_notes, set(df_kept["accession_number"]))
    return {Tables.insider_footnotes: df_kept_notes, Tables.insider_transactions: df_rows}


def bulk_frontier_floor(context: Context) -> pd.Timestamp | None:
    """The day after the last stored zip quarter (EDGAR reads only what follows); None when no quarter is stored."""
    if "quarter" not in context.store.columns(Tables.insider_transactions):
        return None
    _, latest_quarter = context.store.bounds(Tables.insider_transactions, "quarter")
    if latest_quarter is None:
        return None
    try:
        return pd.Period(str(latest_quarter).upper(), freq="Q").end_time.normalize() + pd.Timedelta(days=1)
    except (TypeError, ValueError):
        return None


def insider_fetch(tickers: Sequence[str], excluded: list[pd.DataFrame] | None = None) -> EdgarFetch:
    """The EDGAR Forms 3/4/5 fetch for universe `tickers`. Done per key and only on EDGAR rows: an
    owner-role marker under one company never hides the filing from its issuer, and a zip row never
    makes a filing look read. Rejected rows are appended to `excluded`."""
    return EdgarFetch(
        desc="insider Forms 3/4/5 (EDGAR)",
        tables=(Tables.insider_footnotes, Tables.insider_transactions),
        forms=tuple(SEC_INSIDER_FORMS),
        parse=partial(parse_insider, universe=list(tickers), excluded=excluded if excluded is not None else []),
        done_table=Tables.insider_transactions,
        done_scope=DONE_PER_KEY,
        done_where=_EDGAR_DONE,
        runtime_floor=bulk_frontier_floor,
    )


def replace_zip_accessions(context: Context, since: pd.Timestamp) -> int:
    """Finish EDGAR's replacement of zip-sourced accessions filed since `since`: stamp each
    accession's stored zip `quarter` on its EDGAR rows that lack one (rows EDGAR added beyond the
    zip's keys), then delete the zip rows of every accession EDGAR now holds. Returns the rows deleted.
    A table without a `quarter` column holds no zip row, so there is nothing to replace."""
    if "quarter" not in context.store.columns(Tables.insider_transactions):
        context.log.info("insider EDGAR: no zip row stored (no `quarter` column) -> nothing to replace")
        return 0
    df_rows = context.store.load(
        Tables.insider_transactions, columns=[*INSIDER_KEY, "source", "quarter"], since=since, date_col="filing_date", optional=True
    )
    if df_rows is None:
        return 0
    quarter_of = df_rows.dropna(subset=["quarter"]).groupby("accession_number")["quarter"].first()
    df_stamp = df_rows.loc[df_rows["source"].eq("edgar") & df_rows["quarter"].isna(), INSIDER_KEY]
    df_stamp = df_stamp.assign(quarter=df_stamp["accession_number"].map(quarter_of)).dropna(subset=["quarter"])
    if not df_stamp.empty:
        context.store.save(Tables.insider_transactions, df_stamp)
    edgar_accessions = set(df_rows.loc[df_rows["source"].eq("edgar"), "accession_number"])
    mixed = sorted(edgar_accessions & set(df_rows.loc[df_rows["source"].eq("zip"), "accession_number"]))
    deleted = sum(
        context.store.delete(Tables.insider_transactions, where={"accession_number": batch, "source": "zip"}) for batch in accession_batches(mixed)
    )
    context.log.info(
        "insider EDGAR: stamped the zip quarter on %d EDGAR row(s); replaced %d zip-sourced filing(s) whole (%d zip row(s) deleted)",
        len(df_stamp),
        len(mixed),
        deleted,
    )
    return deleted


def fetch_insider_edgar(
    context: Context,
    tickers: list[str],
    years_history: int,
    *,
    full: bool = False,
    as_of: pd.Timestamp | None = None,
    no_cap: bool = False,
) -> None:
    """Read every indexed Form 3/4/5 after the last zip quarter not yet stored from EDGAR (all of them
    under `full`); afterwards, even on failure, log the run's exclusions and replace the zip rows of
    every accession EDGAR now holds."""
    excluded: list[pd.DataFrame] = []
    since = bulk_frontier_floor(context)
    try:
        run_edgar_fetch(context, tickers, years_history, insider_fetch(tickers, excluded), full=full, as_of=as_of, no_cap=no_cap)
    except BaseException:
        _finish_run(context, since, excluded, after_failure=True)
        raise
    _finish_run(context, since, excluded, after_failure=False)


def _finish_run(context: Context, since: pd.Timestamp | None, excluded: list[pd.DataFrame], *, after_failure: bool) -> None:
    """Log the run's exclusions and replace the zip rows it re-read (none without a stored zip
    quarter). After a failed run a reconcile error is logged, never raised, so the run's own error propagates."""
    log_exclusions(_LOG, "EDGAR run", excluded)
    if since is None:
        return
    if not after_failure:
        replace_zip_accessions(context, since)
        return
    try:
        replace_zip_accessions(context, since)
    except Exception:
        _LOG.exception("insider EDGAR: the zip reconcile after a failed run raised; the run's own error follows")
