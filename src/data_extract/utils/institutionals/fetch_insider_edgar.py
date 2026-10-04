"""Daily EDGAR ownership filings (Forms 3/4/5) into `insider_transactions`.

EDGAR is authoritative. Each run lists filings from the latest stored `filing_date` minus 7 days
and skips accessions already stored from EDGAR; a zip-sourced accession it re-reads is replaced
whole (its zip `quarter` is kept), so no accession holds rows from both sources. Rejected rows are
never stored, only summarised once per run.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any, cast

import pandas as pd
from edgar import Filing

from src.constants.constants import SEC_INSIDER_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import EdgarFetch, EdgarScope, run_edgar_fetch
from src.data_extract.utils.common.identity import Identity, load_identity
from src.data_extract.utils.common.registrant import resolve_registrant_filings
from src.data_extract.utils.common.sec_atom import (
    SEC_INSIDER_FORM_FAMILIES,
    AtomEntry,
    AtomPageError,
    atom_filing,
    iter_atom_pages,
    keep_atom_entry,
)
from src.data_extract.utils.institutionals.insider_common import (
    INSIDER_COLUMNS,
    INSIDER_KEY,
    OWNER_STRING_COLUMNS,
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
#: Calendar days the listing window reaches back before the latest stored filing date.
_LISTING_OVERLAP_DAYS = 7
_LOG = logging.getLogger(__name__)


def ownership_filings(
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None,
    through: pd.Timestamp,
    done_accessions: frozenset[str],
) -> list[Filing]:
    """List issuer ownership filings, including forms submitted under an owner's CIK."""
    start_date = pd.Timestamp(since).normalize() if since is not None else None
    end_date = pd.Timestamp(through).normalize()
    filings: dict[str, Filing] = {}
    for family in SEC_INSIDER_FORM_FAMILIES:
        filings.update(_family_filings(ticker, cik, family, start_date, end_date, done_accessions))
    return sorted(filings.values(), key=lambda filing: filing.filing_date)


def _family_filings(
    ticker: str,
    cik: str,
    family: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp,
    done_accessions: frozenset[str],
) -> dict[str, Filing]:
    """Page one form family newest-first until a short page or an entry older than `start_date`.

    A failed page logs a warning and ends this family with the pages already read.
    """
    filings: dict[str, Filing] = {}
    try:
        for _offset, entries in iter_atom_pages(pad_cik(cik), family, start_date, end_date, ticker, retry=False):
            page, oldest = _page_filings(entries, ticker, cik, start_date, end_date, done_accessions)
            filings.update(page)
            if start_date is not None and oldest < start_date:
                break
    except AtomPageError as exc:  # preserve other discovery channels
        _LOG.warning("ownership filing search failed for %s form %s at offset %d: %r", ticker, family, exc.offset, exc.__cause__)
    return filings


def _page_filings(
    entries: list[AtomEntry | None],
    ticker: str,
    cik: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp,
    done_accessions: frozenset[str],
) -> tuple[dict[str, Filing], pd.Timestamp]:
    """Kept insider filings on one page by accession, and the page's oldest dated entry (capped at `end_date`)."""
    target_forms = frozenset(SEC_INSIDER_FORMS)
    filings: dict[str, Filing] = {}
    oldest = end_date
    for entry in entries:
        if entry is None:
            continue
        oldest = min(oldest, entry.filing_date)
        if keep_atom_entry(entry, target_forms, done_accessions, start_date, end_date):
            filings[cast(str, entry.accession)] = atom_filing(entry, cik=cik, company=ticker)
    return filings, oldest


def insider_filings(
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None,
    through: pd.Timestamp,
    done_accessions: frozenset[str],
    scope: EdgarScope,
) -> list[Any]:
    """Union issuer submissions with the owner-inclusive issuer search."""
    issuer_filings = resolve_registrant_filings(
        ticker,
        SEC_INSIDER_FORMS,
        since=since,
        done_accessions=done_accessions,
        registrants=scope.registrants,
        identity=scope.identity,
    )
    discovered: dict[str, Any] = {str(filing.accession_number): filing for filing in issuer_filings}
    for filing in ownership_filings(
        ticker,
        cik,
        since=since,
        through=through,
        done_accessions=done_accessions,
    ):
        discovered.setdefault(str(filing.accession_number), filing)
    return sorted(
        discovered.values(),
        key=lambda filing: pd.Timestamp(filing.filing_date),
    )


def _acceptance_datetime(filing: object) -> pd.Timestamp:
    """The filing header's acceptance time, timezone-naive; NaT when the header has none."""
    filing_obj = cast(Any, filing)
    try:
        raw = getattr(filing_obj.header, "acceptance_datetime", None)
    except Exception:  # noqa: BLE001 -- optional EDGAR header metadata
        raw = None
    value = pd.to_datetime(cast(Any, raw), errors="coerce")
    if pd.notna(value) and value.tz is not None:
        value = value.tz_localize(None)
    return value


def _ticker_strings(filings: Sequence[Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ownership filings -> (transaction strings, owner strings, footnotes, filing metadata), keyed on accession."""
    transaction_frames: list[pd.DataFrame] = []
    owner_frames: list[pd.DataFrame] = []
    footnote_frames: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    for filing in filings:
        xml = filing.xml()
        if not xml:
            raise ValueError(f"{getattr(filing, 'accession_number', '?')}: no ownership XML")
        accession = str(filing.accession_number)
        df_str, df_owners, df_notes = extract_xml_strings(xml, accession)
        if df_str.empty:
            continue
        transaction_frames.append(df_str)
        owner_frames.append(df_owners)
        if not df_notes.empty:
            footnote_frames.append(df_notes)
        metadata.append(
            {
                "accession_number": accession,
                "filing_date": pd.Timestamp(filing.filing_date).normalize(),
                "acceptance_datetime": _acceptance_datetime(filing),
            }
        )
    df_str = pd.concat(transaction_frames, ignore_index=True) if transaction_frames else pd.DataFrame()
    df_owners = pd.concat(owner_frames, ignore_index=True) if owner_frames else pd.DataFrame(columns=OWNER_STRING_COLUMNS)
    df_notes = pd.concat(footnote_frames, ignore_index=True) if footnote_frames else empty_footnotes()
    return df_str, df_owners, df_notes, pd.DataFrame(metadata)


def _screen_edgar_rows(
    df_str: pd.DataFrame, df_owners: pd.DataFrame, df_meta: pd.DataFrame, universe: Sequence[str], identity: Identity
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Type the string rows, attach filing metadata and `source='edgar'`, then screen into (kept, rejected in scope)."""
    df_built = build_insider_frame(df_str, df_owners, date_formats=XML_DATE_FORMATS)
    df_built = df_built.merge(df_meta.drop_duplicates("accession_number", keep="last"), on="accession_number", how="left")
    return screen_insider_rows(df_built.assign(source="edgar"), universe, identity)


def _edgar_frames(filings: Sequence[Any], *, universe: Sequence[str], identity: Identity) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ownership filings -> (kept transactions, footnotes of kept accessions, rejected in-scope rows)."""
    df_str, df_owners, df_notes, df_meta = _ticker_strings(filings)
    if df_str.empty:
        return pd.DataFrame(), empty_footnotes(), pd.DataFrame()
    df_kept, df_rejected = _screen_edgar_rows(df_str, df_owners, df_meta, universe, identity)
    df_kept_notes = filter_footnotes(df_notes, set(df_kept["accession_number"])) if not df_kept.empty else empty_footnotes()
    return df_kept, df_kept_notes, df_rejected


def build_ticker_insider_edgar(
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None = None,
    done_accessions: frozenset[str] = frozenset(),
    universe: Sequence[str],
    identity: Identity,
    scan_through: pd.Timestamp,
    scope: EdgarScope,
    excluded: list[pd.DataFrame],
    rescan_stored: bool = False,
) -> dict[Table, pd.DataFrame]:
    """One ticker's EDGAR transaction and footnote frames; its rejected in-scope rows are appended
    to the run's `excluded` collector.

    `rescan_stored` (a `--full` run) ignores `done_accessions` and re-reads stored filings.
    """
    fetched_at = pd.Timestamp.now(tz="UTC").tz_localize(None)
    filings = insider_filings(
        ticker, cik, since=since, through=scan_through, done_accessions=frozenset() if rescan_stored else done_accessions, scope=scope
    )
    df_kept, df_notes, df_rejected = _edgar_frames(filings, universe=universe, identity=identity)
    if not df_rejected.empty:
        excluded.append(exclusion_rows(df_rejected))

    df_rows = pd.DataFrame(columns=_EDGAR_COLUMNS)
    if not df_kept.empty:
        df_stamped = df_kept.assign(fetched_at=fetched_at)
        df_rows = df_stamped[[column for column in _EDGAR_COLUMNS if column in df_stamped.columns]].drop_duplicates(subset=INSIDER_KEY, keep="last")
    return {Tables.insider_transactions: df_rows, Tables.insider_footnotes: df_notes}


def listing_since(context: Context, years_history: int) -> pd.Timestamp:
    """The EDGAR listing start: the latest stored `filing_date` minus `_LISTING_OVERLAP_DAYS`, or
    `years_history` back from today on an empty table."""
    latest = context.store.max_date(Tables.insider_transactions, "filing_date")
    if latest is None:
        return pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    return latest - pd.Timedelta(days=_LISTING_OVERLAP_DAYS)


def replace_zip_accessions(context: Context, since: pd.Timestamp) -> int:
    """Finish EDGAR's replacement of zip-sourced accessions filed since `since`: stamp each
    accession's stored zip `quarter` on its EDGAR rows that lack one (rows EDGAR added beyond the
    zip's keys), then delete the zip rows of every accession EDGAR now holds. Returns the rows deleted."""
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
) -> None:
    """List and save EDGAR ownership filings from `listing_since`, skipping accessions stored from
    EDGAR unless `full`; afterwards, even on failure, log the run's exclusions and replace the
    zip rows of every re-read accession."""
    identity = load_identity(context)
    since = listing_since(context, years_history)
    excluded: list[pd.DataFrame] = []
    fetch = EdgarFetch(
        desc="insider Forms 3/4/5 (EDGAR)",
        tables=(Tables.insider_transactions, Tables.insider_footnotes),
        build=partial(
            build_ticker_insider_edgar,
            universe=tickers,
            identity=identity,
            scan_through=pd.Timestamp.today().normalize(),
            excluded=excluded,
            rescan_stored=full,
        ),
        identity_aware=False,
        listing_since=since,
        done_where={"source": "edgar"},
    )
    try:
        run_edgar_fetch(context, tickers, years_history, fetch, full=full)
    finally:
        log_exclusions(_LOG, "EDGAR run", excluded)
        replace_zip_accessions(context, since)
